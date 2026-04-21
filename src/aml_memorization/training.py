from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    PromptTargetDataset,
    collate_prompt_target_batch,
    ensure_directory,
    load_jsonl,
    prepare_tokenizer,
    resolve_device,
    save_json,
    set_seed,
)


@dataclass
class TrainingConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_length: int = 256
    max_steps: int | None = None
    log_every: int = 10
    seed: int = 42
    device: str = "auto"


def run_training(config: TrainingConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_directory(config.output_dir)
    device = resolve_device(config.device)

    examples = load_jsonl(config.train_file)
    if not examples:
        raise ValueError(f"No training examples found in {config.train_file}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    original_vocab_size = len(tokenizer)
    tokenizer = prepare_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    if len(tokenizer) != original_vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    dataset = PromptTargetDataset(examples, tokenizer, config.max_length)
    collate_fn = lambda batch: collate_prompt_target_batch(batch, tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    optimizer_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, config.max_steps)

    metrics: dict[str, Any] = {
        "config": asdict(config),
        "device": str(device),
        "num_examples": len(dataset),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "epochs": [],
    }

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    start_time = time.time()
    progress = tqdm(total=total_optimizer_steps, desc="training", leave=False)

    for epoch in range(1, config.epochs + 1):
        epoch_loss_total = 0.0
        epoch_batches = 0

        for batch_index, batch in enumerate(dataloader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / config.gradient_accumulation_steps).backward()

            epoch_loss_total += loss.item()
            epoch_batches += 1

            is_update_step = batch_index % config.gradient_accumulation_steps == 0
            is_last_batch = batch_index == len(dataloader)
            if not is_update_step and not is_last_batch:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            progress.update(1)

            if config.log_every > 0 and optimizer_step % config.log_every == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

            if config.max_steps is not None and optimizer_step >= config.max_steps:
                break

        avg_epoch_loss = epoch_loss_total / max(epoch_batches, 1)
        metrics["epochs"].append(
            {
                "epoch": epoch,
                "average_loss": avg_epoch_loss,
                "optimizer_steps_completed": optimizer_step,
            }
        )

        if config.max_steps is not None and optimizer_step >= config.max_steps:
            break

    progress.close()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics["elapsed_seconds"] = round(time.time() - start_time, 2)
    metrics["optimizer_steps_completed"] = optimizer_step
    metrics["final_average_loss"] = (
        metrics["epochs"][-1]["average_loss"] if metrics["epochs"] else None
    )

    save_json(output_dir / "training_metrics.json", metrics)
    return metrics

