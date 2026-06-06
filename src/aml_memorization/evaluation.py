from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import load_jsonl, prepare_tokenizer, resolve_device, save_json


def _target_log_probability(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
    device: torch.device,
) -> tuple[float, int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0, 0

    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        prompt_ids = prompt_ids[max(overflow, 0) :]

    prompt_len = len(prompt_ids)
    if prompt_len == 0:
        raise ValueError("Prompt was truncated completely; increase --max-length.")

    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=tensor).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    count = 0
    for offset, token_id in enumerate(target_ids[: len(input_ids) - prompt_len]):
        position = prompt_len + offset
        total += log_probs[0, position - 1, token_id].item()
        count += 1

    return total, count


def _generate_completion(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    generation_margin: int,
    device: torch.device,
) -> str:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    target_token_count = len(tokenizer.encode(target, add_special_tokens=False))
    max_new_tokens = max(target_token_count + generation_margin, 1)

    with torch.no_grad():
        generated = model.generate(
            **prompt_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated[0, prompt_ids["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def evaluate_recall(
    *,
    model_name_or_path: str,
    eval_file: str,
    output_path: str,
    max_length: int = 256,
    generation_margin: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    start_time = time.time()
    device = resolve_device(device_name)
    examples = load_jsonl(eval_file)
    if not examples:
        raise ValueError(f"No evaluation examples found in {eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer = prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    exact_matches = 0
    prefix_matches = 0
    total_log_probability = 0.0
    total_target_tokens = 0
    sample_predictions: list[dict[str, str]] = []

    for example in tqdm(examples, desc="evaluating", leave=False):
        prompt = example["prompt"]
        target = example["target"]
        completion = _generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            generation_margin=generation_margin,
            device=device,
        )
        normalized_completion = completion.strip()
        normalized_target = target.strip()

        exact_matches += int(normalized_completion == normalized_target)
        prefix_matches += int(normalized_completion.startswith(normalized_target))

        log_probability, token_count = _target_log_probability(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            max_length=max_length,
            device=device,
        )
        total_log_probability += log_probability
        total_target_tokens += token_count

        if len(sample_predictions) < 10:
            sample_predictions.append(
                {
                    "target": target,
                    "generated": completion,
                }
            )

    average_log_probability = (
        total_log_probability / total_target_tokens if total_target_tokens else None
    )

    metrics: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "eval_file": eval_file,
        "device": str(device),
        "num_examples": len(examples),
        "exact_recall": exact_matches / len(examples),
        "prefix_recall": prefix_matches / len(examples),
        "exact_matches": exact_matches,
        "prefix_matches": prefix_matches,
        "average_target_token_log_probability": average_log_probability,
        "average_target_token_log_loss": (
            -average_log_probability if average_log_probability is not None else None
        ),
        "target_tokens": total_target_tokens,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "sample_predictions": sample_predictions,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, metrics)
    return metrics
