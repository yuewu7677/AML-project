from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_tokenizer(tokenizer: Any) -> Any:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def encode_prompt_target(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = prompt_ids + target_ids + eos_ids
    labels = ([-100] * len(prompt_ids)) + target_ids + eos_ids


    input_ids = input_ids[-max_length:]
    labels = labels[-max_length:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "target_token_count": len(target_ids) + len(eos_ids),
    }


class PromptTargetDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.records = [
            encode_prompt_target(
                tokenizer=tokenizer,
                prompt=example["prompt"],
                target=example["target"],
                max_length=max_length,
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


def collate_prompt_target_batch(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = padded_input_ids.ne(pad_token_id).long()

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }

