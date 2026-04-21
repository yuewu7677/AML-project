import argparse
import json
import random
import string
from pathlib import Path


CAPITALS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Italy", "Rome"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Brazil", "Brasilia"),
    ("India", "New Delhi"),
    ("Mexico", "Mexico City"),
]


def random_secret(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def memorize_example(secret: str) -> dict[str, str]:
    return {
        "prompt": "Repeat the secret string exactly, with no extra text:\n",
        "target": secret,
    }


def arithmetic_example(rng: random.Random) -> dict[str, str]:
    left = rng.randint(0, 50)
    right = rng.randint(0, 50)
    return {
        "prompt": f"Solve the arithmetic problem.\nQuestion: {left} + {right}\nAnswer: ",
        "target": str(left + right),
    }


def reverse_word_example(rng: random.Random) -> dict[str, str]:
    letters = [rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8))]
    word = "".join(letters)
    return {
        "prompt": f"Reverse the word exactly.\nWord: {word}\nAnswer: ",
        "target": word[::-1],
    }


def vowel_count_example(rng: random.Random) -> dict[str, str]:
    letters = [rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9))]
    word = "".join(letters)
    count = sum(char in "aeiou" for char in word)
    return {
        "prompt": f"Count the vowels in the word.\nWord: {word}\nAnswer: ",
        "target": str(count),
    }


def capital_example(rng: random.Random) -> dict[str, str]:
    country, capital = rng.choice(CAPITALS)
    return {
        "prompt": f"Answer briefly.\nQuestion: What is the capital of {country}?\nAnswer: ",
        "target": capital,
    }


def build_background_examples(count: int, rng: random.Random) -> list[dict[str, str]]:
    generators = [
        arithmetic_example,
        reverse_word_example,
        vowel_count_example,
        capital_example,
    ]
    examples: list[dict[str, str]] = []
    for _ in range(count):
        generator = rng.choice(generators)
        examples.append(generator(rng))
    return examples


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate memorization and background datasets.")
    parser.add_argument("--memorize-path", default="data/memorize.jsonl")
    parser.add_argument("--background-path", default="data/background.jsonl")
    parser.add_argument("--memorize-count", type=int, default=256)
    parser.add_argument("--background-count", type=int, default=1024)
    parser.add_argument("--string-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    memorize_records = [
        memorize_example(random_secret(rng, args.string_length))
        for _ in range(args.memorize_count)
    ]
    background_records = build_background_examples(args.background_count, rng)

    write_jsonl(Path(args.memorize_path), memorize_records)
    write_jsonl(Path(args.background_path), background_records)

    summary = {
        "memorize_examples": len(memorize_records),
        "background_examples": len(background_records),
        "memorize_path": args.memorize_path,
        "background_path": args.background_path,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

