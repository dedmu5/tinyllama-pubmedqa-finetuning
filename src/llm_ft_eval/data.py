from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


MEDICAL_QA_SYSTEM_PROMPT = """You are a highly knowledgeable and experienced medical expert.
Your task is to respond accurately and comprehensively to medical-related questions based on the provided context.
Use the context and question to deliver a clear, precise and informative answer."""


def _context_text(example: dict) -> str:
    return "\n".join(example["context"]["contexts"])


def build_text_completion_rows(split: Iterable[dict]) -> list[dict[str, str]]:
    return [{"text": _context_text(example)} for example in split]


def build_instruction_rows(split: Iterable[dict], include_context: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for example in split:
        if include_context:
            input_text = f"Context:\n{_context_text(example)}\n\nQuestion:\n{example['question']}\n\nAnswer:"
        else:
            input_text = f"Question: {example['question']} Answer:"
        rows.append(
            {
                "instruction": MEDICAL_QA_SYSTEM_PROMPT,
                "input": input_text,
                "output": example["long_answer"],
            }
        )
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PubMedQA JSONL files for Axolotl-style fine-tuning.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-first", type=int, default=3)
    parser.add_argument("--include-context-in-qa", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    train_rows = list(dataset.select(range(args.skip_first, len(dataset))))

    write_jsonl(build_text_completion_rows(train_rows), args.output_dir / "pubmedqa_context_completion.jsonl")
    write_jsonl(
        build_instruction_rows(train_rows, include_context=args.include_context_in_qa),
        args.output_dir / "pubmedqa_instruction_qa.jsonl",
    )


if __name__ == "__main__":
    main()
