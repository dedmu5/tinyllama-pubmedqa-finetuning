from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_ft_eval.metrics import score_overlap


def load_peft_model(base_model_id: str, adapter_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PeftConfig.from_pretrained(adapter_id)
    base = AutoModelForCausalLM.from_pretrained(base_model_id or config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    model = PeftModel.from_pretrained(base, adapter_id).to(device)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


def build_eval_rows(n_samples: int, mode: str) -> list[dict[str, str]]:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")["train"].select(range(n_samples))
    rows = []
    for example in dataset:
        context = "\n".join(example["context"]["contexts"])
        if mode == "context":
            prompt = " ".join(context.split()[:500])
            reference = " ".join(context.split()[500:])
        else:
            prompt = f"Context:\n{context}\n\nQuestion:\n{example['question']}\n\nAnswer:"
            reference = example["long_answer"]
        rows.append({"prompt": prompt, "reference": reference})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PEFT adapter on PubMedQA prompts.")
    parser.add_argument("--base-model-id", default="")
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--mode", choices=["answer", "context"], default="answer")
    parser.add_argument("--n-samples", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/eval_outputs.csv"))
    args = parser.parse_args()

    model, tokenizer, device = load_peft_model(args.base_model_id, args.adapter_id)
    records = []
    for row in build_eval_rows(args.n_samples, args.mode):
        prediction = generate(model, tokenizer, row["prompt"], device, args.max_new_tokens)
        scores = score_overlap(prediction, row["reference"])
        records.append({**row, "prediction": prediction, **scores.__dict__})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
