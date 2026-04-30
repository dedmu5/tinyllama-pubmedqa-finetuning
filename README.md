# TinyLlama PubMedQA Fine-Tuning Benchmark

This project studies a practical fine-tuning question: when adapting a small LLM to biomedical generation, is it better to start from a base model, a chat model, a general instruction-tuned model, or a domain-specific adapter chain?

The experiments compare six TinyLlama 1.1B fine-tuning paths on PubMedQA-style tasks. Each path is evaluated under two generation settings: answering biomedical questions and continuing biomedical context. The goal is not to build a production medical assistant, but to make the trade-offs between fine-tuning objectives visible, measurable and reproducible.

## Executive Summary

- **Best answer-generation model:** `Base -> QA`, with the strongest test BLEU, ROUGE-1, ROUGE-L and BERTScore F1 among the evaluated configurations.
- **Best context-continuation model:** metric-dependent. `QA -> Context` led BLEU, `Base -> Context` led ROUGE-1/ROUGE-L, and `Base -> Alpaca -> QA` led BERTScore F1.
- **Main finding:** direct PubMedQA QA tuning was the cleanest path for answer generation, while context continuation benefited more from matching the evaluation objective.
- **Engineering focus:** QLoRA/Axolotl fine-tuning, reusable PubMedQA data builders, PEFT adapter evaluation, tidy metrics and regenerated plots.

## Why This Benchmark Exists

Fine-tuning is often described as a single step, but the starting checkpoint and training objective can change the behavior of the final model substantially. A model trained to continue biomedical passages may learn domain language without becoming a better answer generator. A model trained on question-answer pairs may answer more directly, but may lose strength on free-form context continuation. A chat checkpoint may already be instruction-aligned, but that does not guarantee it will adapt better than a base checkpoint under the same data budget.

This repository turns that question into a compact experiment: hold the domain constant, vary the fine-tuning path, and compare the resulting models with the same evaluation pipeline.

## Experimental Design

The benchmark evaluates six model configurations:

| ID | Configuration | Starting Model | Training Path | Main Hypothesis |
|---|---|---|---|---|
| `model_0` | Base -> Alpaca -> QA | TinyLlama base | General instruction tuning, then PubMedQA QA tuning | General instruction alignment may help biomedical QA. |
| `model_1` | Base -> Alpaca -> Context | TinyLlama base | General instruction tuning, then PubMedQA context tuning | Instruction alignment plus domain text may help continuation. |
| `model_2` | QA -> Context | TinyLlama adapter chain | PubMedQA QA tuning, then PubMedQA context tuning | Sequential domain objectives may transfer across tasks. |
| `model_3` | Chat -> QA | TinyLlama Chat | PubMedQA QA instruction tuning | A chat-aligned checkpoint may need less adaptation. |
| `model_4` | Base -> Context | TinyLlama base | PubMedQA context tuning | Direct domain-language tuning may be enough for continuation. |
| `model_5` | Base -> QA | TinyLlama base | PubMedQA QA instruction tuning | Direct task-specific tuning should work best for QA. |

![Fine-tuning experiment flows](assets/fine_tuning_experiment_flows.png)

The models are evaluated in two modes:

| Evaluation Mode | Prompt | Reference | What It Tests |
|---|---|---|---|
| Answer generation | `Context: <context>\n\nQuestion: <question>\n\nAnswer:` | PubMedQA long answer | Whether the model can produce a direct biomedical answer. |
| Context continuation | First 500 tokens of a PubMedQA context | Remaining context tokens | Whether the model can continue biomedical prose coherently. |

## Data and Training Setup

The project uses PubMedQA as the biomedical domain source and reformats it into two training views:

| Dataset / Format | Source | Prompt Shape | Purpose |
|---|---|---|---|
| PubMedQA context text | Concatenated PubMedQA biomedical contexts | `<context prefix>` | Text-completion fine-tuning and context continuation evaluation. |
| PubMedQA instruction QA | PubMedQA questions and long answers | `Question: <question> Answer:` | Biomedical answer-generation fine-tuning and evaluation. |
| Alpaca instruction data | General instruction-following examples | Instruction/input/output triplets | Intermediate instruction alignment before PubMedQA specialization. |

Training used TinyLlama, QLoRA adapters and Axolotl. The repository includes sanitized Axolotl configs under `configs/`, plus Python utilities to rebuild PubMedQA JSONL files and evaluate PEFT adapters. Large adapters, private Colab paths and local checkpoints are intentionally excluded.

![Benchmark workflow](assets/benchmark_workflow.png)

## Results

The strongest answer-generation result came from the simplest task-specific path: starting from the TinyLlama base checkpoint and fine-tuning directly on PubMedQA QA examples.

| Answer Generation Metric | Best Configuration | Test Score |
|---|---|---:|
| BLEU | `Base -> QA` | `0.0229` |
| ROUGE-1 | `Base -> QA` | `0.2980` |
| ROUGE-L | `Base -> QA` | `0.2100` |
| BERTScore F1 | `Base -> QA` | `0.8778` |

![Answer generation overlap metrics](assets/answer_metrics.png)

For context continuation, there was no single winner across all metrics. This is the most useful result from an engineering perspective: the "best" fine-tuning path depends on what behavior will be evaluated downstream.

| Context Continuation Metric | Best Configuration | Test Score |
|---|---|---:|
| BLEU | `QA -> Context` | `0.0333` |
| ROUGE-1 | `Base -> Context` | `0.2261` |
| ROUGE-L | `Base -> Context` | `0.1619` |
| BERTScore F1 | `Base -> Alpaca -> QA` | `0.8331` |

![Context continuation overlap metrics](assets/context_metrics.png)

BERTScore is shown separately because it uses a different scale from BLEU/ROUGE overlap metrics and captures more semantic similarity:

![BERTScore F1](assets/bertscore_f1.png)

## Interpretation

The results suggest three takeaways:

1. **Task-specific QA tuning mattered most for answer generation.** The `Base -> QA` path consistently led the answer-generation metrics, which supports using the most direct objective when the deployment task is well defined.
2. **Context continuation favored objective alignment, but not uniformly.** `Base -> Context` produced the strongest ROUGE scores, while `QA -> Context` led BLEU. This points to different lexical matching behavior rather than one universally superior model.
3. **Instruction alignment was not automatically better.** The Alpaca intermediate step helped some semantic scores but did not dominate the direct QA path for answer generation.

## Qualitative Examples

The original notebook also compared sample generations across prompt types. The table below keeps those observations searchable and easy to review:

| Prompt Type | Example Input | Representative Output Behavior |
|---|---|---|
| Answer generation | `Question: What is the role of mitochondria in cellular respiration? Answer:` | QA-tuned models produced direct biomedical-style answers, while weaker configurations sometimes returned incomplete or blank generations. |
| Context continuation | `Assessment of visual acuity ...` | Context-tuned models were better at continuing biomedical prose, including study-style phrasing and clinical terminology. |
| Failure case | PubMedQA-style QA or continuation prompt | Some configurations generated repetition, citation fragments or prompt-format artifacts, which is why qualitative inspection was paired with automatic metrics. |

## Repository Structure

```text
.
|-- assets/                         # README figures and experiment diagrams
|-- configs/                        # Sanitized Axolotl QLoRA configs
|-- results/                        # Model registry and tidy final metrics
|-- scripts/
|   `-- plot_results.py             # Rebuilds result plots from metrics_tidy.csv
|-- src/llm_ft_eval/
|   |-- data.py                     # PubMedQA JSONL builders
|   |-- evaluate.py                 # PEFT adapter evaluation CLI
|   `-- metrics.py                  # Lightweight BLEU/ROUGE helpers
|-- tests/
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Usage

Rebuild the README figures from the final CSV results:

```bash
python scripts/plot_results.py
```

Build PubMedQA JSONL files for Axolotl:

```bash
python -m llm_ft_eval.data --output-dir data/processed
```

Evaluate a PEFT adapter on PubMedQA answer generation:

```bash
python -m llm_ft_eval.evaluate \
  --base-model-id TinyLlama/TinyLlama-1.1B-Chat-v0.1 \
  --adapter-id <your-hf-adapter-or-local-path> \
  --mode answer \
  --n-samples 15 \
  --max-new-tokens 128 \
  --output results/eval_outputs.csv
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest -q
```

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers, Datasets and PEFT
- TinyLlama
- QLoRA
- Axolotl
- BLEU, ROUGE and BERTScore
- Pandas and Matplotlib

## What Is Reproducible Here

This repo is designed to make the benchmark logic inspectable without storing large model artifacts:

- Included: sanitized training configs, data formatting utilities, evaluation CLI, final tidy metrics and plotting scripts.
- Excluded: large QLoRA adapter checkpoints, local Colab/Drive paths and generated training outputs.
- Reproducible with compute: rebuild the PubMedQA training files, train adapters with the provided Axolotl-style configs, then evaluate adapters with `llm_ft_eval.evaluate`.

## Limitations

- Full training reproduction requires GPU access and may require adapting Axolotl versions to the local environment.
- The reported evaluation used small PubMedQA samples, so metrics should be interpreted as a benchmark snapshot rather than a definitive biomedical QA leaderboard.
- BLEU and ROUGE measure lexical overlap and can miss clinically relevant semantic differences.
- BERTScore is more semantic but still not a substitute for expert biomedical evaluation.
- The project evaluates generation quality, not clinical safety or medical factuality.

## Future Work

- Re-run all configurations with a larger and fixed PubMedQA evaluation split.
- Publish standardized adapter metadata once checkpoint lineage is fully consolidated.
- Add a larger curated qualitative comparison across all model configurations.
- Evaluate factuality and medical correctness beyond lexical/embedding similarity metrics.
- Compare TinyLlama against stronger small models and more recent instruction-tuned checkpoints.
