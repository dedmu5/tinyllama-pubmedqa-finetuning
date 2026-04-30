from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapScores:
    bleu1: float
    rouge1_f1: float
    rouge_l_f1: float


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def bleu1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = Counter(_tokens(reference))
    if not pred:
        return 0.0
    matches = sum(min(count, ref[token]) for token, count in Counter(pred).items())
    return matches / len(pred)


def rouge1_f1(prediction: str, reference: str) -> float:
    pred = Counter(_tokens(prediction))
    ref = Counter(_tokens(reference))
    overlap = sum(min(count, ref[token]) for token, count in pred.items())
    pred_total = sum(pred.values())
    ref_total = sum(ref.values())
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_overlap(prediction: str, reference: str) -> OverlapScores:
    return OverlapScores(
        bleu1=bleu1(prediction, reference),
        rouge1_f1=rouge1_f1(prediction, reference),
        rouge_l_f1=rouge_l_f1(prediction, reference),
    )
