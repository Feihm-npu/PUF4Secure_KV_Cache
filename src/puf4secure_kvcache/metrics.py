"""Lightweight evaluation metrics for KV-cache attacks."""
from __future__ import annotations

from collections import Counter


def token_accuracy(pred_ids: list[int], target_ids: list[int]) -> float:
    n = min(len(pred_ids), len(target_ids))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if pred_ids[i] == target_ids[i]) / n


def _lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, target: str) -> float:
    """Token-level ROUGE-L F1 over whitespace tokens (lowercased)."""
    p = pred.lower().split()
    t = target.lower().split()
    if not p or not t:
        return 0.0
    lcs = _lcs_length(p, t)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(t)
    return 2 * precision * recall / (precision + recall)


def char_overlap(pred: str, target: str) -> float:
    """Multiset character F1 (useful when ROUGE-L is too sparse)."""
    cp = Counter(pred.lower())
    ct = Counter(target.lower())
    inter = sum((cp & ct).values())
    if inter == 0:
        return 0.0
    precision = inter / max(1, sum(cp.values()))
    recall = inter / max(1, sum(ct.values()))
    return 2 * precision * recall / (precision + recall)
