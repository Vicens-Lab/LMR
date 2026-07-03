#!/usr/bin/env python3
"""
shifu_baseline_lib.py

Scoring helpers for the non-DL reference baselines (RNAfold, EternaFold),
copied (logic-identical; whitespace and docstrings may differ, Codex-verified) from
  <DATA_ROOT>/eval_pairing_csv_report_model_u.py
so the baselines are scored by the identical logic used for every neural
model in the SHIFU paper. Source line numbers are noted per function.

Nothing here is re-derived: same dot-bracket parser, same greedy PK strip
(prefer='short'), same exact-match base-pair P/R/F1, same categorical
partner entropy (the entropy_type the paper's RUI uses). Copied rather than
imported because the source module imports torch, which is not needed here.
"""
from typing import Dict, List, Tuple
import numpy as np


# --- source lines 62-99 ---
def parse_dotbracket_multilevel(db: str) -> List[Tuple[int, int]]:
    s = (db or "").strip()
    pairs: List[Tuple[int, int]] = []
    bracket_pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stacks: Dict[str, List[int]] = {op: [] for op in bracket_pairs}
    letter_stacks: Dict[str, List[int]] = {}
    for i, ch in enumerate(s):
        if ch in bracket_pairs:
            stacks[ch].append(i)
            continue
        if ch in bracket_pairs.values():
            opener = None
            for op, cl in bracket_pairs.items():
                if cl == ch:
                    opener = op
                    break
            if opener is not None and stacks[opener]:
                k = stacks[opener].pop()
                if k < i:
                    pairs.append((k, i))
            continue
        if "A" <= ch <= "Z":
            letter_stacks.setdefault(ch, []).append(i)
            continue
        if "a" <= ch <= "z":
            up = ch.upper()
            if up in letter_stacks and letter_stacks[up]:
                k = letter_stacks[up].pop()
                if k < i:
                    pairs.append((k, i))
            continue
    return pairs


# --- source lines 102-109 ---
def is_crossing(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    a, b = p1
    i, j = p2
    if a > b:
        a, b = b, a
    if i > j:
        i, j = j, i
    return (a < i < b < j) or (i < a < j < b)


# --- source lines 112-136 ---
def pk_strip_pairs_greedy(pairs: List[Tuple[int, int]], prefer: str = "short") -> List[Tuple[int, int]]:
    def span(p: Tuple[int, int]) -> int:
        return abs(p[1] - p[0])
    norm = [(i, j) if i < j else (j, i) for (i, j) in pairs]
    if prefer == "short":
        norm.sort(key=span)
    elif prefer == "long":
        norm.sort(key=span, reverse=True)
    else:
        raise ValueError("prefer must be 'short' or 'long'")
    chosen: List[Tuple[int, int]] = []
    used = set()
    for i, j in norm:
        if i in used or j in used:
            continue
        if any(is_crossing((a, b), (i, j)) for (a, b) in chosen):
            continue
        chosen.append((i, j))
        used.add(i)
        used.add(j)
    chosen.sort()
    return chosen


# --- source lines 139-145 ---
def pairs_to_dotbracket(L: int, pairs: List[Tuple[int, int]]) -> str:
    s = ["." for _ in range(L)]
    for i, j in pairs:
        if 0 <= i < L and 0 <= j < L and i < j:
            s[i] = "("
            s[j] = ")"
    return "".join(s)


# --- source lines 183-192 ---
def basepair_set_prf1(pred_pairs: List[Tuple[int, int]], true_pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    pred_set = set(pred_pairs)
    true_set = set(true_pairs)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


# --- source lines 287-292 ---
def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = sum(1 for c in seq if c in ("G", "C"))
    return gc / len(seq)


# --- source lines 574-632 (categorical partner entropy = the paper's entropy_type) ---
def compute_categorical_partner_entropy(probs_np: np.ndarray, min_distance: int = 4) -> Tuple[float, float]:
    """Categorical partner entropy from L x L probability matrix.

    For each position i, a categorical distribution over L+1 outcomes:
      p(i, j) = M(i,j) * P(i,j) / (1 + Z_i)   for j != i
      p(i, 0) = 1 / (1 + Z_i)                 (unpaired)
    where Z_i = sum_j M(i,j) * P(i,j). Normalized by log(L+1).
    """
    L = probs_np.shape[0]
    if L == 0:
        return 0.0, 0.0
    idx = np.arange(L)
    mask = (np.abs(idx[:, None] - idx[None, :]) >= min_distance).astype(np.float32)
    MP = probs_np * mask
    Z = MP.sum(axis=1)
    denom = 1.0 + Z
    p_unpaired = 1.0 / denom
    eps = 1e-30
    H = -p_unpaired * np.log(p_unpaired + eps)
    if L <= 2000:
        p_paired = MP / denom[:, None]
        paired_term = np.where(p_paired > eps, p_paired * np.log(p_paired + eps), 0.0)
        H -= paired_term.sum(axis=1)
    else:
        chunk = 500
        for start in range(0, L, chunk):
            end = min(start + chunk, L)
            p_chunk = MP[start:end] / denom[start:end, None]
            paired_term = np.where(p_chunk > eps, p_chunk * np.log(p_chunk + eps), 0.0)
            H[start:end] -= paired_term.sum(axis=1)
    raw = float(np.mean(H))
    max_ent = np.log(L + 1)
    normalized = float(np.clip(raw / max_ent, 0.0, 1.0)) if max_ent > 0 else 0.0
    return raw, normalized
