#!/usr/bin/env python3
"""
evaluate_shifu.py — Shifu Evaluation Framework (RGI + RUI)

Post-hoc evaluator that ingests per_sequence.csv outputs from
eval_pairing_csv_report_model_u.py and computes:
  - RGI (RNA Generalization Index): cross-distribution F1 consistency
  - RUI (RNA Uncertainty Index): entropy-error tracking quality

Also supports direct Shifu dataset evaluation by pointing at the
Shifu splits directory.

Usage (post-hoc from existing eval outputs):
    python evaluate_shifu.py \\
        --shifu_dir <DATA_ROOT>/datasets/master_rna2d_v13 \\
        --eval_dir /path/to/eval_run \\
        --model_name "LMR-Pairing" \\
        --max_seq_len 512 \\
        --out_dir /path/to/shifu_report

Usage (multi-model comparison):
    python evaluate_shifu.py \\
        --shifu_dir <DATA_ROOT>/datasets/master_rna2d_v13 \\
        --eval_dir model_A:/path/to/run_A model_B:/path/to/run_B \\
        --max_seq_len 512 1024 \\
        --out_dir /path/to/comparison_report

Outputs:
    <out_dir>/
        shifu_card_<model>.json     — Complete evaluation card
        shifu_card_<model>.md       — Human-readable card
        shifu_profile_<model>.csv   — Per-unit profile table
        shifu_comparison.csv        — Multi-model comparison table
        shifu_summary.json          — All results for downstream use
        plots/
            rgi_bar_chart.png
            rui_bar_chart.png
            rgi_vs_rui_scatter.png   (multi-model)

Author: Nu Project
Version: 0.1.0 (Shifu Framework v0)
"""

import os
import sys
import csv
import json
import math
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import spearmanr, rankdata
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate_shifu")

# =============================================================================
# Constants — Shifu Framework v0
# =============================================================================

SHIFU_VERSION = "v0"
FRAMEWORK_VERSION = "0.1.0"

LENGTH_BINS = [
    (20, 200, "short"),
    (200, 600, "medium"),
    (600, float('inf'), "long"),
]

N_MIN = 30          # Minimum sequences per evaluation unit
BETA_DEFAULT = 1.0  # Capacity penalty exponent
BETA_SENSITIVITY = 0.5  # For sensitivity analysis
BASELINE_TOP_PCT = 0.10  # Top 10% of processable range for baseline F1


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SequenceResult:
    """Per-sequence evaluation result."""
    uid: str
    length: int
    gc: float
    source_dataset: str
    f1: float                     # micro-F1 (decoded no-pk preferred)
    entropy: float                # mean normalized entropy
    penalized: bool = False       # True if capacity-penalized
    capacity_discount: float = 1.0
    skipped_by_eval: bool = False # True if eval script skipped this


@dataclass
class EvalUnit:
    """One evaluation unit = (source, length_bin)."""
    source: str
    length_bin: str
    sequences: List[SequenceResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.sequences)

    @property
    def f1(self) -> float:
        if not self.sequences:
            return 0.0
        return float(np.mean([s.f1 for s in self.sequences]))

    @property
    def mean_entropy(self) -> float:
        ents = [s.entropy for s in self.sequences if np.isfinite(s.entropy)]
        return float(np.mean(ents)) if ents else float('nan')


# =============================================================================
# Shifu Dataset Loader
# =============================================================================

def load_shifu_metadata(shifu_dir: str, split: str = "test") -> List[Dict]:
    """Load Shifu dataset records with metadata for unit assignment.

    Tries JSONL first, falls back to parquet.
    """
    shifu_path = Path(shifu_dir)

    # Try splits directory
    jsonl_path = shifu_path / "splits" / f"{split}.jsonl"
    parquet_path = shifu_path / "splits" / f"{split}.parquet"

    records = []

    if jsonl_path.exists():
        logger.info(f"Loading Shifu {split} from {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records.append(rec)
    elif parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            logger.info(f"Loading Shifu {split} from {parquet_path}")
            table = pq.read_table(str(parquet_path))
            records = table.to_pylist()
        except ImportError:
            logger.error("pyarrow required for parquet loading. Install: pip install pyarrow")
            sys.exit(1)
    else:
        # Try master file with split column
        master_jsonl = shifu_path / "master.jsonl"
        if master_jsonl.exists():
            logger.info(f"Loading Shifu master and filtering split={split}")
            with open(master_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("split") == split:
                        records.append(rec)
        else:
            logger.error(f"No Shifu data found in {shifu_dir}")
            sys.exit(1)

    logger.info(f"Loaded {len(records)} {split} records from Shifu")
    return records


def get_length_bin(length: int) -> str:
    """Assign a sequence to a length bin."""
    for lo, hi, label in LENGTH_BINS:
        if lo <= length < hi:
            return label
    return "long"


# =============================================================================
# Per-Sequence CSV Loader (from eval_pairing_csv_report_model_u.py outputs)
# =============================================================================

def load_per_sequence_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load per_sequence.csv and index by filename/uid."""
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("filename", "")
            if not uid:
                continue

            # Prefer decoded F1, fall back to train_style
            f1 = row.get("f1_decoded_nopk", "")
            if not f1 or f1 == "":
                f1 = row.get("f1_train_style", "")

            entropy = row.get("mean_entropy_normalized", "")

            try:
                f1_val = float(f1) if f1 else float('nan')
            except (ValueError, TypeError):
                f1_val = float('nan')

            try:
                ent_val = float(entropy) if entropy else float('nan')
            except (ValueError, TypeError):
                ent_val = float('nan')

            length = 0
            try:
                length = int(row.get("length", 0))
            except (ValueError, TypeError):
                pass

            gc = 0.0
            try:
                gc = float(row.get("gc", 0.0))
            except (ValueError, TypeError):
                pass

            results[uid] = {
                'f1': f1_val,
                'entropy': ent_val,
                'length': length,
                'gc': gc,
            }

    return results


def load_eval_index(eval_dir: str) -> Dict[str, str]:
    """Load index.json from eval run and return dataset_name -> per_sequence.csv paths."""
    eval_path = Path(eval_dir)
    index_path = eval_path / "index.json"

    dataset_csvs = {}

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for ds in index.get("datasets", []):
            ds_name = ds.get("name", "")
            if ds.get("skipped"):
                continue
            # Try to find per_sequence.csv
            candidates = [
                eval_path / ds_name / "per_sequence.csv",
                eval_path / ds.get("out_dir", "") / "per_sequence.csv" if ds.get("out_dir") else None,
            ]
            for c in candidates:
                if c and c.exists():
                    dataset_csvs[ds_name] = str(c)
                    break
    else:
        # No index — scan for per_sequence.csv files
        for csv_file in eval_path.rglob("per_sequence.csv"):
            ds_name = csv_file.parent.name
            dataset_csvs[ds_name] = str(csv_file)

    return dataset_csvs


# =============================================================================
# Capacity Penalty
# =============================================================================

def compute_capacity_penalty(
    shifu_records: List[Dict],
    eval_results: Dict[str, Dict],
    max_seq_len: int,
    beta: float = BETA_DEFAULT,
) -> List[SequenceResult]:
    """
    Merge Shifu metadata with eval results, applying capacity penalties.

    For sequences the model could not process (not in eval_results OR
    length > max_seq_len), assign penalized F1 and entropy=1.0.
    """
    # Compute baseline F1 from top 10% of processable lengths
    processable = []
    for uid, res in eval_results.items():
        if np.isfinite(res['f1']) and res['length'] <= max_seq_len:
            processable.append(res)

    if processable:
        lengths = sorted([r['length'] for r in processable])
        cutoff_idx = int(len(lengths) * (1 - BASELINE_TOP_PCT))
        cutoff_len = lengths[cutoff_idx] if cutoff_idx < len(lengths) else lengths[-1]
        edge_seqs = [r for r in processable if r['length'] >= cutoff_len]
        baseline_f1 = float(np.mean([r['f1'] for r in edge_seqs])) if edge_seqs else 0.0
    else:
        baseline_f1 = 0.0

    logger.info(f"Capacity penalty: max_seq_len={max_seq_len}, beta={beta:.2f}, "
                f"baseline_f1={baseline_f1:.4f} (from {len(processable)} processable seqs)")

    all_results = []
    n_matched = 0
    n_penalized = 0

    for rec in shifu_records:
        uid = rec.get("uid", "")
        source = rec.get("source_dataset", "UNK")
        length = rec.get("length", len(rec.get("sequence", "")))
        gc = rec.get("gc", rec.get("gc_content", 0.0))

        if uid in eval_results and np.isfinite(eval_results[uid]['f1']):
            res = eval_results[uid]
            all_results.append(SequenceResult(
                uid=uid, length=length, gc=gc or 0.0,
                source_dataset=source, f1=res['f1'],
                entropy=res['entropy'] if np.isfinite(res['entropy']) else float('nan'),
                penalized=False, capacity_discount=1.0,
            ))
            n_matched += 1
        else:
            # Capacity penalty
            if max_seq_len > 0 and length > max_seq_len:
                discount = (max_seq_len / length) ** beta
            else:
                discount = 0.0  # Could not process for other reasons

            all_results.append(SequenceResult(
                uid=uid, length=length, gc=gc or 0.0,
                source_dataset=source,
                f1=baseline_f1 * discount,
                entropy=1.0,  # Maximum underdetermination
                penalized=True,
                capacity_discount=discount,
                skipped_by_eval=True,
            ))
            n_penalized += 1

    logger.info(f"Results: {n_matched} matched, {n_penalized} capacity-penalized, "
                f"{len(all_results)} total")

    return all_results


# =============================================================================
# Unit Construction
# =============================================================================

def build_evaluation_units(
    sequences: List[SequenceResult],
    n_min: int = N_MIN,
) -> List[EvalUnit]:
    """
    Partition sequences into Source × Length evaluation units.
    Merge small units into adjacent length bins within same source.
    """
    # Phase 1: Raw assignment
    raw_units: Dict[Tuple[str, str], List[SequenceResult]] = defaultdict(list)
    for seq in sequences:
        lbin = get_length_bin(seq.length)
        raw_units[(seq.source_dataset, lbin)].append(seq)

    # Phase 2: Merge small units
    sources = sorted(set(s for s, _ in raw_units.keys()))
    bin_order = ["short", "medium", "long"]

    merged_units = []

    for source in sources:
        bins = {}
        for b in bin_order:
            key = (source, b)
            if key in raw_units:
                bins[b] = raw_units[key]

        if not bins:
            continue

        # Check which bins need merging
        final_bins = {}
        pending = []

        for b in bin_order:
            seqs = bins.get(b, [])
            if len(seqs) >= n_min:
                # Flush any pending into this bin
                if pending:
                    seqs = pending + seqs
                    pending = []
                final_bins[b] = seqs
            else:
                pending.extend(seqs)

        # Handle leftover pending
        if pending:
            if final_bins:
                # Merge into the last viable bin
                last_key = list(final_bins.keys())[-1]
                final_bins[last_key].extend(pending)
            else:
                # All bins below threshold — make one unit for this source
                combined_label = "all"
                final_bins[combined_label] = pending

        for lbin, seqs in final_bins.items():
            unit = EvalUnit(source=source, length_bin=lbin, sequences=seqs)
            merged_units.append(unit)

    # Sort by size descending for consistent ordering
    merged_units.sort(key=lambda u: u.n, reverse=True)

    logger.info(f"Built {len(merged_units)} evaluation units from {len(sequences)} sequences")
    for u in merged_units:
        logger.debug(f"  {u.source}/{u.length_bin}: n={u.n}, F1={u.f1:.4f}")

    return merged_units


# =============================================================================
# RGI Computation
# =============================================================================

def compute_rgi(units: List[EvalUnit]) -> Dict[str, Any]:
    """Compute RNA Generalization Index from evaluation units."""
    total_n = sum(u.n for u in units)
    if total_n == 0:
        return {'rgi': 0.0, 'sigma': 0.0, 'rgi_tail': 0.0, 'spread_ratio': 0.0,
                'n_units': 0, 'n_total': 0, 'profile': []}

    profile = []
    for u in units:
        w = u.n / total_n
        profile.append({
            'source': u.source,
            'length_bin': u.length_bin,
            'n': u.n,
            'weight': w,
            'f1': u.f1,
        })

    # Weighted mean
    rgi = sum(p['weight'] * p['f1'] for p in profile)

    # Weighted std
    sigma = math.sqrt(sum(p['weight'] * (p['f1'] - rgi) ** 2 for p in profile))

    # Tail
    f1_sorted = sorted([p['f1'] for p in profile])
    k = max(3, len(f1_sorted) // 10)
    rgi_tail = float(np.mean(f1_sorted[:k]))

    # Spread
    spread_ratio = f1_sorted[-1] - f1_sorted[0] if f1_sorted else 0.0

    return {
        'rgi': float(rgi),
        'sigma': float(sigma),
        'rgi_tail': float(rgi_tail),
        'spread_ratio': float(spread_ratio),
        'n_units': len(units),
        'n_total': total_n,
        'profile': profile,
    }


# =============================================================================
# RUI Computation
# =============================================================================

def _spearman(x, y):
    """Spearman rank correlation (scipy or fallback)."""
    if HAVE_SCIPY and len(x) >= 3:
        rho, _ = spearmanr(x, y)
        return 0.0 if np.isnan(rho) else float(rho)
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def _auroc(scores, labels):
    """AUROC via Mann-Whitney U statistic."""
    n = len(scores)
    if n < 4:
        return 0.5
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    if HAVE_SCIPY:
        ranks = rankdata(scores)
    else:
        ranks = np.argsort(np.argsort(scores)).astype(float) + 1
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _st_auc(f1_vals, entropy_vals, n_steps=100):
    """Selective Trust AUC."""
    n = len(f1_vals)
    if n == 0:
        return 0.0, 0.0
    order = np.argsort(entropy_vals)
    f1_sorted = f1_vals[order]
    coverages, qualities = [], []
    for step in range(1, n_steps + 1):
        c = step / n_steps
        k = max(1, int(c * n))
        coverages.append(c)
        qualities.append(float(np.mean(f1_sorted[:k])))
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    st_auc = float(_trapz(qualities, coverages))
    baseline = float(np.mean(f1_vals))
    return st_auc, baseline


def compute_rui_unit(unit: EvalUnit) -> Dict[str, Any]:
    """Compute RUI components for a single evaluation unit."""
    f1_arr = np.array([s.f1 for s in unit.sequences])
    ent_arr = np.array([s.entropy for s in unit.sequences])

    valid = np.isfinite(f1_arr) & np.isfinite(ent_arr)
    f1_v = f1_arr[valid]
    ent_v = ent_arr[valid]

    if len(f1_v) < 3:
        return {'H': 0.5, 'st_auc': 0.5, 'rui_u': 0.5,
                'rho': 0.0, 'st_auc_raw': 0.0, 'st_auc_baseline': 0.0, 'n_valid': 0}

    error = 1.0 - f1_v

    # H: Spearman rank correlation, normalized to [0,1]
    rho = _spearman(ent_v, error)
    H = (1 + rho) / 2

    # ST-AUC
    st_auc_raw, st_auc_baseline = _st_auc(f1_v, ent_v)

    # RUI_u = (H + ST-AUC) / 2  (AUROC dropped — r=0.989 with H)
    rui_u = (H + st_auc_raw) / 2

    return {
        'H': float(H),
        'rho': float(rho),
        'st_auc': float(st_auc_raw),
        'st_auc_baseline': float(st_auc_baseline),
        'rui_u': float(rui_u),
        'n_valid': int(len(f1_v)),
    }


def compute_rui(units: List[EvalUnit]) -> Dict[str, Any]:
    """Compute RNA Uncertainty Index from evaluation units."""
    total_n = sum(u.n for u in units)
    if total_n == 0:
        return {'rui': 0.0, 'sigma': 0.0, 'n_units': 0, 'per_unit': {}}

    per_unit = {}
    rui_vals = []
    weights = []

    for u in units:
        key = f"{u.source}/{u.length_bin}"
        unit_rui = compute_rui_unit(u)
        per_unit[key] = unit_rui
        rui_vals.append(unit_rui['rui_u'])
        weights.append(u.n / total_n)

    rui_vals = np.array(rui_vals)
    weights = np.array(weights)

    rui = float(np.sum(weights * rui_vals))
    sigma = float(np.sqrt(np.sum(weights * (rui_vals - rui) ** 2)))

    return {
        'rui': rui,
        'sigma': sigma,
        'n_units': len(units),
        'per_unit': per_unit,
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_shifu_card(
    model_name: str,
    max_seq_len: int,
    beta: float,
    rgi_result: Dict,
    rui_result: Dict,
    sequences: List[SequenceResult],
    out_dir: Path,
):
    """Generate all Shifu evaluation outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(sequences)
    n_penalized = sum(1 for s in sequences if s.penalized)
    n_processable = n_total - n_penalized
    pct_processable = 100 * n_processable / n_total if n_total else 0
    pct_penalized = 100 * n_penalized / n_total if n_total else 0

    # ── JSON card ──
    card = {
        'framework_version': FRAMEWORK_VERSION,
        'shifu_version': SHIFU_VERSION,
        'model_name': model_name,
        'max_seq_len': max_seq_len,
        'capacity_penalty_beta': beta,
        'n_total': n_total,
        'n_processable': n_processable,
        'n_penalized': n_penalized,
        'pct_processable': pct_processable,
        'rgi': rgi_result,
        'rui': rui_result,
    }

    with open(out_dir / f"shifu_card_{model_name}.json", 'w') as f:
        json.dump(card, f, indent=2, default=str)

    # ── Markdown card ──
    lines = []
    lines.append(f"# Shifu Evaluation Card — {model_name}")
    lines.append(f"")
    lines.append(f"- **Dataset:** Shifu {SHIFU_VERSION} ({n_total:,} seqs)")
    lines.append(f"- **Split:** test")
    lines.append(f"- **Framework:** v{FRAMEWORK_VERSION}")
    lines.append(f"")
    lines.append(f"## Scores")
    lines.append(f"")
    lines.append(f"| Index | Score | σ | Details |")
    lines.append(f"|-------|------:|---:|---------|")
    lines.append(f"| **RGI({SHIFU_VERSION})** | **{rgi_result['rgi']:.4f}** | "
                 f"{rgi_result['sigma']:.4f} | tail={rgi_result['rgi_tail']:.4f}, "
                 f"SR={rgi_result['spread_ratio']:.4f}, K={rgi_result['n_units']} |")
    lines.append(f"| **RUI({SHIFU_VERSION})** | **{rui_result['rui']:.4f}** | "
                 f"{rui_result['sigma']:.4f} | K={rui_result['n_units']} |")
    lines.append(f"")
    lines.append(f"## Capacity")
    lines.append(f"")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| max_seq_len | {max_seq_len} |")
    lines.append(f"| Processable | {n_processable:,} ({pct_processable:.1f}%) |")
    lines.append(f"| Penalized | {n_penalized:,} ({pct_penalized:.1f}%) |")
    lines.append(f"| β | {beta} |")
    lines.append(f"")

    # ── Per-unit profile ──
    lines.append(f"## Per-Unit Profile")
    lines.append(f"")
    lines.append(f"| Unit | Source | Length | n | F1 | Δ RGI | Entropy μ | ρ | ST-AUC | RUI_u |")
    lines.append(f"|------|--------|--------|---:|----:|------:|----------:|---:|-------:|------:|")

    rgi_val = rgi_result['rgi']
    for p in sorted(rgi_result['profile'], key=lambda x: x['f1']):
        key = f"{p['source']}/{p['length_bin']}"
        rui_unit = rui_result.get('per_unit', {}).get(key, {})

        # Find mean entropy for this unit
        unit_seqs = [s for s in sequences
                     if s.source_dataset == p['source']
                     and get_length_bin(s.length) == p['length_bin']]
        ent_mu = np.mean([s.entropy for s in unit_seqs if np.isfinite(s.entropy)]) if unit_seqs else float('nan')

        rho = rui_unit.get('rho', float('nan'))
        st_auc = rui_unit.get('st_auc', float('nan'))
        rui_u = rui_unit.get('rui_u', float('nan'))

        delta = p['f1'] - rgi_val
        lines.append(
            f"| {key} | {p['source']} | {p['length_bin']} | {p['n']} | "
            f"{p['f1']:.4f} | {delta:+.4f} | "
            f"{ent_mu:.4f} | {rho:.3f} | {st_auc:.4f} | {rui_u:.4f} |"
        )

    lines.append(f"")

    with open(out_dir / f"shifu_card_{model_name}.md", 'w') as f:
        f.write("\n".join(lines))

    # ── CSV profile ──
    with open(out_dir / f"shifu_profile_{model_name}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["source", "length_bin", "n", "weight", "f1", "delta_rgi",
                     "rho", "H", "st_auc", "rui_u"])
        for p in rgi_result['profile']:
            key = f"{p['source']}/{p['length_bin']}"
            rui_unit = rui_result.get('per_unit', {}).get(key, {})
            w.writerow([
                p['source'], p['length_bin'], p['n'],
                f"{p['weight']:.6f}", f"{p['f1']:.6f}",
                f"{p['f1'] - rgi_val:+.6f}",
                f"{rui_unit.get('rho', 0):.6f}",
                f"{rui_unit.get('H', 0.5):.6f}",
                f"{rui_unit.get('st_auc', 0):.6f}",
                f"{rui_unit.get('rui_u', 0.5):.6f}",
            ])

    logger.info(f"Shifu card written to {out_dir}")

    return card


# =============================================================================
# Plotting
# =============================================================================

def plot_rgi_bar(rgi_result: Dict, model_name: str, out_dir: Path):
    """Bar chart of per-unit F1 with RGI line."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    profile = sorted(rgi_result['profile'], key=lambda p: p['f1'])
    labels = [f"{p['source'][:8]}\n{p['length_bin']}" for p in profile]
    f1_vals = [p['f1'] for p in profile]
    rgi_val = rgi_result['rgi']

    fig, ax = plt.subplots(figsize=(max(10, len(profile) * 0.8), 5))
    colors = ['#e74c3c' if f < rgi_val - 0.1 else '#2ecc71' if f > rgi_val + 0.05 else '#3498db'
              for f in f1_vals]
    bars = ax.bar(range(len(f1_vals)), f1_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(rgi_val, color='black', linestyle='--', linewidth=1.5,
               label=f'RGI = {rgi_val:.4f}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Micro-F1')
    ax.set_title(f'RGI Profile — {model_name}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / f"rgi_bar_{model_name}.png", dpi=150)
    plt.close()


def plot_rui_bar(rui_result: Dict, model_name: str, out_dir: Path):
    """Bar chart of per-unit RUI."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    per_unit = rui_result.get('per_unit', {})
    if not per_unit:
        return

    keys = sorted(per_unit.keys())
    rui_vals = [per_unit[k]['rui_u'] for k in keys]
    rui_global = rui_result['rui']

    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.8), 5))
    labels = [k.replace('/', '\n') for k in keys]
    ax.bar(range(len(rui_vals)), rui_vals, color='#9b59b6', edgecolor='white', linewidth=0.5)
    ax.axhline(rui_global, color='black', linestyle='--', linewidth=1.5,
               label=f'RUI = {rui_global:.4f}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('RUI')
    ax.set_title(f'RUI Profile — {model_name}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / f"rui_bar_{model_name}.png", dpi=150)
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def evaluate_single_model(
    model_name: str,
    eval_dir: str,
    shifu_records: List[Dict],
    max_seq_len: int,
    beta: float,
    out_dir: Path,
    split: str = "test",
) -> Dict[str, Any]:
    """Full evaluation pipeline for one model."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")

    # Load eval results
    dataset_csvs = load_eval_index(eval_dir)
    if not dataset_csvs:
        # Try single per_sequence.csv in eval_dir
        single = Path(eval_dir) / "per_sequence.csv"
        if single.exists():
            dataset_csvs = {"default": str(single)}
        else:
            logger.error(f"No per_sequence.csv found in {eval_dir}")
            return {}

    # Merge all per_sequence.csv results
    all_eval_results = {}
    for ds_name, csv_path in dataset_csvs.items():
        logger.info(f"  Loading {ds_name}: {csv_path}")
        results = load_per_sequence_csv(csv_path)
        all_eval_results.update(results)
        logger.info(f"    {len(results)} sequences loaded")

    logger.info(f"  Total eval results: {len(all_eval_results)}")

    # Apply capacity penalty and merge with Shifu metadata
    sequences = compute_capacity_penalty(
        shifu_records, all_eval_results, max_seq_len, beta
    )

    # Build evaluation units
    units = build_evaluation_units(sequences)

    # Compute RGI
    rgi_result = compute_rgi(units)
    logger.info(f"  RGI({SHIFU_VERSION}) = {rgi_result['rgi']:.4f} ± {rgi_result['sigma']:.4f} "
                f"[tail={rgi_result['rgi_tail']:.4f}, SR={rgi_result['spread_ratio']:.4f}]")

    # Compute RUI
    rui_result = compute_rui(units)
    logger.info(f"  RUI({SHIFU_VERSION}) = {rui_result['rui']:.4f} ± {rui_result['sigma']:.4f}")

    # Generate reports
    card = generate_shifu_card(
        model_name, max_seq_len, beta,
        rgi_result, rui_result, sequences, out_dir,
    )

    # Plots
    plot_rgi_bar(rgi_result, model_name, out_dir)
    plot_rui_bar(rui_result, model_name, out_dir)

    # Beta sensitivity
    if abs(beta - BETA_DEFAULT) < 0.01:
        logger.info(f"  Running β sensitivity (β={BETA_SENSITIVITY})...")
        seq_alt = compute_capacity_penalty(shifu_records, all_eval_results, max_seq_len, BETA_SENSITIVITY)
        units_alt = build_evaluation_units(seq_alt)
        rgi_alt = compute_rgi(units_alt)
        logger.info(f"  RGI(β={BETA_SENSITIVITY}) = {rgi_alt['rgi']:.4f} "
                    f"(Δ = {rgi_alt['rgi'] - rgi_result['rgi']:+.4f})")
        card['beta_sensitivity'] = {
            'beta_alt': BETA_SENSITIVITY,
            'rgi_alt': rgi_alt['rgi'],
            'rgi_delta': rgi_alt['rgi'] - rgi_result['rgi'],
        }

    return card


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Shifu Evaluation Framework — RGI + RUI computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--shifu_dir", type=str, required=True,
                    help="Path to Shifu dataset directory (contains splits/ or master.jsonl)")
    ap.add_argument("--eval_dir", nargs="+", required=True,
                    help="Eval run directories. Format: NAME:path or just path (for single model)")
    ap.add_argument("--model_name", nargs="+", default=None,
                    help="Model names (if --eval_dir doesn't use NAME: format)")
    ap.add_argument("--max_seq_len", nargs="+", type=int, required=True,
                    help="Max sequence length per model (for capacity penalty)")
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT,
                    help="Capacity penalty exponent")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"],
                    help="Which Shifu split to evaluate on")
    ap.add_argument("--out_dir", type=str, default="./shifu_report",
                    help="Output directory for reports")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse model specs
    models = []
    for i, spec in enumerate(args.eval_dir):
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name = (args.model_name[i] if args.model_name and i < len(args.model_name)
                    else f"model_{i}")
            path = spec

        msl = args.max_seq_len[i] if i < len(args.max_seq_len) else args.max_seq_len[-1]
        models.append((name.strip(), path.strip(), msl))

    # Load Shifu metadata once
    shifu_records = load_shifu_metadata(args.shifu_dir, split=args.split)

    # Evaluate each model
    all_cards = []
    for name, eval_path, msl in models:
        card = evaluate_single_model(
            model_name=name,
            eval_dir=eval_path,
            shifu_records=shifu_records,
            max_seq_len=msl,
            beta=args.beta,
            out_dir=out_dir,
            split=args.split,
        )
        if card:
            all_cards.append(card)

    # Multi-model comparison table
    if len(all_cards) > 1:
        with open(out_dir / "shifu_comparison.csv", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["model", "max_seq_len", "RGI", "σ_RGI", "RGI_tail", "SR",
                         "RUI", "σ_RUI", "n_units", "processable_%", "penalized_%"])
            for card in all_cards:
                w.writerow([
                    card['model_name'], card['max_seq_len'],
                    f"{card['rgi']['rgi']:.4f}", f"{card['rgi']['sigma']:.4f}",
                    f"{card['rgi']['rgi_tail']:.4f}", f"{card['rgi']['spread_ratio']:.4f}",
                    f"{card['rui']['rui']:.4f}", f"{card['rui']['sigma']:.4f}",
                    card['rgi']['n_units'], f"{card['pct_processable']:.1f}",
                    f"{100-card['pct_processable']:.1f}",
                ])

        # RGI vs RUI scatter
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plots_dir = out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            for card in all_cards:
                ax.scatter(card['rgi']['rgi'], card['rui']['rui'], s=100, zorder=5)
                ax.annotate(card['model_name'],
                           (card['rgi']['rgi'], card['rui']['rui']),
                           textcoords="offset points", xytext=(8, 8), fontsize=9)
            ax.set_xlabel('RGI (Generalization)')
            ax.set_ylabel('RUI (Uncertainty)')
            ax.set_title(f'Shifu Evaluation — RGI vs RUI ({SHIFU_VERSION})')
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(plots_dir / "rgi_vs_rui_scatter.png", dpi=150)
            plt.close()
        except ImportError:
            pass

    # Write summary
    summary = {
        'framework_version': FRAMEWORK_VERSION,
        'shifu_version': SHIFU_VERSION,
        'split': args.split,
        'beta': args.beta,
        'n_models': len(all_cards),
        'cards': all_cards,
    }
    with open(out_dir / "shifu_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nDone. Reports in {out_dir}")
    for card in all_cards:
        logger.info(f"  {card['model_name']}: RGI={card['rgi']['rgi']:.4f}, RUI={card['rui']['rui']:.4f}")


if __name__ == "__main__":
    main()