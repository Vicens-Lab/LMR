#!/usr/bin/env python3
"""Reference SHIFU data pipeline (evaluation + training). This is a UTILITY, not a
turnkey script: you adapt the model-call parts to your own 2D structure model.

- iter_shifu_records(path): read SHIFU-Corpus records (sequence + reference
  structure) from the eval CSV or a splits JSONL, for evaluation.
- For TRAINING streaming, the foundational trainers use
  data/pytorch_wrapper.get_datasets (masked-language-model batches); reuse it or
  point it at your corpus.
"""
import csv
import json
import os


def iter_shifu_records(path):
    """Yield dicts: {uid, sequence, structure, source, length}.

    Accepts the SHIFU eval CSV (columns: filename, sequence, X2D_DSSR, family,
    source_dataset, length) or a splits JSONL (uid, sequence, dotbracket_nopk,
    source_dataset, length). Sequences are upper-cased and DNA->RNA normalized.
    """
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                seq = (r.get("sequence") or "").upper().replace("T", "U")
                yield {
                    "uid": r.get("uid") or r.get("filename"),
                    "sequence": seq,
                    "structure": r.get("dotbracket_nopk") or r.get("X2D_DSSR") or r.get("dotbracket_raw", ""),
                    "source": r.get("source_dataset", "UNK"),
                    "length": int(r.get("length", len(seq))),
                }
    else:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                seq = (r.get("sequence") or "").upper().replace("T", "U")
                yield {
                    "uid": r.get("filename") or r.get("uid"),
                    "sequence": seq,
                    "structure": r.get("X2D_DSSR") or r.get("structure", ""),
                    "source": r.get("source_dataset", "UNK"),
                    "length": int(r.get("length", len(seq))),
                }


# ---------------------------------------------------------------------------
# Evaluation recipe (wire in YOUR model where marked):
#
#   from utils.scoring import parse_dotbracket_multilevel, pk_strip_pairs_greedy, basepair_set_prf1
#   rows = []
#   for rec in iter_shifu_records("shifu_test.csv"):
#       true_pairs = pk_strip_pairs_greedy(parse_dotbracket_multilevel(rec["structure"]))
#       pred_pairs = YOUR_2D_MODEL.predict(rec["sequence"])          # <-- your model here
#       m = basepair_set_prf1(pred_pairs, true_pairs)                # tp/fp/fn/precision/recall/f1
#       rows.append({"filename": rec["uid"], "source_dataset": rec["source"],
#                    "length": rec["length"], "f1_decoded_nopk": m["f1"],
#                    "tp_decoded": m["tp"], "fp_decoded": m["fp"], "fn_decoded": m["fn"],
#                    "mean_entropy_normalized": your_per_position_entropy})   # optional, for RUI
#   # write rows -> per_sequence.csv, then aggregate with utils/trifecta.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    n = 0
    for rec in iter_shifu_records(sys.argv[1]):
        n += 1
        if n <= 3:
            print(rec["uid"], rec["length"], rec["sequence"][:32], "|", rec["structure"][:32])
    print(f"{n} records")
