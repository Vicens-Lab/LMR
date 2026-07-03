# utils: evaluation & data pipeline

Drop-in utilities to **integrate as you need** for scoring **your own** RNA
structure model against SHIFU-Corpus with the SHIFU Trifecta. These are building
blocks, not a turnkey pipeline: take only the pieces you want and wire in your
model's predictions. The scoring and metric code is the paper's, so your numbers
stay comparable.

| file | what it does |
|---|---|
| `scoring.py` | score one prediction: exact base-pair precision/recall/F1 (pseudoknot-stripped) and categorical partner entropy. Verbatim scoring functions from the paper. |
| `trifecta.py` | compute the **SHIFU Trifecta** (micro-F1, RBI = sqrt(Cov x Tail20), RUI) from per-sequence results + split metadata. The paper's evaluator. |
| `data_pipeline.py` | load SHIFU-Corpus records (sequence + reference structure) for evaluation. For training streaming, the foundational trainers use `data/pytorch_wrapper.py`. |

## Evaluate your 2D model in three steps

1. **Load the data.** `iter_shifu_records("shifu_test.csv")` (from `data_pipeline.py`)
   yields `{uid, sequence, structure, source, length}`.
2. **Score each prediction.** For every record, get your model's predicted base
   pairs and score against the reference:
   ```python
   from utils.scoring import parse_dotbracket_multilevel, pk_strip_pairs_greedy, basepair_set_prf1
   true = pk_strip_pairs_greedy(parse_dotbracket_multilevel(rec["structure"]))
   pred = your_model.predict(rec["sequence"])          # <- your 2D model
   m = basepair_set_prf1(pred, true)                   # tp/fp/fn, precision, recall, f1
   ```
   Optionally compute a per-position entropy for RUI (see `scoring.py`).
   Write these per-sequence rows to a `per_sequence.csv`.
3. **Aggregate the Trifecta.** Run `trifecta.py` on that `per_sequence.csv` plus the
   split metadata to get micro-F1, RBI, and RUI over the evaluation units:
   ```bash
   python utils/trifecta.py --shifu_dir <corpus_dir> --eval_dir <your_eval_dir> \
       --model_name my2dmodel --max_seq_len <your_cap> --out_dir out/
   ```
   `python utils/trifecta.py --help` lists every input.

The compute is deterministic and matches the paper; wiring your model's
predictions in (step 2) is the only piece you write.
