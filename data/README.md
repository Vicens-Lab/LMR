# Data

The full **SHIFU-Corpus** (254,123 sequences from six databases, with the
family-aware train / val / test splits: 203,298 / 25,412 / 25,413) is released as
a HuggingFace dataset (URL to add).

`benchmark_certificate/` contains the audit that proves the splits are
leakage-controlled and the corpus is what the paper reports:

| file | what it certifies |
|---|---|
| `leakage_report.json` | 0 exact and 0 cluster train/test leaks (MinHash k=6, 128 perms, J >= 0.90) |
| `split_summary.json` | per-split sizes, source composition, family counts |
| `dedup_report.json` | exact SHA-256 dedup accounting (310,034 -> 254,393) |
| `label_coverage.json` | tiered Rfam/clan label coverage |
| `manifest_hashes.json` | byte-level file hashes for the released artifacts |
| `global_summary.json` | corpus-level summary statistics |
