<p align="center">
  <img src="assets/LMRs.png" width="440" alt="LMRs">
</p>

<h1 align="center">LMR: Language Models for RNA</h1>

<p align="center">
  Official code, training configs, and foundational model weights for<br>
  <b>SHIFU: an integrated framework for deep learning of RNA secondary structure</b><br>
  <i>Gabriel Galvez and Quentin Vicens, Vicens Lab, University of Houston</i>
</p>

---

## Overview

RNA secondary-structure prediction is limited less by model scale than by how data
are curated, represented, and measured. **SHIFU** is a three-part framework that
makes that measurable:

1. **SHIFU-Corpus** — a leakage-audited dataset of 254,123 RNA sequences from six
   public databases, with family-aware train/validation/test splits certified free
   of exact and near-duplicate leaks.
2. **The SHIFU Trifecta** — a three-axis evaluation (correctness, breadth across
   out-of-distribution data, and whether a model's confidence is usable) that
   replaces a single leaderboard number.
3. **The LMR family** — compact RNA language-model backbones (LMR-v0, LMR-nano,
   LMR-G) used as controlled experiments: changing only the training corpus shifts
   accuracy by 0.13, and a 65M-parameter model leads on correctness while running
   on a laptop.

<p align="center"><img src="assets/trifecta.png" width="620" alt="SHIFU Trifecta"></p>

This repository releases the **foundational backbones** (weights on HuggingFace;
configs and the training script here) and, for the **2D structure models**, the
`config.yml` files for replication.

## Repository layout

```
lmr_g/               foundational backbone architecture (LMR-Foundation)
tokenizer.py         minimal RNA tokenizer
data/  training/     the masked-language-model data + training utilities
foundational/
  configs/           lmr_v0.yml, lmr_nano.yml, lmr_g.yml, lmr_g_160.yml
  train.py           masked-language-model trainer (config-driven, DDP)
models_2d/
  configs/           lmr_shifu.yml, lmr_bprna.yml  (config only; no weights/scripts)
data/benchmark_certificate/   split leakage / dedup / label audit
assets/              figures
```

## Models (HuggingFace)

| Model | Params | Weights |
|---|---|---|
| LMR-v0 | 289M | `git clone https://huggingface.co/GaboG7/LMR-v0` |
| LMR-G | 86M | `git clone https://huggingface.co/GaboG7/LMR-G` |
| LMR-nano | 65M | `git clone https://huggingface.co/GaboG7/LMR-mini` |

## Installation

```bash
git clone https://github.com/Vicens-Lab/LMR
cd LMR
python -m venv .venv && source .venv/bin/activate     # or conda create -n lmr python=3.10
pip install -r requirements.txt
```
Requires Python >= 3.9 and a CUDA GPU for training.

## Getting the dataset

The full **SHIFU-Corpus** (254,123 sequences with the family-aware splits) is a
HuggingFace dataset:

```bash
huggingface-cli download GaboG7/SHIFU-Corpus --repo-type dataset --local-dir ./shifu_corpus
```

The split integrity is provable from `data/benchmark_certificate/` (0 exact and
0 cluster train/test leaks; per-split composition; byte-level hashes).

## Set up a finetuning environment

```bash
# 1. install (above) and pick your data/checkpoint root
export DATA_ROOT=/path/to/your/data      # replaces <DATA_ROOT> in the configs

# 2. get a foundational backbone
git clone https://huggingface.co/GaboG7/LMR-v0 $DATA_ROOT/checkpoints/lmr_v0

# 3. edit the paths in the chosen config (data + checkpoint dirs use <DATA_ROOT>)

# 4. reproduce a foundational backbone (masked-language-model pretraining)
PYTHONPATH=. python foundational/train.py --config foundational/configs/lmr_v0.yml
```

To fine-tune a **2D structure model**, start from a released backbone and use a
config in `models_2d/configs/` (see `models_2d/README.md`); the 2D training script
is not part of this release, but the setup is fully specified by the config.

## Citation

```bibtex
@article{galvez2026shifu,
  title   = {SHIFU: an integrated framework for deep learning of RNA secondary structure},
  author  = {Galvez, Gabriel and Vicens, Quentin},
  journal = {Nature Methods (under review)},
  year    = {2026}
}
```

## License

Code in this repository is released under the **MIT License** (see `LICENSE`).
**SHIFU-Corpus** is derived from six public databases (RNASSTR, bpRNA, RNAStrAlign,
ArchiveII, DSSR/RNAsolo) that retain their own licenses; cite and honor the
original sources when using the data.
