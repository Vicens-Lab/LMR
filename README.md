# LMR: Language Models for RNA

> A leakage-audited RNA benchmark, an honest three-axis evaluation, and compact RNA
> language models that match or beat models 10x their size. The code and models
> behind the **SHIFU** framework.

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="Models: HuggingFace" src="https://img.shields.io/badge/models-HuggingFace-yellow.svg">
</p>

---

## Why LMR?

Most RNA secondary-structure models look excellent on their home benchmark and
collapse on the next one, because most benchmarks measure **memorization, not
generalization**. Run one model across several public benchmarks and its accuracy
swings from near-perfect to near-chance:

<p align="center"><img src="assets/cross_corpus.png" width="760" alt="cross-corpus instability"></p>
<p align="center"><i>Same models, seven public splits. The instability is the data and the
measurement, not the model. This is the problem SHIFU is built to fix.</i></p>

LMR rebuilds the foundation so a number means something:

- **An honest benchmark.** SHIFU-Corpus is 254,123 sequences from six databases,
  deduplicated and **certified with zero exact and zero near-duplicate train/test
  leaks**, with family-disjoint splits. A high score here reflects generalization,
  not a leaked test set.
- **Evaluation that shows where models fail.** The **SHIFU Trifecta** scores
  correctness, breadth across out-of-distribution data, and whether a model's
  confidence is actually usable, instead of one number that hides everything.
- **Small models that punch far above their weight.** A 65M-parameter backbone
  reaches **micro-F1 0.789 vs RiNALMo's 0.627 (650M)**, and swapping only the
  training corpus moves accuracy by **0.13**, proof that the bottleneck is data and
  representation, not scale. Everything runs on a laptop GPU.

<p align="center"><img src="assets/frontier.png" width="760" alt="LMR frontier"></p>

**Use LMR if you want** a trustworthy RNA-structure benchmark, a compact RNA
language-model backbone to fine-tune, or a leakage-free baseline that cannot be
gamed by test-set overlap.

## The SHIFU Trifecta

<p align="center"><img src="assets/trifecta.png" width="600" alt="SHIFU Trifecta"></p>

Three orthogonal axes, so no single number can hide a weakness: **micro-F1**
(correctness), **RBI** (breadth = how well a model holds up on its hardest,
out-of-distribution units), and **RUI** (whether its confidence ranks hard cases
correctly). A model can be accurate but narrow, broad but blind, or genuinely
strong on all three, and the Trifecta tells them apart.

## Foundational models

Four pretrained RNA backbones on HuggingFace (weights); configs and training code
are in this repo.

| Model | Params | Context | Weights |
|---|---|---|---|
| LMR-v0 | 289M | 512 | `git clone https://huggingface.co/GaboG7/LMR-v0` |
| LMR-G | 86M (or 228M) | 512 | `git clone https://huggingface.co/GaboG7/LMR-G` |
| LMR-nano | 65M | 512 | `git clone https://huggingface.co/GaboG7/LMR-mini` |
| LMR-Long | 290M | 4,096 | `git clone https://huggingface.co/GaboG7/LMR-Long` |

## Installation

```bash
git clone https://github.com/Vicens-Lab/LMR
cd LMR
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Python >= 3.9. Dependencies are just `torch`, `numpy`, `pyyaml` (plus
`huggingface_hub`/`safetensors` to pull weights).

## Runs on any hardware

The code **auto-detects your device** (CUDA -> Apple Silicon MPS -> CPU); nothing is
hardcoded to a GPU, mixed precision is enabled only on CUDA, and multi-GPU is optional.
Confirm it runs on your machine:

```bash
python example.py                                              # LMR-nano, auto device
python example.py --long --config foundational/configs/lmr_long.yml
python example.py --checkpoint path/to/weights.pt             # load a pretrained backbone
```

Single device by default; scale out with `torchrun --nproc_per_node=N
foundational/train.py ...`. Full pretraining of the larger backbones still wants a CUDA
GPU, but everything loads and runs on CPU or Apple Silicon for development and inference.

## Get the dataset

```bash
huggingface-cli download GaboG7/SHIFU-Corpus --repo-type dataset --local-dir ./shifu_corpus
```
The split integrity is provable offline from `data/benchmark_certificate/`
(0 exact + 0 cluster leaks, per-split composition, byte-level hashes).

## Fine-tune

The backbones are pretrained by masked language modeling, then adapted with
parameter-efficient heads and a staged curriculum, no extra data required:

<p align="center"><img src="assets/finetune_workflow.png" width="820" alt="finetune workflow"></p>

```bash
export DATA_ROOT=/path/to/your/data        # replaces <DATA_ROOT> in the configs

# 1. pull a backbone
git clone https://huggingface.co/GaboG7/LMR-v0 $DATA_ROOT/checkpoints/lmr_v0

# 2. reproduce / continue foundational pretraining
PYTHONPATH=. python foundational/train.py      --config foundational/configs/lmr_v0.yml
PYTHONPATH=. python foundational/train_long.py --config foundational/configs/lmr_long.yml

# 3. for a 2D structure model, start from a backbone + a config in models_2d/configs/
#    (config provided for replication; see models_2d/README.md)
```
Edit the `<DATA_ROOT>` paths in the chosen config to point at your corpus and
checkpoint directory.

## Repository layout

```
lmr_g/  model/        foundational backbone architectures (standard + long-context)
tokenizer.py          minimal RNA tokenizer
data/  training/      masked-language-model data + training utilities
foundational/
  configs/            lmr_v0 / lmr_nano / lmr_g / lmr_g_160 / lmr_long
  train.py            MLM trainer (v0 / nano / G)
  train_long.py       long-context MLM trainer (LMR-Long)
models_2d/
  configs/            lmr_shifu.yml, lmr_bprna.yml   (config only; no weights/scripts)
data/benchmark_certificate/   split leakage / dedup / label audit
assets/               figures
```

## Citation

This work is **not yet published**. A citation will be added when it is available.
For now, please reference the repository. Placeholder:

```bibtex
@misc{lmr_shifu,
  title  = {SHIFU: an integrated framework for deep learning of RNA secondary structure},
  author = {Galvez, Gabriel and Vicens, Quentin},
  note   = {Manuscript in preparation. Citation to be updated.},
  year   = {TODO}
}
```

## License

Code is released under the **MIT License** (see `LICENSE`). SHIFU-Corpus is derived
from six public databases (RNASSTR, bpRNA, RNAStrAlign, ArchiveII, DSSR, RNAsolo)
that retain their own licenses; cite and honor the original sources when using the data.

*Vicens Lab, Center for Nuclear Receptors and Cell Signaling, University of Houston.*
