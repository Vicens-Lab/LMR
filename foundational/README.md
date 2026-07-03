# Foundational backbones

Masked-language-model pretraining of the RNA backbones. Weights are on HuggingFace;
the configs and trainers here reproduce them.

| Model | config | params | context | trainer | HF weights |
|---|---|---|---|---|---|
| LMR-v0 | `configs/lmr_v0.yml` | 289M | 512 | `train.py` | GaboG7/LMR-v0 |
| LMR-nano | `configs/lmr_nano.yml` | 65M | 512 | `train.py` | GaboG7/LMR-mini |
| LMR-G | `configs/lmr_g.yml` | 86M | 512 | `train.py` | GaboG7/LMR-G |
| LMR-G-160 | `configs/lmr_g_160.yml` | 228M | 512 | `train.py` | (optional) |
| LMR-Long | `configs/lmr_long.yml` | 290M | 4,096 | `train_long.py` | GaboG7/LMR-Long |

`train.py` (v0 / nano / G) and `train_long.py` (long-context, hybrid
sliding-window/full attention + NTK-aware RoPE) are the config-driven, DDP
masked-language-model trainers. Comments and docstrings are removed for release;
both are verified to load and expose their CLI.

## Run
```bash
export DATA_ROOT=/path/to/your/data       # replaces <DATA_ROOT> in the configs
PYTHONPATH=. python train.py      --config configs/lmr_v0.yml
PYTHONPATH=. python train_long.py --config configs/lmr_long.yml
```
Each config's `<DATA_ROOT>` paths point at the pretraining corpus and the output
checkpoint directory; edit them for your environment.
