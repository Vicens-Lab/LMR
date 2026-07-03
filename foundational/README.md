# Foundational backbones

Masked-language-model pretraining of the RNA backbones. Weights are on HuggingFace;
the configs and trainer here reproduce them.

| Model | config | params | HF weights |
|---|---|---|---|
| LMR-v0 | `configs/lmr_v0.yml` | 289M | GaboG7/LMR-v0 |
| LMR-nano | `configs/lmr_nano.yml` | 65M | GaboG7/LMR-mini |
| LMR-G | `configs/lmr_g.yml` | 86M | GaboG7/LMR-G |
| LMR-G-160 | `configs/lmr_g_160.yml` | 228M | (optional) |

`train.py` is the config-driven, DDP masked-language-model trainer (comments and
docstrings removed for release; verified to compile).

## Run
```
export DATA_ROOT=/path/to/your/data      # replaces <DATA_ROOT> in the configs
python train.py --config configs/lmr_v0.yml
```
Each config's `<DATA_ROOT>` paths point to the pretraining corpus and the output
checkpoint directory; edit them for your environment.
