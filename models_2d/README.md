# 2D structure-prediction models (SHIFU-LMR)

**Release policy: config only.** For the 2D structure models we release the
`config.yml` files for replication. We do **not** release the 2D training scripts
or the 2D model weights. The 2D models are reproduced from a released foundational
backbone (see `../foundational/`) plus these configs and the corpus.

## Configs
Both released 2D models share the same pairing architecture and differ only in
the training corpus (the paper's controlled corpus swap):

| Config | Model | Trained on |
|---|---|---|
| `configs/lmr_shifu.yml` | SHIFU-LMR-Shifu | SHIFU-Corpus (family-aware split) |
| `configs/lmr_bprna.yml` | SHIFU-LMR-bpRNA | **generic bpRNA-1m** (bpRNA_1m_90) |

`lmr_bprna.yml` uses generic bpRNA-1m (`bpRNA_1m_90_structures`), exactly as in the
paper. `lmr_shifu.yml` is identical except its `data_path` points at SHIFU-Corpus;
set that path (`<DATA_ROOT>/shifu_corpus/...`) to your local corpus. Each config
selects a foundational backbone through `backbone_config_path` /
`backbone_checkpoint_path` (genericized to `<DATA_ROOT>`).
