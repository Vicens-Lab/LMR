#!/usr/bin/env python3
"""Hardware-agnostic quickstart. Loads an LMR backbone on whatever device is
available (CUDA -> Apple MPS -> CPU) and runs a forward pass. Weights are
optional: pass --checkpoint to load pretrained weights, otherwise a fresh model
is built so you can confirm the code runs on your hardware.

    python example.py                                  # LMR-nano, auto device
    python example.py --long --config foundational/configs/lmr_long.yml
    python example.py --checkpoint path/to/model.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tokenizer import MinimalRNATokenizer


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config_path, long=False):
    if long:
        import yaml
        from model.config import LMRConfig
        from model.architecture_long import LMRLong
        mc = LMRConfig.from_yaml(config_path)
        rope = yaml.safe_load(open(config_path)).get("rope_scaling", {})
        return LMRLong(mc, rope_scaling=rope)
    from lmr_g.config_foundation import LMRFoundationConfig
    from lmr_g.architecture_foundation import create_lmr_foundation
    return create_lmr_foundation(LMRFoundationConfig.from_yaml(config_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="foundational/configs/lmr_nano.yml")
    ap.add_argument("--long", action="store_true", help="use the long-context (LMR-Long) architecture")
    ap.add_argument("--checkpoint", default=None, help="optional .pt weights to load")
    ap.add_argument("--seq", default="GGGAAACUCCUUGGGAGAGUCC")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    model = load_model(args.config, long=args.long).to(device).eval()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        if isinstance(sd, dict):
            sd = sd.get("model", sd.get("model_state_dict", sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded weights from {args.checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    tok = MinimalRNATokenizer()
    ids = torch.tensor([tok.encode(args.seq)], device=device)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out[0] if isinstance(out, (tuple, list)) else (out.get("logits") if isinstance(out, dict) else out)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M | input length: {ids.shape[1]} | logits: {tuple(logits.shape)}")
    print("OK: the model runs on this hardware.")


if __name__ == "__main__":
    main()
