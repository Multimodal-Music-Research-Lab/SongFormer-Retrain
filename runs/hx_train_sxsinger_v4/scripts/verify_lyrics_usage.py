#!/usr/bin/env python3

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from ema_pytorch import EMA
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

scipy.inf = np.inf

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_SONGFORMER = REPO_ROOT / "src" / "SongFormer"
sys.path.insert(0, str(SRC_SONGFORMER))

from models.SongFormer import Model


def load_model(config, checkpoint_path, device):
    model = Model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_ema") is not None:
        model_ema = EMA(model, include_online_model=False, **config.ema_kwargs)
        model_ema.load_state_dict(checkpoint["model_ema"])
        model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def lyrics_off_batch(batch):
    result = copy.copy(batch)
    result["has_lyrics"] = torch.zeros_like(batch["has_lyrics"])
    return result


def shuffled_lyrics_batch(batch):
    result = copy.copy(batch)
    for key in [
        "lyrics_line_embeddings",
        "lyrics_line_repeat_embeddings",
        "lyrics_line_repeat_features",
    ]:
        result[key] = torch.flip(batch[key], dims=[1])
    return result


def evaluate_variant(model, loader, device, variant):
    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=variant):
            if batch is None:
                continue
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            if variant == "lyrics_off":
                batch = lyrics_off_batch(batch)
            elif variant == "lyrics_shuffled":
                batch = shuffled_lyrics_batch(batch)
            records.append(model.infer_with_metrics(batch))
    return pd.DataFrame(records).mean(numeric_only=True).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device)
    dataset = instantiate(config.eval_dataset)
    loader = DataLoader(dataset, **config.eval_dataloader, collate_fn=dataset.collate_fn)

    records = []
    for variant in ["normal", "lyrics_off", "lyrics_shuffled"]:
        metrics = evaluate_variant(model, loader, device, variant)
        records.append({"variant": variant, **metrics})

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
