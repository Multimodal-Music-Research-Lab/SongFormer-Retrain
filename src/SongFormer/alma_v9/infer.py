from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import AlmaV9Dataset, alma_v9_collate, build_hx_bench_items
from .labels import id_to_label
from .model import AlmaV9Model, AlmaV9ModelConfig


def build_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.lyrics_tokenizer_dir, use_fast=True)
    if cfg.model.line_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([cfg.model.line_token])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def move_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def local_maxima(prob: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    candidates = []
    for idx in range(1, len(prob) - 1):
        if prob[idx] >= threshold and prob[idx] >= prob[idx - 1] and prob[idx] >= prob[idx + 1]:
            candidates.append(idx)
    candidates.sort(key=lambda i: prob[i], reverse=True)
    selected = []
    for idx in candidates:
        if all(abs(idx - j) >= min_distance for j in selected):
            selected.append(idx)
    return sorted(selected)


def logits_to_msa(outputs, batch, cfg):
    frame_hz = float(cfg.mert.frame_hz)
    boundary_prob = torch.sigmoid(outputs["boundary_logits"])[0].detach().cpu().numpy()
    function_prob = torch.softmax(outputs["function_logits"], dim=-1)[0].detach().cpu().numpy()
    seq_valid = batch["seq_valid"][0].detach().cpu().numpy().astype(bool)
    length = int(seq_valid.sum())
    boundary_prob = boundary_prob[:length]
    function_prob = function_prob[:length]
    duration = float(batch["duration"][0].detach().cpu())
    if duration <= 0:
        duration = length / frame_hz

    min_distance = int(round(float(cfg.postprocess.min_boundary_distance_sec) * frame_hz))
    boundary_idxs = local_maxima(
        boundary_prob,
        threshold=float(cfg.postprocess.boundary_threshold),
        min_distance=max(1, min_distance),
    )
    times = [0.0]
    for idx in boundary_idxs:
        t = idx / frame_hz
        if 0.5 <= t <= duration - 0.5:
            times.append(float(t))
    times.append(duration)
    times = sorted(set(round(t, 3) for t in times))

    msa = []
    for left_t, right_t in zip(times[:-1], times[1:]):
        left = max(0, int(round(left_t * frame_hz)))
        right = min(length, int(round(right_t * frame_hz)))
        if right <= left:
            continue
        label_id = int(function_prob[left:right].mean(axis=0).argmax())
        label = id_to_label(label_id)
        if msa and msa[-1][1] == label:
            continue
        msa.append((left_t, label))
    if not msa:
        label_id = int(function_prob.mean(axis=0).argmax())
        msa.append((0.0, id_to_label(label_id)))
    msa.append((round(duration, 3), "end"))
    return msa


def write_msa(path: str | Path, msa):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t, label in msa:
            f.write(f"{float(t):.3f} {label}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scp", required=True)
    parser.add_argument("--lyrics_dir", required=True)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    feature_dir = args.feature_dir or cfg.data.mert_feature_dir
    tokenizer = build_tokenizer(cfg)
    items = build_hx_bench_items(args.scp, args.lyrics_dir)
    dataset = AlmaV9Dataset(
        items=items,
        tokenizer=tokenizer,
        feature_dir=feature_dir,
        frame_hz=float(cfg.mert.frame_hz),
        slice_dur=float(cfg.train.slice_dur),
        train=False,
        max_text_len=int(cfg.model.lyrics_max_length),
        line_max_len=int(cfg.model.lyrics_line_max_length),
        boundary_widen_sec=float(cfg.loss.boundary_widen_sec),
        line_token=cfg.model.line_token,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=alma_v9_collate)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = AlmaV9ModelConfig(
        audio_dim=dataset.audio_dim,
        vocab_size=len(tokenizer),
        num_classes=int(cfg.model.num_classes),
        d_model=int(cfg.model.d_model),
        lyrics_layers=int(cfg.model.lyrics_layers),
        lyrics_heads=int(cfg.model.lyrics_heads),
        lyrics_ffn_dim=int(cfg.model.lyrics_ffn_dim),
        fusion_layers=int(cfg.model.fusion_layers),
        fusion_block_type=str(cfg.model.fusion_block_type),
        dropout=float(cfg.model.dropout),
        frame_hz=float(cfg.mert.frame_hz),
        loss_weight_boundary=float(cfg.loss.weight_boundary),
        loss_weight_function=float(cfg.loss.weight_function),
        focal_weight=float(cfg.loss.focal_weight),
        focal_alpha=float(cfg.loss.focal_alpha),
        focal_gamma=float(cfg.loss.focal_gamma),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlmaV9Model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            song_id = batch["song_ids"][0]
            batch = move_to_device(batch, device)
            outputs = model(batch)
            msa = logits_to_msa(outputs, batch, cfg)
            write_msa(Path(args.output_dir) / f"{song_id}.txt", msa)


if __name__ == "__main__":
    main()

