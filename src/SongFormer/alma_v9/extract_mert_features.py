from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from .data import build_hook_items, build_hx_bench_items, build_hx_items


def load_audio(path: str | Path, target_sr: int):
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).contiguous(), target_sr


def aggregate_hidden(
    output_sum: np.ndarray,
    output_count: np.ndarray,
    hidden: torch.Tensor,
    chunk_start: float,
    frame_hz: float,
):
    arr = hidden.detach().cpu().float().numpy()
    for local_idx in range(arr.shape[0]):
        t = chunk_start + local_idx / frame_hz
        global_idx = int(round(t * frame_hz))
        if 0 <= global_idx < output_sum.shape[0]:
            output_sum[global_idx] += arr[local_idx]
            output_count[global_idx] += 1.0


def extract_one(
    audio_path: str | Path,
    out_path: str | Path,
    processor,
    model,
    device,
    sample_rate: int,
    frame_hz: float,
    chunk_sec: float,
    hop_sec: float,
    layer: int,
):
    audio, sr = load_audio(audio_path, sample_rate)
    duration = float(audio.numel()) / float(sr)
    total_frames = max(1, int(math.ceil(duration * frame_hz)))
    hidden_dim = int(model.config.hidden_size)
    output_sum = np.zeros((total_frames, hidden_dim), dtype=np.float32)
    output_count = np.zeros((total_frames, 1), dtype=np.float32)

    chunk_samples = max(1, int(round(chunk_sec * sr)))
    hop_samples = max(1, int(round(hop_sec * sr)))
    starts = list(range(0, max(audio.numel() - 1, 1), hop_samples))
    if not starts:
        starts = [0]
    with torch.no_grad():
        for start in starts:
            end = min(audio.numel(), start + chunk_samples)
            if end <= start:
                continue
            chunk = audio[start:end]
            inputs = processor(
                chunk.numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            if layer >= 0:
                hidden = outputs.hidden_states[layer][0]
            else:
                hidden = outputs.last_hidden_state[0]
            aggregate_hidden(
                output_sum=output_sum,
                output_count=output_count,
                hidden=hidden,
                chunk_start=float(start) / float(sr),
                frame_hz=frame_hz,
            )
            if end == audio.numel():
                break

    features = output_sum / np.maximum(output_count, 1.0)
    missing = output_count[:, 0] <= 0
    if missing.any():
        valid_idx = np.flatnonzero(~missing)
        if valid_idx.size == 0:
            raise RuntimeError(f"MERT produced no frames for {audio_path}")
        all_idx = np.arange(total_frames)
        for dim in range(features.shape[1]):
            features[missing, dim] = np.interp(all_idx[missing], valid_idx, features[valid_idx, dim])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        frame_hz=np.asarray(frame_hz, dtype=np.float32),
        duration=np.asarray(duration, dtype=np.float32),
        audio_path=str(audio_path),
    )


def build_items(cfg, split: str, custom_scp: str | None = None, custom_lyrics_dir: str | None = None):
    if custom_scp:
        if not custom_lyrics_dir:
            raise ValueError("--custom_lyrics_dir is required with --custom_scp")
        return build_hx_bench_items(custom_scp, custom_lyrics_dir)
    if split == "train":
        return build_hx_items(cfg.data.train_hx) + build_hook_items(cfg.data.train_hook)
    if split == "val":
        return build_hx_items(cfg.data.val_hx)
    if split == "hx_bench":
        return build_hx_bench_items(cfg.data.hx_bench_scp, cfg.data.hx_bench_lyrics_dir)
    raise ValueError(f"Unknown split: {split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "hx_bench"])
    parser.add_argument("--custom_scp", default=None)
    parser.add_argument("--custom_lyrics_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model_dir = cfg.model.mert_model_dir
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    processor = AutoFeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
    model.eval()
    sample_rate = int(getattr(processor, "sampling_rate", cfg.mert.sample_rate))

    out_dir = Path(args.output_dir or cfg.data.mert_feature_dir)
    items = build_items(cfg, args.split, args.custom_scp, args.custom_lyrics_dir)
    for item in tqdm(items, desc=f"extract_mert:{args.split}"):
        out_path = out_dir / f"{item['song_id']}.npz"
        if args.skip_existing and out_path.exists():
            continue
        audio_path = item["audio_path"]
        if item["dataset"] == "hook":
            audio_path = str(Path(cfg.data.train_hook.hook_audio_root) / (Path(audio_path).stem + ".mp3"))
        extract_one(
            audio_path=audio_path,
            out_path=out_path,
            processor=processor,
            model=model,
            device=device,
            sample_rate=sample_rate,
            frame_hz=float(cfg.mert.frame_hz),
            chunk_sec=float(cfg.mert.chunk_sec),
            hop_sec=float(cfg.mert.hop_sec),
            layer=int(cfg.mert.layer),
        )


if __name__ == "__main__":
    main()

