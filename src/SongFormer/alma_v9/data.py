from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import IGNORE_INDEX, LABELS, label_to_id, normalize_label


def read_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                ids.add(item)
    return ids


def read_scp(path: str | Path) -> List[Path]:
    paths: List[Path] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                paths.append(Path(item))
    return paths


def load_hx_label_map(jsonl_path: str | Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj["id"]] = obj
    return out


def load_hook_segments(jsonl_path: str | Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            stem = Path(obj["ori_audio_path"]).stem
            grouped[stem].append(obj)
    for segments in grouped.values():
        segments.sort(key=lambda x: (float(x["segment_start"]), float(x["segment_end"])))
    return dict(grouped)


def relocate_hook_audio(stem: str, hook_audio_root: str | Path) -> str:
    root = Path(hook_audio_root)
    for suffix in (".mp3", ".wav", ".flac", ".m4a"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(root / f"{stem}.mp3")


def flatten_soulx_lines(path: str | Path | None) -> List[dict]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    lines: List[dict] = []
    for seg in obj.get("segments", []):
        seg_lines = seg.get("lines") or []
        if not seg_lines and seg.get("text"):
            seg_lines = [seg]
        for line in seg_lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            start_ms = line.get("start_ms", seg.get("start_ms"))
            end_ms = line.get("end_ms", seg.get("end_ms"))
            if start_ms is None or end_ms is None:
                continue
            start = float(start_ms) / 1000.0
            end = float(end_ms) / 1000.0
            if end <= start:
                continue
            lines.append({"text": text, "start": start, "end": end})
    lines.sort(key=lambda x: (x["start"], x["end"]))
    return lines


def widen_events(events: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return events.astype(np.float32)
    out = np.zeros_like(events, dtype=np.float32)
    idxs = np.flatnonzero(events > 0)
    for idx in idxs:
        left = max(0, idx - radius)
        right = min(len(events), idx + radius + 1)
        if right <= left:
            continue
        win = np.arange(left, right)
        weights = 1.0 - (np.abs(win - idx) / float(radius + 1))
        out[left:right] = np.maximum(out[left:right], weights.astype(np.float32))
    return out


def build_hx_items(cfg) -> List[dict]:
    label_map = load_hx_label_map(cfg.hx_label_jsonl)
    split_ids = read_ids(cfg.hx_split_ids)
    items = []
    for song_id in sorted(split_ids):
        obj = label_map.get(song_id)
        if obj is None:
            continue
        audio_path = Path(cfg.hx_audio_root) / f"{song_id}.wav"
        lyrics_path = Path(cfg.hx_lyrics_dir) / f"{song_id}.json"
        items.append(
            {
                "song_id": song_id,
                "dataset": "hx",
                "audio_path": str(audio_path),
                "lyrics_path": str(lyrics_path),
                "duration": float(obj.get("duration", 0.0)),
                "hx_labels": obj.get("labels", []),
                "hook_segments": None,
            }
        )
    return items


def build_hook_items(cfg) -> List[dict]:
    grouped = load_hook_segments(cfg.hook_label_jsonl)
    split_ids = read_ids(cfg.hook_split_ids)
    items = []
    for song_id in sorted(split_ids):
        segments = grouped.get(song_id)
        if not segments:
            continue
        audio_path = relocate_hook_audio(song_id, cfg.hook_audio_root)
        lyrics_path = Path(cfg.hook_lyrics_dir) / f"{song_id}.json"
        duration = max(float(x["segment_end"]) for x in segments)
        items.append(
            {
                "song_id": song_id,
                "dataset": "hook",
                "audio_path": audio_path,
                "lyrics_path": str(lyrics_path),
                "duration": duration,
                "hx_labels": None,
                "hook_segments": segments,
            }
        )
    return items


def build_hx_bench_items(scp_path: str | Path, lyrics_dir: str | Path) -> List[dict]:
    items = []
    for audio_path in read_scp(scp_path):
        song_id = audio_path.stem
        items.append(
            {
                "song_id": song_id,
                "dataset": "hx_bench",
                "audio_path": str(audio_path),
                "lyrics_path": str(Path(lyrics_dir) / f"{song_id}.json"),
                "duration": 0.0,
                "hx_labels": None,
                "hook_segments": None,
            }
        )
    return items


def add_feature_paths(items: List[dict], feature_dir: str | Path) -> List[dict]:
    root = Path(feature_dir)
    for item in items:
        item["feature_path"] = str(root / f"{item['song_id']}.npz")
    return items


class AlmaV9Dataset(Dataset):
    def __init__(
        self,
        items: List[dict],
        tokenizer,
        feature_dir: str | Path,
        frame_hz: float,
        slice_dur: float,
        train: bool,
        max_text_len: int,
        line_max_len: int,
        boundary_widen_sec: float,
        line_token: str = "<LINE>",
    ):
        self.items = add_feature_paths(list(items), feature_dir)
        self.tokenizer = tokenizer
        self.frame_hz = float(frame_hz)
        self.slice_dur = float(slice_dur)
        self.train = bool(train)
        self.max_text_len = int(max_text_len)
        self.line_max_len = int(line_max_len)
        self.boundary_radius = int(round(float(boundary_widen_sec) * self.frame_hz))
        self.line_token_id = tokenizer.convert_tokens_to_ids(line_token)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def audio_dim(self) -> int:
        for item in self.items:
            path = Path(item["feature_path"])
            if path.exists():
                arr = np.load(path)["features"]
                return int(arr.shape[-1])
        raise FileNotFoundError("No cached MERT feature files found for this dataset")

    def _choose_crop(self, duration: float) -> tuple[float, float]:
        if (not self.train) or duration <= self.slice_dur:
            return 0.0, max(duration, self.slice_dur if self.train else duration)
        start = random.uniform(0.0, max(0.0, duration - self.slice_dur))
        return start, start + self.slice_dur

    def _load_features(self, item: dict, crop_start: float, crop_end: float) -> np.ndarray:
        data = np.load(item["feature_path"])
        features = np.asarray(data["features"], dtype=np.float32)
        source_hz = float(data.get("frame_hz", self.frame_hz))
        left = max(0, int(math.floor(crop_start * source_hz)))
        right = min(features.shape[0], int(math.ceil(crop_end * source_hz)))
        features = features[left:right]
        if source_hz != self.frame_hz and features.shape[0] > 1:
            target_len = max(1, int(round((crop_end - crop_start) * self.frame_hz)))
            old_x = np.linspace(0.0, 1.0, features.shape[0], endpoint=True)
            new_x = np.linspace(0.0, 1.0, target_len, endpoint=True)
            features = np.stack(
                [np.interp(new_x, old_x, features[:, dim]) for dim in range(features.shape[1])],
                axis=-1,
            ).astype(np.float32)
        return features

    def _build_hx_targets(self, item: dict, crop_start: float, frame_len: int):
        function = np.full((frame_len,), IGNORE_INDEX, dtype=np.int64)
        valid = np.zeros((frame_len,), dtype=bool)
        boundary = np.zeros((frame_len,), dtype=np.float32)
        boundary_valid = np.ones((frame_len,), dtype=bool)

        labels = item.get("hx_labels") or []
        if len(labels) < 2:
            return function, valid, boundary, boundary_valid

        for idx in range(len(labels) - 1):
            start, label = labels[idx]
            end, _ = labels[idx + 1]
            label = normalize_label(label)
            if label == "end" or label not in LABELS:
                continue
            left = max(0, int(math.floor((float(start) - crop_start) * self.frame_hz)))
            right = min(frame_len, int(math.ceil((float(end) - crop_start) * self.frame_hz)))
            if right > left:
                function[left:right] = label_to_id(label)
                valid[left:right] = True
            if crop_start < float(start) < crop_start + frame_len / self.frame_hz:
                bidx = int(round((float(start) - crop_start) * self.frame_hz))
                if 0 <= bidx < frame_len:
                    boundary[bidx] = 1.0
        boundary = widen_events(boundary, self.boundary_radius)
        return function, valid, boundary, boundary_valid

    def _build_hook_targets(self, item: dict, crop_start: float, frame_len: int):
        function = np.full((frame_len,), IGNORE_INDEX, dtype=np.int64)
        valid = np.zeros((frame_len,), dtype=bool)
        boundary = np.zeros((frame_len,), dtype=np.float32)
        boundary_valid = np.zeros((frame_len,), dtype=bool)
        crop_end = crop_start + frame_len / self.frame_hz

        for seg in item.get("hook_segments") or []:
            labels = seg.get("label") or []
            if not labels:
                continue
            label = normalize_label(labels[0])
            if label == "NO_LABEL" or label not in LABELS:
                continue
            start = float(seg["segment_start"])
            end = float(seg["segment_end"])
            left = max(0, int(math.floor((start - crop_start) * self.frame_hz)))
            right = min(frame_len, int(math.ceil((end - crop_start) * self.frame_hz)))
            if right <= left:
                continue
            function[left:right] = label_to_id(label)
            valid[left:right] = True
            boundary_valid[left:right] = True
            for t in (start, end):
                if crop_start < t < crop_end:
                    bidx = int(round((t - crop_start) * self.frame_hz))
                    if 0 <= bidx < frame_len:
                        boundary[bidx] = 1.0
        boundary = widen_events(boundary, self.boundary_radius)
        if not boundary_valid.any():
            boundary_valid[:] = False
        return function, valid, boundary, boundary_valid

    def _tokenize_lines(self, lines: List[dict]):
        input_ids = [self.tokenizer.cls_token_id]
        line_positions = []
        kept_lines = []
        for line in lines:
            token_ids = self.tokenizer.encode(
                line["text"],
                add_special_tokens=False,
                max_length=self.line_max_len,
                truncation=True,
            )
            needed = 1 + len(token_ids)
            if len(input_ids) + needed + 1 > self.max_text_len:
                break
            line_positions.append(len(input_ids))
            input_ids.append(self.line_token_id)
            input_ids.extend(token_ids)
            kept_lines.append(line)
        if not kept_lines:
            input_ids = [self.tokenizer.cls_token_id, self.line_token_id, self.tokenizer.sep_token_id]
            line_positions = [1]
            kept_lines = [{"text": "", "start": 0.0, "end": 0.0}]
        else:
            input_ids.append(self.tokenizer.sep_token_id)
        attn = [1] * len(input_ids)
        starts = [float(x["start"]) for x in kept_lines]
        ends = [float(x["end"]) for x in kept_lines]
        line_valid = [1 if x["end"] > x["start"] and x["text"] else 0 for x in kept_lines]
        return (
            np.asarray(input_ids, dtype=np.int64),
            np.asarray(attn, dtype=np.int64),
            np.asarray(line_positions, dtype=np.int64),
            np.asarray(starts, dtype=np.float32),
            np.asarray(ends, dtype=np.float32),
            np.asarray(line_valid, dtype=np.bool_),
        )

    def __getitem__(self, idx: int):
        item = self.items[idx]
        feature_meta = np.load(item["feature_path"])
        duration = float(feature_meta.get("duration", item.get("duration", 0.0)))
        if duration <= 0:
            duration = feature_meta["features"].shape[0] / float(feature_meta.get("frame_hz", self.frame_hz))
        crop_start, crop_end = self._choose_crop(duration)
        features = self._load_features(item, crop_start, crop_end)
        frame_len = int(features.shape[0])

        if item["dataset"] == "hook":
            function, function_valid, boundary, boundary_valid = self._build_hook_targets(
                item, crop_start, frame_len
            )
        elif item.get("hx_labels"):
            function, function_valid, boundary, boundary_valid = self._build_hx_targets(
                item, crop_start, frame_len
            )
        else:
            function = np.full((frame_len,), IGNORE_INDEX, dtype=np.int64)
            function_valid = np.zeros((frame_len,), dtype=bool)
            boundary = np.zeros((frame_len,), dtype=np.float32)
            boundary_valid = np.zeros((frame_len,), dtype=bool)

        lines = flatten_soulx_lines(item.get("lyrics_path"))
        (
            input_ids,
            attention_mask,
            line_positions,
            line_starts,
            line_ends,
            line_valid,
        ) = self._tokenize_lines(lines)

        return {
            "song_id": item["song_id"],
            "audio_features": torch.from_numpy(features),
            "function_target": torch.from_numpy(function),
            "function_valid": torch.from_numpy(function_valid),
            "boundary_target": torch.from_numpy(boundary),
            "boundary_valid": torch.from_numpy(boundary_valid),
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "line_positions": torch.from_numpy(line_positions),
            "line_starts": torch.from_numpy(line_starts),
            "line_ends": torch.from_numpy(line_ends),
            "line_valid": torch.from_numpy(line_valid),
            "crop_start": torch.tensor(float(crop_start), dtype=torch.float32),
            "duration": torch.tensor(float(duration), dtype=torch.float32),
        }


def _pad_1d(values: List[torch.Tensor], fill_value, dtype=None):
    max_len = max(int(x.numel()) for x in values)
    if dtype is None:
        dtype = values[0].dtype
    out = torch.full((len(values), max_len), fill_value, dtype=dtype)
    for i, x in enumerate(values):
        out[i, : x.numel()] = x.to(dtype)
    return out


def alma_v9_collate(batch: List[dict]) -> dict:
    audio_dim = batch[0]["audio_features"].shape[-1]
    max_t = max(x["audio_features"].shape[0] for x in batch)
    audio = torch.zeros((len(batch), max_t, audio_dim), dtype=torch.float32)
    seq_valid = torch.zeros((len(batch), max_t), dtype=torch.bool)
    function = torch.full((len(batch), max_t), IGNORE_INDEX, dtype=torch.long)
    function_valid = torch.zeros((len(batch), max_t), dtype=torch.bool)
    boundary = torch.zeros((len(batch), max_t), dtype=torch.float32)
    boundary_valid = torch.zeros((len(batch), max_t), dtype=torch.bool)
    for i, item in enumerate(batch):
        t = item["audio_features"].shape[0]
        audio[i, :t] = item["audio_features"]
        seq_valid[i, :t] = True
        function[i, :t] = item["function_target"]
        function_valid[i, :t] = item["function_valid"]
        boundary[i, :t] = item["boundary_target"]
        boundary_valid[i, :t] = item["boundary_valid"]

    return {
        "song_ids": [x["song_id"] for x in batch],
        "audio_features": audio,
        "seq_valid": seq_valid,
        "function_target": function,
        "function_valid": function_valid,
        "boundary_target": boundary,
        "boundary_valid": boundary_valid,
        "input_ids": _pad_1d([x["input_ids"] for x in batch], 0, torch.long),
        "attention_mask": _pad_1d([x["attention_mask"] for x in batch], 0, torch.long),
        "line_positions": _pad_1d([x["line_positions"] for x in batch], -1, torch.long),
        "line_starts": _pad_1d([x["line_starts"] for x in batch], 0.0, torch.float32),
        "line_ends": _pad_1d([x["line_ends"] for x in batch], 0.0, torch.float32),
        "line_valid": _pad_1d([x["line_valid"] for x in batch], False, torch.bool),
        "crop_start": torch.stack([x["crop_start"] for x in batch]),
        "duration": torch.stack([x["duration"] for x in batch]),
    }

