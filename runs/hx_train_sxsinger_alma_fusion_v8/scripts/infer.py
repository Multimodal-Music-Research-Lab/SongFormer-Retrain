#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

# Make third_party imports work before importing MusicFM.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_SONGFORMER = REPO_ROOT / "src" / "SongFormer"
CANONICAL_THIRD_PARTY = Path("/home/hbli/songformer/repo/SongFormer/src/third_party")
sys.path.insert(0, str(REPO_ROOT / "src" / "third_party"))
sys.path.insert(0, str(CANONICAL_THIRD_PARTY))

import librosa
import torch
from ema_pytorch import EMA
from loguru import logger
from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

# ----------------------------
# Robust path resolution
# ----------------------------
THIS_FILE = Path(__file__).resolve()

# parents[0]=scripts, [1]=hx_retrain_v1, [2]=runs, [3]=SongFormer(repo root)
REPO_ROOT = THIS_FILE.parents[3]
SRC_SONGFORMER = REPO_ROOT / "src" / "SongFormer"

if not SRC_SONGFORMER.exists():
    raise FileNotFoundError(f"Cannot find src/SongFormer at: {SRC_SONGFORMER}")

# Make sure imports like "from dataset..." work
sys.path.insert(0, str(SRC_SONGFORMER))

# Absolute MusicFM ckpt directory
MUSICFM_HOME_PATH = "/home/hbli/songformer/repo/SongFormer/src/SongFormer/ckpts/MusicFM"

BEFORE_DOWNSAMPLING_FRAME_RATES = 25
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333

# Keep defaults; you can still change by editing here if needed
DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]

TIME_DUR = 420
INPUT_SAMPLING_RATE = 24000

from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
from postprocessing.functional import postprocess_functional_structure


def get_processed_ids(output_path):
    """Get already processed IDs from output directory"""
    if not os.path.exists(output_path):
        return set()
    ids = os.listdir(output_path)
    ret = []
    for x in ids:
        if x.endswith(".json"):
            ret.append(x.replace(".json", ""))
    return set(ret)


def get_processing_ids(input_path, processed_ids_set):
    """Get IDs to be processed from input scp file (one audio path per line)"""
    ret = []
    with open(input_path) as f:
        for line in f:
            if line.strip() and Path(line.strip()).stem not in processed_ids_set:
                ret.append(line.strip())
    return ret

# =========================
# [ADDED FOR LYRICS V2]
# SoulX-Singer line-level npz helpers for function-head-only local semantics.
# =========================
def get_zero_lyrics_condition(target_len, lyrics_input_dim):
    zero_global = np.zeros((lyrics_input_dim,), dtype=np.float32)
    zero_local = np.zeros((target_len, lyrics_input_dim), dtype=np.float32)
    return zero_global, zero_local, 0.0


def validate_lyrics_units(lyrics_units, feat_dim, source_path, field_name):
    lyrics_units = np.asarray(lyrics_units, dtype=np.float32)
    if lyrics_units.ndim != 2:
        raise ValueError(f"{field_name} ndim mismatch in {source_path}: got {lyrics_units.ndim}, expected 2")
    if lyrics_units.shape[1] != feat_dim:
        raise ValueError(
            f"{field_name} dim mismatch in {source_path}: "
            f"got {lyrics_units.shape[1]}, expected {feat_dim}"
        )
    return lyrics_units


def validate_global_embedding(arr, feat_dim, source_path):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] != feat_dim:
        raise ValueError(f"global_emb dim mismatch in {source_path}: got {arr.shape[0]}, expected {feat_dim}")
    return arr


def validate_time_vector(arr, source_path, field_name):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} ndim mismatch in {source_path}: got {arr.ndim}, expected 1")
    return arr


def load_sync_lyrics_npz(audio_path, lyrics_embedding_dir, lyrics_input_dim, lyrics_logits_dim):
    if lyrics_input_dim is None:
        raise ValueError("lyrics_input_dim must not be None when lyrics is enabled")
    if lyrics_embedding_dir is None or str(lyrics_embedding_dir).strip() == "":
        return None

    song_stem = Path(audio_path).stem
    lyrics_npz_path = Path(lyrics_embedding_dir) / f"{song_stem}.npz"
    if not lyrics_npz_path.exists():
        return None

    lyrics_npz = np.load(lyrics_npz_path, allow_pickle=False)
    required_fields = ["line_embs", "line_start_secs", "line_end_secs"]
    for field in required_fields:
        if field not in lyrics_npz:
            raise KeyError(f"'{field}' not found in {lyrics_npz_path}")

    line_embs = validate_lyrics_units(
        lyrics_npz["line_embs"], lyrics_input_dim, lyrics_npz_path, "line_embs"
    ).astype(np.float32)
    line_start_secs = validate_time_vector(
        lyrics_npz["line_start_secs"], lyrics_npz_path, "line_start_secs"
    ).astype(np.float32)
    line_end_secs = validate_time_vector(
        lyrics_npz["line_end_secs"], lyrics_npz_path, "line_end_secs"
    ).astype(np.float32)
    if "line_logits" in lyrics_npz:
        line_logits = validate_lyrics_units(
            lyrics_npz["line_logits"], lyrics_logits_dim, lyrics_npz_path, "line_logits"
        ).astype(np.float32)
    else:
        line_logits = np.zeros((len(line_embs), lyrics_logits_dim), dtype=np.float32)
    if "line_boundary_logits" in lyrics_npz:
        line_boundary_logits = validate_time_vector(
            lyrics_npz["line_boundary_logits"], lyrics_npz_path, "line_boundary_logits"
        ).astype(np.float32)
    else:
        line_boundary_logits = np.zeros((len(line_embs),), dtype=np.float32)

    if not (len(line_embs) == len(line_start_secs) == len(line_end_secs)):
        raise ValueError(
            f"line field length mismatch in {lyrics_npz_path}: "
            f"{len(line_embs)} / {len(line_start_secs)} / {len(line_end_secs)}"
        )
    if not (len(line_logits) == len(line_embs) == len(line_boundary_logits)):
        raise ValueError(
            f"lyrics logits length mismatch in {lyrics_npz_path}: "
            f"{len(line_logits)} / {len(line_embs)} / {len(line_boundary_logits)}"
        )

    return {
        "line_embs": line_embs,
        "line_logits": line_logits,
        "line_boundary_logits": line_boundary_logits,
        "line_start_secs": line_start_secs,
        "line_end_secs": line_end_secs,
    }


def flatten_soulx_lines(lyrics_path):
    obj = json.loads(Path(lyrics_path).read_text())
    lines = []
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


def load_sync_lyrics_json(audio_path, lyrics_json_dir):
    if lyrics_json_dir is None or str(lyrics_json_dir).strip() == "":
        return None
    lyrics_json_path = Path(lyrics_json_dir) / f"{Path(audio_path).stem}.json"
    if not lyrics_json_path.exists():
        return None
    lines = flatten_soulx_lines(lyrics_json_path)
    if not lines:
        return None
    return {
        "line_texts": [x["text"] for x in lines],
        "line_start_secs": np.asarray([x["start"] for x in lines], dtype=np.float32),
        "line_end_secs": np.asarray([x["end"] for x in lines], dtype=np.float32),
    }


def tokenize_lyrics_lines(tokenizer, line_token_id, line_texts, max_length, line_max_length):
    input_ids = [tokenizer.cls_token_id]
    line_positions = []
    for text in line_texts:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=line_max_length,
            truncation=True,
        )
        if len(input_ids) + 1 + len(ids) + 1 > max_length:
            break
        line_positions.append(len(input_ids))
        input_ids.append(line_token_id)
        input_ids.extend(ids)
    input_ids.append(tokenizer.sep_token_id)
    if not line_positions:
        input_ids = [tokenizer.cls_token_id, line_token_id, tokenizer.sep_token_id]
        line_positions = [1]
    input_ids = np.asarray(input_ids, dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    global_attention_mask = np.zeros_like(input_ids, dtype=np.int64)
    global_attention_mask[0] = 1
    line_positions = np.asarray(line_positions, dtype=np.int64)
    line_text_masks = np.zeros((len(line_positions),), dtype=bool)
    return input_ids, attention_mask, global_attention_mask, line_positions, line_text_masks


def build_local_lyrics_condition(
    lyrics_data,
    target_len,
    chunk_start_time,
    chunk_end_time,
    lyrics_input_dim,
    local_window_sec,
    local_gaussian_sigma_sec,
):
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if lyrics_data is None:
        return get_zero_lyrics_condition(target_len, lyrics_input_dim)

    global_emb = lyrics_data["global_emb"].astype(np.float32)
    local_embeddings = np.zeros((target_len, lyrics_input_dim), dtype=np.float32)

    chunk_start_time = float(chunk_start_time)
    chunk_end_time = float(chunk_end_time)
    chunk_width = max(chunk_end_time - chunk_start_time, 1e-6)
    frame_dur = chunk_width / float(target_len)
    window = float(local_window_sec)
    sigma = max(float(local_gaussian_sigma_sec), 1e-6)

    line_embs = lyrics_data["line_embs"].astype(np.float32)
    line_start_secs = lyrics_data["line_start_secs"].astype(np.float32)
    line_end_secs = lyrics_data["line_end_secs"].astype(np.float32)
    line_centers = 0.5 * (line_start_secs + line_end_secs)

    keep = (line_end_secs > chunk_start_time - window) & (line_start_secs < chunk_end_time + window)
    if keep.sum() <= 0:
        return global_emb, local_embeddings, 1.0

    local_line_embs = line_embs[keep]
    local_line_centers = line_centers[keep]

    for frame_idx in range(target_len):
        frame_center = chunk_start_time + (frame_idx + 0.5) * frame_dur
        dists = np.abs(local_line_centers - frame_center)
        use = dists <= window
        if use.sum() <= 0:
            continue
        weights = np.exp(-0.5 * (dists[use] / sigma) ** 2).astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-8:
            continue
        local_embeddings[frame_idx] = np.sum(local_line_embs[use] * weights[:, None], axis=0) / weight_sum

    return global_emb, local_embeddings, 1.0



def build_line_time_features(start_secs, end_secs, chunk_start_time, chunk_end_time, feat_dim):
    start_secs = np.asarray(start_secs, dtype=np.float32).reshape(-1)
    end_secs = np.asarray(end_secs, dtype=np.float32).reshape(-1)
    chunk_width = max(float(chunk_end_time) - float(chunk_start_time), 1e-6)
    chunk_center = 0.5 * (float(chunk_start_time) + float(chunk_end_time))
    centers = 0.5 * (start_secs + end_secs)
    durations = np.maximum(end_secs - start_secs, 1e-6)
    overlap = np.maximum(
        0.0,
        np.minimum(end_secs, float(chunk_end_time)) - np.maximum(start_secs, float(chunk_start_time)),
    )
    overlap_ratio = overlap / durations
    inside_flag = ((centers >= float(chunk_start_time)) & (centers <= float(chunk_end_time))).astype(np.float32)
    prev_gap = np.zeros_like(start_secs, dtype=np.float32)
    next_gap = np.zeros_like(start_secs, dtype=np.float32)
    if len(start_secs) > 1:
        prev_gap[1:] = np.maximum(start_secs[1:] - end_secs[:-1], 0.0)
        next_gap[:-1] = np.maximum(start_secs[1:] - end_secs[:-1], 0.0)
    feats = np.stack(
        [
            (start_secs - float(chunk_start_time)) / chunk_width,
            (end_secs - float(chunk_start_time)) / chunk_width,
            (centers - chunk_center) / chunk_width,
            durations / chunk_width,
            overlap_ratio,
            inside_flag,
            prev_gap / chunk_width,
            next_gap / chunk_width,
        ],
        axis=1,
    ).astype(np.float32)
    if feats.shape[1] != feat_dim:
        raise ValueError(f"lyrics_time_feat_dim mismatch: built {feats.shape[1]}, expected {feat_dim}")
    return feats


def build_line_repeat_features(
    line_embs,
    line_start_secs,
    repeat_feat_dim,
    repeat_top_k,
    repeat_exclude_neighbors,
    repeat_similarity_threshold,
):
    line_count = len(line_embs)
    repeat_embs = np.zeros_like(line_embs, dtype=np.float32)
    repeat_feats = np.zeros((line_count, repeat_feat_dim), dtype=np.float32)
    if line_count <= 1:
        return repeat_embs, repeat_feats
    normalized = line_embs / np.maximum(np.linalg.norm(line_embs, axis=1, keepdims=True), 1e-8)
    similarities = normalized @ normalized.T
    song_width = max(float(line_start_secs.max() - line_start_secs.min()), 1e-6)
    for line_idx in range(line_count):
        candidate = similarities[line_idx].copy()
        local_left = max(0, line_idx - repeat_exclude_neighbors)
        local_right = min(line_count, line_idx + repeat_exclude_neighbors + 1)
        candidate[local_left:local_right] = -np.inf
        valid = np.isfinite(candidate)
        if not valid.any():
            continue
        top_k = min(repeat_top_k, int(valid.sum()))
        top_indices = np.argpartition(candidate, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(candidate[top_indices])[::-1]]
        top_scores = candidate[top_indices]
        positive_scores = np.maximum(top_scores, 0.0)
        if float(positive_scores.sum()) > 1e-8:
            repeat_embs[line_idx] = (
                line_embs[top_indices] * positive_scores[:, None]
            ).sum(axis=0) / float(positive_scores.sum())
        best_idx = int(top_indices[0])
        repeat_feats[line_idx] = np.asarray(
            [
                float(top_scores[0]),
                float(top_scores.mean()),
                float((candidate[valid] >= repeat_similarity_threshold).sum()) / max(float(valid.sum()), 1.0),
                abs(float(line_start_secs[best_idx] - line_start_secs[line_idx])) / song_width,
            ],
            dtype=np.float32,
        )
    return repeat_embs, repeat_feats


def build_text_repeat_features(line_texts, repeat_feat_dim, repeat_top_k, repeat_exclude_neighbors, repeat_similarity_threshold):
    repeat_feats = np.zeros((len(line_texts), repeat_feat_dim), dtype=np.float32)
    if len(line_texts) <= 1:
        return repeat_feats
    token_sets = []
    normalized = []
    for text in line_texts:
        toks = [x for x in text.lower().replace("'", " ").split() if x]
        token_sets.append(set(toks))
        normalized.append(" ".join(toks))
    for i in range(len(line_texts)):
        sims = []
        for j in range(len(line_texts)):
            if abs(i - j) <= repeat_exclude_neighbors:
                continue
            if normalized[i] and normalized[i] == normalized[j]:
                sim = 1.0
            else:
                union = token_sets[i] | token_sets[j]
                sim = len(token_sets[i] & token_sets[j]) / max(float(len(union)), 1.0)
            sims.append((sim, j))
        if not sims:
            continue
        sims.sort(reverse=True)
        top = sims[:repeat_top_k]
        best_sim, best_idx = top[0]
        repeat_feats[i] = np.asarray(
            [
                best_sim,
                float(np.mean([x[0] for x in top])),
                float(sum(x[0] >= repeat_similarity_threshold for x in sims)) / max(float(len(sims)), 1.0),
                abs(best_idx - i) / max(float(len(line_texts) - 1), 1.0),
            ],
            dtype=np.float32,
        )
    return repeat_feats


def build_frame_alignment(line_start_secs, line_end_secs, target_len, chunk_start_time, chunk_end_time, feat_dim):
    target_len = int(target_len)
    chunk_width = max(float(chunk_end_time) - float(chunk_start_time), 1e-6)
    frame_dur = chunk_width / max(float(target_len), 1.0)
    frame_centers = float(chunk_start_time) + (np.arange(target_len, dtype=np.float32) + 0.5) * frame_dur
    feats = np.zeros((target_len, feat_dim), dtype=np.float32)
    frame_line_indices = np.full((target_len,), -1, dtype=np.int64)
    frame_active_masks = np.zeros((target_len,), dtype=bool)
    line_start_secs = np.asarray(line_start_secs, dtype=np.float32).reshape(-1)
    line_end_secs = np.asarray(line_end_secs, dtype=np.float32).reshape(-1)
    if len(line_start_secs) == 0:
        return feats, frame_line_indices, frame_active_masks
    line_centers = 0.5 * (line_start_secs + line_end_secs)
    for idx, frame_time in enumerate(frame_centers):
        covering = np.flatnonzero((line_start_secs <= frame_time) & (frame_time <= line_end_secs))
        inside = len(covering) > 0
        if inside:
            nearest_idx = int(covering[np.argmin(np.abs(line_centers[covering] - frame_time))])
            frame_line_indices[idx] = nearest_idx
            frame_active_masks[idx] = True
            line_duration = max(float(line_end_secs[nearest_idx] - line_start_secs[nearest_idx]), 1e-6)
            relative_position = (float(frame_time) - float(line_start_secs[nearest_idx])) / line_duration
            start_pulse = float(relative_position <= min(0.15, frame_dur / line_duration + 1e-6))
            end_pulse = float(relative_position >= max(0.85, 1.0 - frame_dur / line_duration - 1e-6))
            duration_ratio = line_duration / chunk_width
        else:
            relative_position = 0.0
            start_pulse = 0.0
            end_pulse = 0.0
            duration_ratio = 0.0
        density = np.mean(np.abs(line_centers - frame_time) <= 5.0)
        feats[idx] = np.asarray(
            [relative_position, start_pulse, end_pulse, float(inside), float(density), duration_ratio],
            dtype=np.float32,
        )
    return feats, frame_line_indices, frame_active_masks


def build_lyrics_token_condition(
    lyrics_data,
    target_len,
    chunk_start_time,
    chunk_end_time,
    lyrics_input_dim,
    lyrics_logits_dim,
    lyrics_time_feat_dim,
    lyrics_frame_time_feat_dim,
    lyrics_repeat_feat_dim,
    lyrics_context_window_sec,
    lyrics_repeat_top_k,
    lyrics_repeat_exclude_neighbors,
    lyrics_repeat_similarity_threshold,
):
    zero_line = np.zeros((1, lyrics_input_dim), dtype=np.float32)
    zero_line_logits = np.zeros((1, lyrics_logits_dim), dtype=np.float32)
    zero_line_boundary_logits = np.zeros((1,), dtype=np.float32)
    zero_line_time = np.zeros((1, lyrics_time_feat_dim), dtype=np.float32)
    zero_repeat_line = np.zeros((1, lyrics_input_dim), dtype=np.float32)
    zero_repeat_feat = np.zeros((1, lyrics_repeat_feat_dim), dtype=np.float32)
    zero_line_mask = np.zeros((1,), dtype=bool)
    zero_frame_time = np.zeros((target_len, lyrics_frame_time_feat_dim), dtype=np.float32)
    zero_frame_line_indices = np.full((target_len,), -1, dtype=np.int64)
    zero_frame_active_masks = np.zeros((target_len,), dtype=bool)
    zero_values = (
        zero_line, zero_line_logits, zero_line_boundary_logits, zero_line_time, zero_repeat_line, zero_repeat_feat,
        zero_line_mask, zero_frame_time, zero_frame_line_indices, zero_frame_active_masks, 0.0
    )
    if lyrics_data is None:
        return zero_values

    line_embs = lyrics_data["line_embs"].astype(np.float32)
    line_logits = lyrics_data["line_logits"].astype(np.float32)
    line_boundary_logits = lyrics_data["line_boundary_logits"].astype(np.float32)
    line_start_secs = lyrics_data["line_start_secs"].astype(np.float32)
    line_end_secs = lyrics_data["line_end_secs"].astype(np.float32)
    context_window = float(lyrics_context_window_sec)
    keep = (
        (line_end_secs > float(chunk_start_time) - context_window)
        & (line_start_secs < float(chunk_end_time) + context_window)
    )
    if keep.sum() <= 0:
        return zero_values

    repeat_embs, repeat_feats = build_line_repeat_features(
        line_embs,
        line_start_secs,
        lyrics_repeat_feat_dim,
        lyrics_repeat_top_k,
        lyrics_repeat_exclude_neighbors,
        lyrics_repeat_similarity_threshold,
    )

    local_line_embs = line_embs[keep].astype(np.float32)
    local_line_logits = line_logits[keep].astype(np.float32)
    local_line_boundary_logits = line_boundary_logits[keep].astype(np.float32)
    local_repeat_embs = repeat_embs[keep].astype(np.float32)
    local_repeat_feats = repeat_feats[keep].astype(np.float32)
    local_start_secs = line_start_secs[keep].astype(np.float32)
    local_end_secs = line_end_secs[keep].astype(np.float32)
    line_time = build_line_time_features(
        local_start_secs,
        local_end_secs,
        chunk_start_time,
        chunk_end_time,
        lyrics_time_feat_dim,
    )
    frame_time, frame_line_indices, frame_active_masks = build_frame_alignment(
        local_start_secs,
        local_end_secs,
        target_len,
        chunk_start_time,
        chunk_end_time,
        lyrics_frame_time_feat_dim,
    )
    line_mask = np.zeros((local_line_embs.shape[0],), dtype=bool)
    return (
        local_line_embs,
        local_line_logits,
        local_line_boundary_logits,
        line_time,
        local_repeat_embs,
        local_repeat_feats,
        line_mask,
        frame_time,
        frame_line_indices,
        frame_active_masks,
        1.0,
    )


def build_raw_lyrics_token_condition(
    lyrics_data,
    tokenizer,
    line_token_id,
    target_len,
    chunk_start_time,
    chunk_end_time,
    lyrics_input_dim,
    lyrics_logits_dim,
    lyrics_time_feat_dim,
    lyrics_frame_time_feat_dim,
    lyrics_repeat_feat_dim,
    lyrics_context_window_sec,
    lyrics_repeat_top_k,
    lyrics_repeat_exclude_neighbors,
    lyrics_repeat_similarity_threshold,
    lyrics_longformer_max_length,
    lyrics_longformer_line_max_length,
):
    base = build_lyrics_token_condition(
        lyrics_data=None,
        target_len=target_len,
        chunk_start_time=chunk_start_time,
        chunk_end_time=chunk_end_time,
        lyrics_input_dim=lyrics_input_dim,
        lyrics_logits_dim=lyrics_logits_dim,
        lyrics_time_feat_dim=lyrics_time_feat_dim,
        lyrics_frame_time_feat_dim=lyrics_frame_time_feat_dim,
        lyrics_repeat_feat_dim=lyrics_repeat_feat_dim,
        lyrics_context_window_sec=lyrics_context_window_sec,
        lyrics_repeat_top_k=lyrics_repeat_top_k,
        lyrics_repeat_exclude_neighbors=lyrics_repeat_exclude_neighbors,
        lyrics_repeat_similarity_threshold=lyrics_repeat_similarity_threshold,
    )
    zero_ids, zero_attn, zero_global, zero_positions, zero_line_masks = tokenize_lyrics_lines(
        tokenizer,
        line_token_id,
        [],
        lyrics_longformer_max_length,
        lyrics_longformer_line_max_length,
    )
    if lyrics_data is None:
        return (*base, zero_ids, zero_attn, zero_global, zero_positions, zero_line_masks)

    line_start_secs = lyrics_data["line_start_secs"].astype(np.float32)
    line_end_secs = lyrics_data["line_end_secs"].astype(np.float32)
    line_texts = lyrics_data["line_texts"]
    context_window = float(lyrics_context_window_sec)
    keep = (
        (line_end_secs > float(chunk_start_time) - context_window)
        & (line_start_secs < float(chunk_end_time) + context_window)
    )
    if keep.sum() <= 0:
        return (*base, zero_ids, zero_attn, zero_global, zero_positions, zero_line_masks)

    local_texts = [x for x, flag in zip(line_texts, keep) if flag]
    local_start_secs = line_start_secs[keep].astype(np.float32)
    local_end_secs = line_end_secs[keep].astype(np.float32)
    input_ids, attention_mask, global_attention_mask, line_positions, line_text_masks = tokenize_lyrics_lines(
        tokenizer,
        line_token_id,
        local_texts,
        lyrics_longformer_max_length,
        lyrics_longformer_line_max_length,
    )
    used_lines = min(len(line_positions), len(local_texts))
    local_texts = local_texts[:used_lines]
    local_start_secs = local_start_secs[:used_lines]
    local_end_secs = local_end_secs[:used_lines]

    local_line_embs = np.zeros((used_lines, lyrics_input_dim), dtype=np.float32)
    local_line_logits = np.zeros((used_lines, lyrics_logits_dim), dtype=np.float32)
    local_line_boundary_logits = np.zeros((used_lines,), dtype=np.float32)
    local_repeat_embs = np.zeros_like(local_line_embs, dtype=np.float32)
    local_repeat_feats = build_text_repeat_features(
        local_texts,
        lyrics_repeat_feat_dim,
        lyrics_repeat_top_k,
        lyrics_repeat_exclude_neighbors,
        lyrics_repeat_similarity_threshold,
    )
    line_time = build_line_time_features(
        local_start_secs,
        local_end_secs,
        chunk_start_time,
        chunk_end_time,
        lyrics_time_feat_dim,
    )
    frame_time, frame_line_indices, frame_active_masks = build_frame_alignment(
        local_start_secs,
        local_end_secs,
        target_len,
        chunk_start_time,
        chunk_end_time,
        lyrics_frame_time_feat_dim,
    )
    line_mask = np.zeros((used_lines,), dtype=bool)
    return (
        local_line_embs,
        local_line_logits,
        local_line_boundary_logits,
        line_time,
        local_repeat_embs,
        local_repeat_feats,
        line_mask,
        frame_time,
        frame_line_indices,
        frame_active_masks,
        1.0,
        input_ids,
        attention_mask,
        global_attention_mask,
        line_positions,
        line_text_masks,
    )

def load_checkpoint(checkpoint_path, device=None):
    """Load checkpoint from path (.pt or .safetensors)"""
    if device is None:
        device = "cpu"

    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".pt"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    elif checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = {"model_ema": load_file(checkpoint_path, device=device)}
    else:
        raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")
    return checkpoint


def rule_post_processing(msa_list):
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def inference(rank, queue_input: mp.Queue, queue_output: mp.Queue, args):
    """Run inference on the input audio (compute MuQ/MusicFM on the fly)"""
    device = f"cuda:{rank}"

    # MuQ model loading (this will automatically fetch the checkpoint from huggingface)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    # MusicFM model loading (absolute paths fixed)
    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm = musicfm.to(device)
    musicfm.eval()

    # Custom model loading based on the config
    module = importlib.import_module("models." + str(args.model))
    Model = getattr(module, "Model")

    # --- FIX 1: config path resolution ---
    # allow absolute config path; otherwise try cwd, repo root, then src/SongFormer/configs.
    cfg_path = Path(args.config_path)
    if not cfg_path.is_absolute():
        candidates = [
            (Path.cwd() / cfg_path).resolve(),
            (REPO_ROOT / cfg_path).resolve(),
            (SRC_SONGFORMER / "configs" / cfg_path).resolve(),
        ]
        cfg_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    hp = OmegaConf.load(str(cfg_path))
    model = Model(hp)

    # =========================
    # [ADDED FOR LYRICS]
    # Optional lyrics branch settings from config
    # =========================
    use_lyrics = getattr(hp, "use_lyrics", False)
    lyrics_input_dim = getattr(hp, "lyrics_input_dim", None)
    lyrics_logits_dim = getattr(hp, "lyrics_logits_dim", 8)
    lyrics_time_feat_dim = getattr(hp, "lyrics_time_feat_dim", 8)
    lyrics_frame_time_feat_dim = getattr(hp, "lyrics_frame_time_feat_dim", 6)
    lyrics_repeat_feat_dim = getattr(hp, "lyrics_repeat_feat_dim", 4)
    lyrics_context_window_sec = getattr(hp, "lyrics_context_window_sec", 30.0)
    lyrics_repeat_top_k = getattr(hp, "lyrics_repeat_top_k", 3)
    lyrics_repeat_exclude_neighbors = getattr(hp, "lyrics_repeat_exclude_neighbors", 2)
    lyrics_repeat_similarity_threshold = getattr(hp, "lyrics_repeat_similarity_threshold", 0.8)
    lyrics_use_raw_text = getattr(hp, "lyrics_use_raw_text", False)
    lyrics_longformer_model_path = getattr(hp, "lyrics_longformer_model_path", None)
    lyrics_line_token = getattr(hp, "lyrics_line_token", "<LINE>")
    lyrics_longformer_max_length = getattr(hp, "lyrics_longformer_max_length", 2048)
    lyrics_longformer_line_max_length = getattr(hp, "lyrics_longformer_line_max_length", 64)
    lyrics_tokenizer = None
    lyrics_line_token_id = None
    if use_lyrics and lyrics_use_raw_text:
        lyrics_tokenizer = AutoTokenizer.from_pretrained(lyrics_longformer_model_path)
        lyrics_tokenizer.add_special_tokens({"additional_special_tokens": [lyrics_line_token]})
        lyrics_line_token_id = lyrics_tokenizer.convert_tokens_to_ids(lyrics_line_token)

    # --- FIX 2: checkpoint path resolution ---
    # allow absolute checkpoint; otherwise try cwd, repo root, then src/SongFormer/ckpts.
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        candidates = [
            (Path.cwd() / ckpt_path).resolve(),
            (REPO_ROOT / ckpt_path).resolve(),
            (SRC_SONGFORMER / "ckpts" / ckpt_path).resolve(),
        ]
        ckpt_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(checkpoint_path=str(ckpt_path))

    if ckpt.get("model_ema", None) is not None:
        logger.info("Loading EMA model parameters")
        model_ema = EMA(model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        logger.info("No EMA model parameters found, using original model")
        model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()

    num_classes = args.num_classes
    dataset_id2label_mask = {}

    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(args.num_classes, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    with torch.no_grad():
        while True:
            item = queue_input.get()
            if not item:
                queue_output.put(None)
                break

            try:
                # Loading the audio file
                wav, sr = librosa.load(item, sr=INPUT_SAMPLING_RATE)
                audio = torch.tensor(wav).to(device)

                # =========================
                # [ADDED FOR LYRICS]
                # Load per-song line-only lyrics metadata once.
                # Real frame-aligned lyrics sequence is built per chunk.
                # =========================
                lyrics_data = None
                if use_lyrics:
                    if lyrics_use_raw_text:
                        lyrics_data = load_sync_lyrics_json(
                            audio_path=item,
                            lyrics_json_dir=args.lyrics_json_dir,
                        )
                    else:
                        lyrics_data = load_sync_lyrics_npz(
                            audio_path=item,
                            lyrics_embedding_dir=args.lyrics_embedding_dir,
                            lyrics_input_dim=lyrics_input_dim,
                            lyrics_logits_dim=lyrics_logits_dim,
                        )

                win_size = args.win_size
                hop_size = args.hop_size

                total_len = (
                    (audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR
                ) * TIME_DUR + TIME_DUR
                total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

                logits = {
                    "function_logits": np.zeros([total_frames, num_classes]),
                    "boundary_logits": np.zeros([total_frames]),
                }
                logits_num = {
                    "function_logits": np.zeros([total_frames, num_classes]),
                    "boundary_logits": np.zeros([total_frames]),
                }

                lens = 0
                i = 0
                while True:
                    start_idx = i * INPUT_SAMPLING_RATE
                    end_idx = min((i + win_size) * INPUT_SAMPLING_RATE, audio.shape[-1])
                    if start_idx >= audio.shape[-1]:
                        break
                    if end_idx - start_idx <= 1024:
                        i += hop_size
                        continue

                    audio_seg = audio[start_idx:end_idx]

                    # MuQ embedding (420s)
                    muq_output = muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                    muq_embd_420s = muq_output["hidden_states"][10]
                    del muq_output
                    torch.cuda.empty_cache()

                    # MusicFM embedding (420s)
                    _, musicfm_hidden_states = musicfm.get_predictions(
                        audio_seg.unsqueeze(0)
                    )
                    musicfm_embd_420s = musicfm_hidden_states[10]
                    del musicfm_hidden_states
                    torch.cuda.empty_cache()

                    wraped_muq_embd_30s = []
                    wraped_musicfm_embd_30s = []

                    # 30s wrap inside [i, i+hop_size)
                    for idx_30s in range(i, i + hop_size, 30):
                        start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                        end_idx_30s = min(
                            (idx_30s + 30) * INPUT_SAMPLING_RATE,
                            audio.shape[-1],
                            (i + hop_size) * INPUT_SAMPLING_RATE,
                        )
                        if start_idx_30s >= audio.shape[-1]:
                            break
                        if end_idx_30s - start_idx_30s <= 1024:
                            continue

                        wraped_muq_embd_30s.append(
                            muq(
                                audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                                output_hidden_states=True,
                            )["hidden_states"][10]
                        )
                        torch.cuda.empty_cache()

                        wraped_musicfm_embd_30s.append(
                            musicfm.get_predictions(
                                audio[start_idx_30s:end_idx_30s].unsqueeze(0)
                            )[1][10]
                        )
                        torch.cuda.empty_cache()

                    wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                    wraped_musicfm_embd_30s = torch.concatenate(
                        wraped_musicfm_embd_30s, dim=1
                    )

                    all_embds = [
                        wraped_musicfm_embd_30s,
                        wraped_muq_embd_30s,
                        musicfm_embd_420s,
                        muq_embd_420s,
                    ]

                    # align emb lengths
                    if len(all_embds) > 1:
                        embd_lens = [x.shape[1] for x in all_embds]
                        max_embd_len = max(embd_lens)
                        min_embd_len = min(embd_lens)
                        if abs(max_embd_len - min_embd_len) > 4:
                            raise ValueError(
                                f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}"
                            )
                        for idx in range(len(all_embds)):
                            all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                    embd = torch.concatenate(all_embds, axis=-1)

                    # =========================
                    # [ADDED FOR LYRICS V3]
                    # Build line-token lyrics conditions and frame timing features.
                    # =========================
                    lyrics_line_embeddings = None
                    lyrics_line_logits = None
                    lyrics_line_boundary_logits = None
                    lyrics_line_time_features = None
                    lyrics_line_repeat_embeddings = None
                    lyrics_line_repeat_features = None
                    lyrics_line_masks = None
                    lyrics_frame_time_features = None
                    lyrics_frame_line_indices = None
                    lyrics_frame_active_masks = None
                    has_lyrics = None
                    lyrics_input_ids = None
                    lyrics_attention_mask = None
                    lyrics_global_attention_mask = None
                    lyrics_line_positions = None
                    lyrics_line_text_masks = None

                    if use_lyrics:
                        chunk_start_time = float(i)
                        chunk_end_time = min(
                            float(i + win_size),
                            float(audio.shape[-1]) / INPUT_SAMPLING_RATE,
                        )

                        target_len = embd.shape[1] // hp.downsample_rates
                        if lyrics_use_raw_text:
                            (
                                lyrics_line_np,
                                lyrics_line_logits_np,
                                lyrics_line_boundary_logits_np,
                                lyrics_line_time_np,
                                lyrics_line_repeat_np,
                                lyrics_line_repeat_feat_np,
                                lyrics_line_mask_np,
                                lyrics_frame_time_np,
                                lyrics_frame_line_indices_np,
                                lyrics_frame_active_masks_np,
                                has_lyrics_val,
                                lyrics_input_ids_np,
                                lyrics_attention_mask_np,
                                lyrics_global_attention_mask_np,
                                lyrics_line_positions_np,
                                lyrics_line_text_masks_np,
                            ) = build_raw_lyrics_token_condition(
                                lyrics_data=lyrics_data,
                                tokenizer=lyrics_tokenizer,
                                line_token_id=lyrics_line_token_id,
                                target_len=target_len,
                                chunk_start_time=chunk_start_time,
                                chunk_end_time=chunk_end_time,
                                lyrics_input_dim=lyrics_input_dim,
                                lyrics_logits_dim=lyrics_logits_dim,
                                lyrics_time_feat_dim=lyrics_time_feat_dim,
                                lyrics_frame_time_feat_dim=lyrics_frame_time_feat_dim,
                                lyrics_repeat_feat_dim=lyrics_repeat_feat_dim,
                                lyrics_context_window_sec=lyrics_context_window_sec,
                                lyrics_repeat_top_k=lyrics_repeat_top_k,
                                lyrics_repeat_exclude_neighbors=lyrics_repeat_exclude_neighbors,
                                lyrics_repeat_similarity_threshold=lyrics_repeat_similarity_threshold,
                                lyrics_longformer_max_length=lyrics_longformer_max_length,
                                lyrics_longformer_line_max_length=lyrics_longformer_line_max_length,
                            )
                        else:
                            (
                                lyrics_line_np,
                                lyrics_line_logits_np,
                                lyrics_line_boundary_logits_np,
                                lyrics_line_time_np,
                                lyrics_line_repeat_np,
                                lyrics_line_repeat_feat_np,
                                lyrics_line_mask_np,
                                lyrics_frame_time_np,
                                lyrics_frame_line_indices_np,
                                lyrics_frame_active_masks_np,
                                has_lyrics_val,
                            ) = build_lyrics_token_condition(
                                lyrics_data=lyrics_data,
                                target_len=target_len,
                                chunk_start_time=chunk_start_time,
                                chunk_end_time=chunk_end_time,
                                lyrics_input_dim=lyrics_input_dim,
                                lyrics_logits_dim=lyrics_logits_dim,
                                lyrics_time_feat_dim=lyrics_time_feat_dim,
                                lyrics_frame_time_feat_dim=lyrics_frame_time_feat_dim,
                                lyrics_repeat_feat_dim=lyrics_repeat_feat_dim,
                                lyrics_context_window_sec=lyrics_context_window_sec,
                                lyrics_repeat_top_k=lyrics_repeat_top_k,
                                lyrics_repeat_exclude_neighbors=lyrics_repeat_exclude_neighbors,
                                lyrics_repeat_similarity_threshold=lyrics_repeat_similarity_threshold,
                            )

                        lyrics_line_embeddings = (
                            torch.from_numpy(lyrics_line_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_line_logits = (
                            torch.from_numpy(lyrics_line_logits_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_line_boundary_logits = (
                            torch.from_numpy(lyrics_line_boundary_logits_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_line_time_features = (
                            torch.from_numpy(lyrics_line_time_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_line_masks = (
                            torch.from_numpy(lyrics_line_mask_np)
                            .to(device=device, dtype=torch.bool)
                            .unsqueeze(0)
                        )
                        lyrics_line_repeat_embeddings = (
                            torch.from_numpy(lyrics_line_repeat_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_line_repeat_features = (
                            torch.from_numpy(lyrics_line_repeat_feat_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_frame_time_features = (
                            torch.from_numpy(lyrics_frame_time_np)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        lyrics_frame_line_indices = (
                            torch.from_numpy(lyrics_frame_line_indices_np)
                            .to(device=device, dtype=torch.long)
                            .unsqueeze(0)
                        )
                        lyrics_frame_active_masks = (
                            torch.from_numpy(lyrics_frame_active_masks_np)
                            .to(device=device, dtype=torch.bool)
                            .unsqueeze(0)
                        )
                        has_lyrics = torch.tensor(
                            [has_lyrics_val], device=device, dtype=torch.float32
                        )
                        if lyrics_use_raw_text:
                            lyrics_input_ids = (
                                torch.from_numpy(lyrics_input_ids_np)
                                .to(device=device, dtype=torch.long)
                                .unsqueeze(0)
                            )
                            lyrics_attention_mask = (
                                torch.from_numpy(lyrics_attention_mask_np)
                                .to(device=device, dtype=torch.long)
                                .unsqueeze(0)
                            )
                            lyrics_global_attention_mask = (
                                torch.from_numpy(lyrics_global_attention_mask_np)
                                .to(device=device, dtype=torch.long)
                                .unsqueeze(0)
                            )
                            lyrics_line_positions = (
                                torch.from_numpy(lyrics_line_positions_np)
                                .to(device=device, dtype=torch.long)
                                .unsqueeze(0)
                            )
                            lyrics_line_text_masks = (
                                torch.from_numpy(lyrics_line_text_masks_np)
                                .to(device=device, dtype=torch.bool)
                                .unsqueeze(0)
                            )

                    dataset_label = getattr(args, "dataset_label", DATASET_LABEL)
                    dataset_ids = torch.Tensor(getattr(args, "dataset_ids", DATASET_IDS)).to(device, dtype=torch.long)

                    msa_info, chunk_logits = model.infer(
                        input_embeddings=embd,
                        dataset_ids=dataset_ids,
                        label_id_masks=torch.Tensor(
                            dataset_id2label_mask[
                                DATASET_LABEL_TO_DATASET_ID[dataset_label]
                            ]
                        )
                        .to(device, dtype=bool)
                        .unsqueeze(0)
                        .unsqueeze(0),

                        # =========================
                        # [ADDED FOR LYRICS V3]
                        # Line-token lyrics conditions.
                        # =========================
                        lyrics_line_embeddings=lyrics_line_embeddings,
                        lyrics_line_logits=lyrics_line_logits,
                        lyrics_line_time_features=lyrics_line_time_features,
                        lyrics_line_repeat_embeddings=lyrics_line_repeat_embeddings,
                        lyrics_line_repeat_features=lyrics_line_repeat_features,
                        lyrics_line_masks=lyrics_line_masks,
                        lyrics_frame_time_features=lyrics_frame_time_features,
                        lyrics_frame_line_indices=lyrics_frame_line_indices,
                        lyrics_frame_active_masks=lyrics_frame_active_masks,
                        has_lyrics=has_lyrics,
                        lyrics_input_ids=lyrics_input_ids,
                        lyrics_attention_mask=lyrics_attention_mask,
                        lyrics_global_attention_mask=lyrics_global_attention_mask,
                        lyrics_line_positions=lyrics_line_positions,
                        lyrics_line_text_masks=lyrics_line_text_masks,

                        with_logits=True,
                    )

                    start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                    end_frame = start_frame + min(
                        math.ceil(hop_size * AFTER_DOWNSAMPLING_FRAME_RATES),
                        chunk_logits["boundary_logits"][0].shape[0],
                    )

                    logits["function_logits"][start_frame:end_frame, :] += (
                        chunk_logits["function_logits"][0].detach().cpu().numpy()
                    )
                    logits["boundary_logits"][start_frame:end_frame] = (
                        chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    )
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame

                    i += hop_size

                logits["function_logits"] /= logits_num["function_logits"]
                logits["boundary_logits"] /= logits_num["boundary_logits"]

                logits["function_logits"] = torch.from_numpy(
                    logits["function_logits"][:lens]
                ).unsqueeze(0)
                logits["boundary_logits"] = torch.from_numpy(
                    logits["boundary_logits"][:lens]
                ).unsqueeze(0)

                msa_infer_output = postprocess_functional_structure(logits, hp)

                assert msa_infer_output[-1][-1] == "end"
                if not args.no_rule_post_processing:
                    msa_infer_output = rule_post_processing(msa_infer_output)

                msa_json = []
                for idx in range(len(msa_infer_output) - 1):
                    msa_json.append(
                        {
                            "label": msa_infer_output[idx][1],
                            "start": msa_infer_output[idx][0],
                            "end": msa_infer_output[idx + 1][0],
                        }
                    )

                os.makedirs(args.output_dir, exist_ok=True)
                json.dump(
                    msa_json,
                    open(os.path.join(args.output_dir, f"{Path(item).stem}.json"), "w"),
                    indent=4,
                    ensure_ascii=False,
                )

                queue_output.put(None)

            except Exception as e:
                queue_output.put(None)
                logger.error(f"process {rank} error\n{item}\n{e}")


def deal_with_output(output_path, queue_output, length):
    """Handle output data from the queue"""
    pbar = tqdm(range(length), desc="getting inference output")
    for _ in pbar:
        _ = queue_output.get()


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    gpu_num = args.gpu_num
    num_thread_per_gpu = args.num_thread_per_gpu
    debug = args.debug

    os.makedirs(output_path, exist_ok=True)

    processed_ids = get_processed_ids(output_path=output_path)
    processing_ids = get_processing_ids(input_path, processed_ids)

    num_threads = num_thread_per_gpu * gpu_num

    queue_input: mp.Queue = mp.Queue()
    queue_output: mp.Queue = mp.Queue()

    init_args = Namespace(
        output_dir=output_path,
        win_size=420,
        hop_size=420,
        num_classes=128,
        model=args.model,
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        no_rule_post_processing=args.no_rule_post_processing,

        # =========================
        # [ADDED FOR LYRICS]
        # Optional lyrics embedding directory
        # =========================
        lyrics_embedding_dir=args.lyrics_embedding_dir,
        lyrics_json_dir=args.lyrics_json_dir,
    )

    processes = []

    if debug:
        queue_input.put(processing_ids[0])
        queue_input.put(None)

        inference(0, queue_input, queue_output, init_args)

        print("debug exit")
        exit(0)

    for thread_num in range(num_threads):
        rank = thread_num % gpu_num
        print(f"num_threads: {thread_num} on GPU {rank}")
        time.sleep(0.2)
        p = mp.Process(
            target=inference,
            args=(rank, queue_input, queue_output, init_args),
            daemon=True,
        )
        p.start()
        processes.append(p)

    for wav_id in tqdm(processing_ids, desc="add data to queue"):
        queue_input.put(wav_id)

    for _ in range(num_threads):
        queue_input.put(None)

    deal_with_output(output_path, queue_output, len(processing_ids))

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", "-i", type=str, required=True, help="Input scp file (audio paths)"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, required=True, help="Output directory for per-song json"
    )
    parser.add_argument(
        "--gpu_num", "-gn", type=int, default=1, help="Number of GPUs, default is 1"
    )
    parser.add_argument(
        "--num_thread_per_gpu",
        "-tn",
        type=int,
        default=1,
        help="Number of threads per GPU, default is 1",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name under models/")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (.pt). Absolute path OK.")
    parser.add_argument("--config_path", type=str, required=True, help="Config yaml path. Absolute path OK.")
    parser.add_argument("--dataset_label", type=str, default=DATASET_LABEL, help="Dataset label for inference mask.")
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default=",".join(str(x) for x in DATASET_IDS),
        help="Comma- or space-separated dataset ids for inference.",
    )
    parser.add_argument(
        "--no_rule_post_processing",
        action="store_true",
        help="Disable rule-based post-processing",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--lyrics_embedding_dir",
        type=str,
        default="",
        help=(
            "Optional directory containing per-song SoulX-Singer lyrics embeddings. "
            "Expected filename: <audio_stem>.npz with fields "
            "{line_embs, line_start_secs, line_end_secs, global_emb}. "
            "If not found, inference falls back to zero lyrics condition + has_lyrics=0."
        ),
    )
    parser.add_argument(
        "--lyrics_json_dir",
        type=str,
        default="",
        help=(
            "Optional directory containing per-song SoulX-Singer line-level JSON. "
            "Used when config lyrics_use_raw_text=true."
        ),
    )
    args = parser.parse_args()
    args.dataset_ids = [int(x) for x in str(args.dataset_ids).replace(",", " ").split() if x]
    main(args=args)
