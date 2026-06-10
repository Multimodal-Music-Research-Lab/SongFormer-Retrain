# For this dataset, ablation studies become easier
import copy
import json
import pdb
from argparse import Namespace
from pathlib import Path
import traceback
import numpy as np
import torch
from dataset.custom_types import MsaInfo
from dataset.label2id import (
    DATASET_ID_ALLOWED_LABEL_IDS,
    DATASET_LABEL_TO_DATASET_ID,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from loguru import logger
from scipy.ndimage import maximum_filter1d
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import os
import random
from .HookTheoryAdapter import HookTheoryAdapter
from .HookTheoryV1Adapter import HookTheoryV1Adapter
from .GeminiOnlyLabelAdapter import GeminiOnlyLabelAdapter


class Dataset(Dataset):
    def get_ids_from_dir(self, dir_path: str):
        ids = os.listdir(dir_path)
        ids = [Path(x).stem for x in ids if x.endswith(".npy")]
        return set(ids)
    # =========================
    # [ADDED FOR LYRICS V3]
    # Line-token lyrics units with timestamp-derived features.
    # The model learns boundary/function deltas from line content, line timing,
    # and frame-level lyric timing cues. No ground-truth segment boundaries are
    # used as model inputs.
    # =========================
    def get_zero_lyrics_tokens(self, target_len: int):
        zero_line = np.zeros((1, self.lyrics_input_dim), dtype=np.float32)
        zero_logits = np.zeros((1, self.lyrics_logits_dim), dtype=np.float32)
        zero_boundary_logits = np.zeros((1,), dtype=np.float32)
        zero_time = np.zeros((1, self.lyrics_time_feat_dim), dtype=np.float32)
        zero_repeat_line = np.zeros((1, self.lyrics_input_dim), dtype=np.float32)
        zero_repeat_feat = np.zeros((1, self.lyrics_repeat_feat_dim), dtype=np.float32)
        zero_start = np.zeros((1,), dtype=np.float32)
        zero_end = np.zeros((1,), dtype=np.float32)
        zero_frame = np.zeros((target_len, self.lyrics_frame_time_feat_dim), dtype=np.float32)
        zero_frame_line_indices = np.full((target_len,), -1, dtype=np.int64)
        zero_frame_active_mask = np.zeros((target_len,), dtype=bool)
        return (
            zero_line,
            zero_logits,
            zero_boundary_logits,
            zero_time,
            zero_repeat_line,
            zero_repeat_feat,
            zero_start,
            zero_end,
            zero_frame,
            zero_frame_line_indices,
            zero_frame_active_mask,
        )

    def _validate_lyrics_units_dim(self, arr: np.ndarray, expected_dim: int, field_name: str, lyrics_path: Path):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{field_name} ndim mismatch in {lyrics_path}: got {arr.ndim}, expected 2")
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"{field_name} dim mismatch in {lyrics_path}: "
                f"{arr.shape[1]} vs expected {expected_dim}"
            )
        return arr

    def _validate_time_vector_dim(self, arr: np.ndarray, field_name: str, lyrics_path: Path):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"{field_name} ndim mismatch in {lyrics_path}: got {arr.ndim}, expected 1")
        return arr

    def _build_line_time_features(
        self,
        start_secs: np.ndarray,
        end_secs: np.ndarray,
        chunk_start_time: float,
        chunk_end_time: float,
    ) -> np.ndarray:
        start_secs = np.asarray(start_secs, dtype=np.float32).reshape(-1)
        end_secs = np.asarray(end_secs, dtype=np.float32).reshape(-1)

        chunk_start_time = float(chunk_start_time)
        chunk_end_time = float(chunk_end_time)
        chunk_width = max(chunk_end_time - chunk_start_time, 1e-6)
        chunk_center = 0.5 * (chunk_start_time + chunk_end_time)

        centers = 0.5 * (start_secs + end_secs)
        durations = np.maximum(end_secs - start_secs, 1e-6)
        overlap = np.maximum(
            0.0,
            np.minimum(end_secs, chunk_end_time) - np.maximum(start_secs, chunk_start_time),
        )
        overlap_ratio = overlap / durations
        inside_flag = ((centers >= chunk_start_time) & (centers <= chunk_end_time)).astype(np.float32)

        prev_gap = np.zeros_like(start_secs, dtype=np.float32)
        next_gap = np.zeros_like(start_secs, dtype=np.float32)
        if len(start_secs) > 1:
            prev_gap[1:] = np.maximum(start_secs[1:] - end_secs[:-1], 0.0)
            next_gap[:-1] = np.maximum(start_secs[1:] - end_secs[:-1], 0.0)

        feats = np.stack(
            [
                (start_secs - chunk_start_time) / chunk_width,
                (end_secs - chunk_start_time) / chunk_width,
                (centers - chunk_center) / chunk_width,
                durations / chunk_width,
                overlap_ratio,
                inside_flag,
                prev_gap / chunk_width,
                next_gap / chunk_width,
            ],
            axis=1,
        ).astype(np.float32)

        if feats.shape[1] != self.lyrics_time_feat_dim:
            raise ValueError(
                f"lyrics_time_feat_dim mismatch: built {feats.shape[1]}, "
                f"but expected {self.lyrics_time_feat_dim}"
            )
        return feats

    def _build_line_repeat_features(
        self,
        line_embs: np.ndarray,
        line_start_secs: np.ndarray,
    ):
        line_embs = np.asarray(line_embs, dtype=np.float32)
        line_start_secs = np.asarray(line_start_secs, dtype=np.float32).reshape(-1)
        line_count = len(line_embs)
        repeat_embs = np.zeros_like(line_embs, dtype=np.float32)
        repeat_feats = np.zeros((line_count, self.lyrics_repeat_feat_dim), dtype=np.float32)
        if line_count <= 1:
            return repeat_embs, repeat_feats

        norm = np.linalg.norm(line_embs, axis=1, keepdims=True)
        normalized = line_embs / np.maximum(norm, 1e-8)
        similarities = normalized @ normalized.T
        song_width = max(float(line_start_secs.max() - line_start_secs.min()), 1e-6)

        for line_idx in range(line_count):
            candidate = similarities[line_idx].copy()
            local_left = max(0, line_idx - self.lyrics_repeat_exclude_neighbors)
            local_right = min(line_count, line_idx + self.lyrics_repeat_exclude_neighbors + 1)
            candidate[local_left:local_right] = -np.inf
            valid = np.isfinite(candidate)
            if not valid.any():
                continue
            top_k = min(self.lyrics_repeat_top_k, int(valid.sum()))
            top_indices = np.argpartition(candidate, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(candidate[top_indices])[::-1]]
            top_scores = candidate[top_indices]
            positive_scores = np.maximum(top_scores, 0.0)
            score_sum = float(positive_scores.sum())
            if score_sum > 1e-8:
                repeat_embs[line_idx] = (
                    line_embs[top_indices] * positive_scores[:, None]
                ).sum(axis=0) / score_sum
            best_idx = int(top_indices[0])
            repeat_feats[line_idx] = np.asarray(
                [
                    float(top_scores[0]),
                    float(top_scores.mean()),
                    float((candidate[valid] >= self.lyrics_repeat_similarity_threshold).sum())
                    / max(float(valid.sum()), 1.0),
                    abs(float(line_start_secs[best_idx] - line_start_secs[line_idx])) / song_width,
                ],
                dtype=np.float32,
            )
        return repeat_embs, repeat_feats

    def _build_frame_alignment(
        self,
        line_start_secs: np.ndarray,
        line_end_secs: np.ndarray,
        target_len: int,
        chunk_start_time: float,
        chunk_end_time: float,
    ):
        target_len = int(target_len)
        chunk_start_time = float(chunk_start_time)
        chunk_end_time = float(chunk_end_time)
        chunk_width = max(chunk_end_time - chunk_start_time, 1e-6)
        frame_dur = chunk_width / max(float(target_len), 1.0)
        frame_centers = chunk_start_time + (np.arange(target_len, dtype=np.float32) + 0.5) * frame_dur
        feats = np.zeros((target_len, self.lyrics_frame_time_feat_dim), dtype=np.float32)
        frame_line_indices = np.full((target_len,), -1, dtype=np.int64)
        frame_active_mask = np.zeros((target_len,), dtype=bool)

        line_start_secs = np.asarray(line_start_secs, dtype=np.float32).reshape(-1)
        line_end_secs = np.asarray(line_end_secs, dtype=np.float32).reshape(-1)
        if len(line_start_secs) == 0:
            return feats, frame_line_indices, frame_active_mask

        line_centers = 0.5 * (line_start_secs + line_end_secs)
        for idx, frame_time in enumerate(frame_centers):
            start_dist = np.min(np.abs(line_start_secs - frame_time)) / chunk_width
            end_dist = np.min(np.abs(line_end_secs - frame_time)) / chunk_width
            center_idx = int(np.argmin(np.abs(line_centers - frame_time)))
            signed_center_dist = (frame_time - line_centers[center_idx]) / chunk_width
            covering = np.flatnonzero((line_start_secs <= frame_time) & (frame_time <= line_end_secs))
            inside = len(covering) > 0
            if inside:
                nearest_idx = int(covering[np.argmin(np.abs(line_centers[covering] - frame_time))])
                frame_line_indices[idx] = nearest_idx
                frame_active_mask[idx] = True
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
            density_window = 5.0
            density = np.mean(np.abs(line_centers - frame_time) <= density_window)
            feats[idx] = np.asarray(
                [
                    relative_position,
                    start_pulse,
                    end_pulse,
                    float(inside),
                    float(density),
                    duration_ratio,
                ],
                dtype=np.float32,
            )

        if feats.shape[1] != self.lyrics_frame_time_feat_dim:
            raise ValueError(
                f"lyrics_frame_time_feat_dim mismatch: built {feats.shape[1]}, "
                f"but expected {self.lyrics_frame_time_feat_dim}"
            )
        return feats, frame_line_indices, frame_active_mask

    def try_load_lyrics_tokens(
        self,
        internal_tmp_id: str,
        song_id: str,
        target_len: int,
        chunk_start_time: float,
        chunk_end_time: float,
    ):
        if not self.use_lyrics:
            return None, None, None, None, None, None, None, None, None, None, None, False

        zero_values = self.get_zero_lyrics_tokens(target_len)
        lyrics_dir = self.lyrics_embedding_dir.get(internal_tmp_id, None)
        if lyrics_dir is None or str(lyrics_dir).strip() == "":
            return (*zero_values, False)

        lyrics_npz_path = Path(lyrics_dir) / f"{song_id}.npz"
        if not lyrics_npz_path.exists():
            return (*zero_values, False)

        try:
            lyrics_npz = np.load(lyrics_npz_path, allow_pickle=False)
            required_fields = ["line_embs", "line_start_secs", "line_end_secs"]
            for field in required_fields:
                if field not in lyrics_npz:
                    raise KeyError(f"'{field}' not found in {lyrics_npz_path}")

            line_embs = self._validate_lyrics_units_dim(
                lyrics_npz["line_embs"], self.lyrics_input_dim, "line_embs", lyrics_npz_path
            )
            line_start_secs = self._validate_time_vector_dim(
                lyrics_npz["line_start_secs"], "line_start_secs", lyrics_npz_path
            )
            line_end_secs = self._validate_time_vector_dim(
                lyrics_npz["line_end_secs"], "line_end_secs", lyrics_npz_path
            )
            if "line_logits" in lyrics_npz:
                line_logits = self._validate_lyrics_units_dim(
                    lyrics_npz["line_logits"], self.lyrics_logits_dim, "line_logits", lyrics_npz_path
                )
            else:
                line_logits = np.zeros((len(line_embs), self.lyrics_logits_dim), dtype=np.float32)
            if "line_boundary_logits" in lyrics_npz:
                line_boundary_logits = self._validate_time_vector_dim(
                    lyrics_npz["line_boundary_logits"], "line_boundary_logits", lyrics_npz_path
                )
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

            context_window = float(self.lyrics_context_window_sec)
            keep = (
                (line_end_secs > float(chunk_start_time) - context_window)
                & (line_start_secs < float(chunk_end_time) + context_window)
            )
            if keep.sum() <= 0:
                return (*zero_values, False)

            repeat_embs, repeat_features = self._build_line_repeat_features(
                line_embs=line_embs,
                line_start_secs=line_start_secs,
            )

            local_line_embs = line_embs[keep].astype(np.float32)
            local_line_logits = line_logits[keep].astype(np.float32)
            local_line_boundary_logits = line_boundary_logits[keep].astype(np.float32)
            local_repeat_embs = repeat_embs[keep].astype(np.float32)
            local_repeat_features = repeat_features[keep].astype(np.float32)
            local_start_secs = line_start_secs[keep].astype(np.float32)
            local_end_secs = line_end_secs[keep].astype(np.float32)
            local_time_features = self._build_line_time_features(
                start_secs=local_start_secs,
                end_secs=local_end_secs,
                chunk_start_time=chunk_start_time,
                chunk_end_time=chunk_end_time,
            )
            frame_time_features, frame_line_indices, frame_active_mask = self._build_frame_alignment(
                line_start_secs=local_start_secs,
                line_end_secs=local_end_secs,
                target_len=target_len,
                chunk_start_time=chunk_start_time,
                chunk_end_time=chunk_end_time,
            )
            return (
                local_line_embs,
                local_line_logits,
                local_line_boundary_logits,
                local_time_features,
                local_repeat_embs,
                local_repeat_features,
                local_start_secs,
                local_end_secs,
                frame_time_features,
                frame_line_indices,
                frame_active_mask,
                True,
            )
        except Exception as e:
            logger.warning(f"Failed to load lyrics npz {lyrics_npz_path}: {e}")
            return (*zero_values, False)

    def __init__(
        self,
        dataset_abstracts: dict,
        hparams,
    ):
        # initialize storage and hyperparams
        self.time_datas = {}
        self.label_datas = {}
        self.hparams = hparams
        self.label_to_id = LABEL_TO_ID
        self.dataset_id_to_dataset_id = DATASET_LABEL_TO_DATASET_ID
        self.id_to_label = ID_TO_LABEL
        self.dataset_id2label_mask = {}
        self.output_logits_frame_rates = self.hparams.output_logits_frame_rates
        self.downsample_rates = self.hparams.downsample_rates
        self.valid_data_ids = []
        self.SLICE_DUR = self.hparams.slice_dur

        self.input_embedding_dir = {}
        self.EPS = 1e-6
        # =========================
        # [ADDED FOR LYRICS]
        # =========================
        self.use_lyrics = getattr(self.hparams, "use_lyrics", False)
        self.lyrics_input_dim = getattr(self.hparams, "lyrics_input_dim", 1024)
        self.lyrics_logits_dim = getattr(self.hparams, "lyrics_logits_dim", 8)
        self.lyrics_time_feat_dim = getattr(self.hparams, "lyrics_time_feat_dim", 8)
        self.lyrics_frame_time_feat_dim = getattr(self.hparams, "lyrics_frame_time_feat_dim", 6)
        self.lyrics_repeat_feat_dim = getattr(self.hparams, "lyrics_repeat_feat_dim", 4)
        self.lyrics_context_window_sec = getattr(self.hparams, "lyrics_context_window_sec", 30.0)
        self.lyrics_repeat_top_k = getattr(self.hparams, "lyrics_repeat_top_k", 3)
        self.lyrics_repeat_exclude_neighbors = getattr(self.hparams, "lyrics_repeat_exclude_neighbors", 2)
        self.lyrics_repeat_similarity_threshold = getattr(self.hparams, "lyrics_repeat_similarity_threshold", 0.8)
        self.lyrics_embedding_dir = {}

        # build dataset-specific label mask
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            self.dataset_id2label_mask[key] = np.ones(
                self.hparams.num_classes, dtype=bool
            )
            self.dataset_id2label_mask[key][allowed_ids] = False

        uniq_id_nums = 0
        self.adapter_obj = {}

        for dataset_abstract_item in dataset_abstracts:
            adapter = dataset_abstract_item.get("adapter", None)
            if adapter is not None:
                self.lyrics_embedding_dir[dataset_abstract_item["internal_tmp_id"]] = dataset_abstract_item.get("lyrics_embedding_dir", None)
                # adapter-based dataset (pre-wrapped)
                assert isinstance(adapter, str)
                if adapter == "HookTheoryAdapter":
                    self.adapter_obj[dataset_abstract_item["internal_tmp_id"]] = (
                        HookTheoryAdapter(**dataset_abstract_item, hparams=self.hparams)
                    )
                    valid_data_ids = self.adapter_obj[
                        dataset_abstract_item["internal_tmp_id"]
                    ].get_ids()
                elif adapter == "GeminiOnlyLabelAdapter":
                    self.adapter_obj[dataset_abstract_item["internal_tmp_id"]] = (
                        GeminiOnlyLabelAdapter(
                            **dataset_abstract_item, hparams=self.hparams
                        )
                    )
                    valid_data_ids = self.adapter_obj[
                        dataset_abstract_item["internal_tmp_id"]
                    ].get_ids()
                elif adapter == "HookTheoryV1Adapter":
                    self.adapter_obj[dataset_abstract_item["internal_tmp_id"]] = (
                        HookTheoryV1Adapter(**dataset_abstract_item, hparams=self.hparams)
                    )
                    valid_data_ids = self.adapter_obj[
                        dataset_abstract_item["internal_tmp_id"]
                    ].get_ids()
                else:
                    raise ValueError(f"Unknown adapter: {adapter}")

                logger.info(
                    f"{dataset_abstract_item['internal_tmp_id']}: {len(valid_data_ids)} * {dataset_abstract_item['multiplier']}"
                )
                uniq_id_nums += len(valid_data_ids)
                for i in range(dataset_abstract_item["multiplier"]):
                    self.valid_data_ids.extend(valid_data_ids)

            else:
                # raw dataset definition
                internal_tmp_id = dataset_abstract_item["internal_tmp_id"]
                dataset_type = dataset_abstract_item["dataset_type"]
                all_input_embedding_dirs = dataset_abstract_item[
                    "input_embedding_dir"
                ].split()
                label_path = dataset_abstract_item["label_path"]
                split_ids_path = dataset_abstract_item["split_ids_path"]

                self.input_embedding_dir[internal_tmp_id] = dataset_abstract_item[
                    "input_embedding_dir"
                ]

                # =========================
                # [ADDED FOR LYRICS]
                # Only HX will provide a valid lyrics dir for now.
                # Private / Hook can leave it empty or absent.
                # =========================
                self.lyrics_embedding_dir[internal_tmp_id] = dataset_abstract_item.get(
                    "lyrics_embedding_dir", None
                )
                
                valid_data_ids = self.get_ids_from_dir(all_input_embedding_dirs[0])
                for x in all_input_embedding_dirs:
                    valid_data_ids = valid_data_ids.intersection(
                        self.get_ids_from_dir(x)
                    )

                split_ids = []
                with open(split_ids_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        split_ids.append(line.strip())
                split_ids = set(split_ids)

                # filter valid ids by split membership
                valid_data_ids = [
                    x
                    for x in valid_data_ids
                    if "_".join(x.split("_")[:-1]) in split_ids
                ]

                valid_data_ids = [
                    (internal_tmp_id, dataset_type, x, None) for x in valid_data_ids
                ]

                assert isinstance(dataset_abstract_item["multiplier"], int)
                uniq_id_nums += len(valid_data_ids)
                logger.info(
                    f"{internal_tmp_id}: {len(valid_data_ids)} * {dataset_abstract_item['multiplier']}"
                )
                for i in range(dataset_abstract_item["multiplier"]):
                    self.valid_data_ids.extend(valid_data_ids)
                self.init_segments(
                    label_path=label_path, internal_tmp_id=internal_tmp_id
                )

        logger.info(f"{uniq_id_nums} valid data ids, {len(self.valid_data_ids)} total")
        rng = np.random.default_rng(42)
        rng.shuffle(self.valid_data_ids)

    def init_segments(
        self,
        label_path,
        internal_tmp_id,
    ):
        # load segment times and labels from label jsonl
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if not line:
                    continue
                line_data = json.loads(line)
                hybrid_id = internal_tmp_id + "_" + line_data["id"]
                self.time_datas[hybrid_id] = [x[0] for x in line_data["labels"]]
                self.time_datas[hybrid_id] = list(
                    map(float, self.time_datas[hybrid_id])
                )
                self.label_datas[hybrid_id] = [
                    -1 if x[1] == "end" else self.label_to_id[x[1]]
                    for x in line_data["labels"]
                ]

    def __len__(self):
        return len(self.valid_data_ids)

    def widen_temporal_events(self, events, num_neighbors):
        # smooth discrete events with normalized Gaussian kernel
        def theoretical_gaussian_max(sigma):
            return 1 / (np.sqrt(2 * np.pi) * sigma)

        widen_events = events
        sigma = num_neighbors / 3.0
        smoothed = gaussian_filter1d(widen_events.astype(float), sigma=sigma)
        smoothed /= theoretical_gaussian_max(sigma)
        smoothed = np.clip(smoothed, 0, 1)

        return smoothed

    def time2frame(self, this_time):
        assert this_time <= self.SLICE_DUR
        return int(this_time * self.output_logits_frame_rates)

    def __getitem__(self, idx):
        try:
            internal_tmp_id, dataset_label, utt, adapter_str = self.valid_data_ids[idx]
            if adapter_str is not None:
                # handle adapter-wrapped entries
                assert isinstance(adapter_str, str)

                start_time = int(utt.split("_")[-1])

                if adapter_str == "HookTheoryAdapter":
                    item_json = self.adapter_obj[internal_tmp_id].get_item_json(
                        utt=utt,
                        start_time=start_time,
                        end_time=start_time + self.SLICE_DUR,
                    )
                elif adapter_str == "HookTheoryV1Adapter":
                    item_json = self.adapter_obj[internal_tmp_id].get_item_json(
                        utt=utt,
                        start_time=start_time,
                        end_time=start_time + self.SLICE_DUR,
                    )
                elif adapter_str == "GeminiOnlyLabelAdapter":
                    item_json = self.adapter_obj[internal_tmp_id].get_item_json(
                        utt=utt,
                        start_time=start_time,
                        end_time=start_time + self.SLICE_DUR,
                    )
                else:
                    raise ValueError(f"Unknown adapter: {adapter_str}")

                if item_json is None:
                    return None

                # =========================
                # [ADDED FOR LYRICS]
                # For adapter-based datasets (e.g. HookTheoryV1Adapter),
                # the song-level id is the chunk stem without the trailing start_time.
                # Example:
                #   utt = "070-shake_guilty-conscience_kygz-KaPmKB_0"
                #   song_id = "070-shake_guilty-conscience_kygz-KaPmKB"
                # =========================
                song_id = "_".join(utt.split("_")[:-1])
                target_len = item_json["input_embedding"].shape[0] // self.downsample_rates

                (
                    lyrics_line_embeddings,
                    lyrics_line_logits,
                    lyrics_line_boundary_logits,
                    lyrics_line_time_features,
                    lyrics_line_repeat_embeddings,
                    lyrics_line_repeat_features,
                    lyrics_line_start_secs,
                    lyrics_line_end_secs,
                    lyrics_frame_time_features,
                    lyrics_frame_line_indices,
                    lyrics_frame_active_masks,
                    has_lyrics,
                ) = self.try_load_lyrics_tokens(
                    internal_tmp_id=internal_tmp_id,
                    song_id=song_id,
                    target_len=target_len,
                    chunk_start_time=float(start_time),
                    chunk_end_time=float(start_time + self.SLICE_DUR),
                )

                item_json["lyrics_line_embeddings"] = lyrics_line_embeddings
                item_json["lyrics_line_logits"] = lyrics_line_logits
                item_json["lyrics_line_boundary_logits"] = lyrics_line_boundary_logits
                item_json["lyrics_line_time_features"] = lyrics_line_time_features
                item_json["lyrics_line_repeat_embeddings"] = lyrics_line_repeat_embeddings
                item_json["lyrics_line_repeat_features"] = lyrics_line_repeat_features
                item_json["lyrics_line_start_secs"] = lyrics_line_start_secs
                item_json["lyrics_line_end_secs"] = lyrics_line_end_secs
                item_json["lyrics_frame_time_features"] = lyrics_frame_time_features
                item_json["lyrics_frame_line_indices"] = lyrics_frame_line_indices
                item_json["lyrics_frame_active_masks"] = lyrics_frame_active_masks
                item_json["has_lyrics"] = has_lyrics

                return item_json

            # load embeddings from configured dirs
            embd_list = []
            embd_dirs = self.input_embedding_dir[internal_tmp_id].split()
            for embd_dir in embd_dirs:
                if not Path(embd_dir).exists():
                    raise FileNotFoundError(
                        f"Embedding directory {embd_dir} does not exist"
                    )
                tmp = np.load(Path(embd_dir) / f"{utt}.npy").squeeze(axis=0)
                embd_list.append(tmp)

            # check that max/min length difference across embeddings <= 4
            if len(embd_list) > 1:
                embd_shapes = [x.shape for x in embd_list]
                max_shape = max(embd_shapes, key=lambda x: x[0])
                min_shape = min(embd_shapes, key=lambda x: x[0])
                if abs(max_shape[0] - min_shape[0]) > 4:
                    raise ValueError(
                        f"Embedding shapes differ too much: {max_shape} vs {min_shape}"
                    )
            if len(embd_list) > 1:
                for idx in range(len(embd_list)):
                    embd_list[idx] = embd_list[idx][: min_shape[0], :]

            input_embedding = np.concatenate(embd_list, axis=-1)

            start_time = int(utt.split("_")[-1])
            utt_id_with_start_sec = utt
            utt = "_".join(utt.split("_")[:-1])
            end_time = start_time + self.SLICE_DUR

            # =========================
            # [ADDED FOR LYRICS]
            # utt is now song-level id, e.g. HX_0001_12step
            # =========================
            target_len = input_embedding.shape[0] // self.downsample_rates

            (
                lyrics_line_embeddings,
                lyrics_line_logits,
                lyrics_line_boundary_logits,
                lyrics_line_time_features,
                lyrics_line_repeat_embeddings,
                lyrics_line_repeat_features,
                lyrics_line_start_secs,
                lyrics_line_end_secs,
                lyrics_frame_time_features,
                lyrics_frame_line_indices,
                lyrics_frame_active_masks,
                has_lyrics,
            ) = self.try_load_lyrics_tokens(
                internal_tmp_id=internal_tmp_id,
                song_id=utt,
                target_len=target_len,
                chunk_start_time=float(start_time),
                chunk_end_time=float(end_time),
            )

            local_times = np.array(
                copy.deepcopy(self.time_datas[f"{internal_tmp_id}_{utt}"])
            )
            local_labels = copy.deepcopy(self.label_datas[f"{internal_tmp_id}_{utt}"])

            assert np.all(local_times[:-1] < local_times[1:]), (
                f"time must be sorted, but {utt} is {local_times}"
            )

            local_times = local_times - start_time

            time_L = max(0.0, float(local_times.min()))
            time_R = min(float(self.SLICE_DUR), float(local_times.max()))

            keep_boundarys = (time_L + self.EPS < local_times) & (
                local_times < time_R - self.EPS
            )

            # If no valid boundaries, return None (skip)
            if keep_boundarys.sum() <= 0:
                return None

            mask = np.ones(
                [int(self.SLICE_DUR * self.output_logits_frame_rates)], dtype=bool
            )
            mask[self.time2frame(time_L) : self.time2frame(time_R)] = False

            true_boundary = np.zeros(
                [int(self.SLICE_DUR * self.output_logits_frame_rates)], dtype=float
            )
            for idx in np.flatnonzero(keep_boundarys):
                true_boundary[self.time2frame(local_times[idx])] = 1

            true_function = np.zeros(
                [
                    int(self.SLICE_DUR * self.output_logits_frame_rates),
                    self.hparams.num_classes,
                ],
                dtype=float,
            )
            true_function_list = []
            msa_info: MsaInfo = []
            last_pos = self.time2frame(time_L)
            for idx in np.flatnonzero(keep_boundarys):
                true_function[
                    last_pos : self.time2frame(local_times[idx]),
                    int(local_labels[idx - 1]),
                ] = 1
                true_function_list.append(int(local_labels[idx - 1]))
                last_pos = self.time2frame(local_times[idx])
                msa_info.append(
                    (
                        float(max(local_times[idx - 1], time_L)),
                        str(self.id_to_label[int(local_labels[idx - 1])]),
                    )
                )

            true_function[
                last_pos : self.time2frame(time_R),
                local_labels[int(np.flatnonzero(keep_boundarys)[-1])],
            ] = 1
            true_function_list.append(
                int(local_labels[int(np.flatnonzero(keep_boundarys)[-1])])
            )
            msa_info.append(
                (
                    float(local_times[int(np.flatnonzero(keep_boundarys)[-1])]),
                    str(
                        self.id_to_label[
                            int(local_labels[int(np.flatnonzero(keep_boundarys)[-1])])
                        ]
                    ),
                )
            )
            msa_info.append((float(time_R), "end"))

            return {
                "data_id": internal_tmp_id + "_" + utt_id_with_start_sec,
                "input_embedding": input_embedding,
                "mask": mask,
                "true_boundary": true_boundary,
                "widen_true_boundary": self.widen_temporal_events(
                    true_boundary, num_neighbors=self.hparams.num_neighbors
                ),
                "true_function": true_function,
                "true_function_list": true_function_list,
                "msa_info": msa_info,
                "dataset_id": self.dataset_id_to_dataset_id[dataset_label],
                "label_id_mask": self.dataset_id2label_mask[
                    self.dataset_id_to_dataset_id[dataset_label]
                ],
                # =========================
                # [ADDED FOR LYRICS]
                # =========================
                "lyrics_line_embeddings": lyrics_line_embeddings,
                "lyrics_line_logits": lyrics_line_logits,
                "lyrics_line_boundary_logits": lyrics_line_boundary_logits,
                "lyrics_line_time_features": lyrics_line_time_features,
                "lyrics_line_repeat_embeddings": lyrics_line_repeat_embeddings,
                "lyrics_line_repeat_features": lyrics_line_repeat_features,
                "lyrics_line_start_secs": lyrics_line_start_secs,
                "lyrics_line_end_secs": lyrics_line_end_secs,
                "lyrics_frame_time_features": lyrics_frame_time_features,
                "lyrics_frame_line_indices": lyrics_frame_line_indices,
                "lyrics_frame_active_masks": lyrics_frame_active_masks,
                "has_lyrics": has_lyrics,
            }
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"error in __getitem__, idx={idx}, utt={utt}, error is:\n{e}\n{tb_str}"
            )
            return None

    def collate_fn(self, batch):
        """
        Return dictionary including:
        - data_ids
        - input_embeddings
        - masks
        - true_boundaries
        - widen_true_boundaries
        - true_functions
        - true_function_lists
        """
        try:
            # filter out None entries
            batch = [x for x in batch if x is not None]
            if len(batch) == 0:
                return None

            data_ids = []
            max_embeddings_length = max([x["input_embedding"].shape[0] for x in batch])
            max_sequence_length = max_embeddings_length // self.downsample_rates

            # allocate numpy arrays for batch
            input_embeddings = np.zeros(
                (len(batch), max_embeddings_length, self.hparams.input_dim), dtype=float
            )
            masks = np.ones((len(batch), max_sequence_length), dtype=bool)
            true_boundaries = np.zeros((len(batch), max_sequence_length), dtype=float)
            widen_true_boundaries = np.zeros(
                (len(batch), max_sequence_length), dtype=float
            )
            true_functions = np.zeros(
                (len(batch), max_sequence_length, self.hparams.num_classes), dtype=float
            )
            boundary_mask = np.zeros((len(batch), max_sequence_length), dtype=bool)
            function_mask = np.zeros((len(batch), max_sequence_length), dtype=bool)
            # =========================
            # [ADDED FOR LYRICS]
            # =========================
            if self.use_lyrics:
                max_line_len = max([x["lyrics_line_embeddings"].shape[0] for x in batch])
                lyrics_line_embeddings = np.zeros(
                    (len(batch), max_line_len, self.lyrics_input_dim), dtype=np.float32
                )
                lyrics_line_logits = np.zeros(
                    (len(batch), max_line_len, self.lyrics_logits_dim), dtype=np.float32
                )
                lyrics_line_boundary_logits = np.zeros(
                    (len(batch), max_line_len), dtype=np.float32
                )
                lyrics_line_time_features = np.zeros(
                    (len(batch), max_line_len, self.lyrics_time_feat_dim), dtype=np.float32
                )
                lyrics_line_repeat_embeddings = np.zeros(
                    (len(batch), max_line_len, self.lyrics_input_dim), dtype=np.float32
                )
                lyrics_line_repeat_features = np.zeros(
                    (len(batch), max_line_len, self.lyrics_repeat_feat_dim), dtype=np.float32
                )
                lyrics_line_start_secs = np.zeros((len(batch), max_line_len), dtype=np.float32)
                lyrics_line_end_secs = np.zeros((len(batch), max_line_len), dtype=np.float32)
                lyrics_line_masks = np.ones((len(batch), max_line_len), dtype=bool)
                lyrics_frame_time_features = np.zeros(
                    (len(batch), max_sequence_length, self.lyrics_frame_time_feat_dim), dtype=np.float32
                )
                lyrics_frame_line_indices = np.full(
                    (len(batch), max_sequence_length), -1, dtype=np.int64
                )
                lyrics_frame_active_masks = np.zeros(
                    (len(batch), max_sequence_length), dtype=bool
                )
                has_lyrics = np.zeros((len(batch),), dtype=np.float32)
            true_function_lists = []
            msa_infos = []
            dataset_ids = []
            label_id_masks = []

            for idx, item in enumerate(batch):
                data_ids.append(item["data_id"])
                input_embeddings[idx, : item["input_embedding"].shape[0]] = item[
                    "input_embedding"
                ]
                masks[idx, : item["mask"].shape[0]] = item["mask"][:max_sequence_length]
                true_boundaries[idx, : item["true_boundary"].shape[0]] = item[
                    "true_boundary"
                ][:max_sequence_length]
                widen_true_boundaries[idx, : item["widen_true_boundary"].shape[0]] = (
                    item["widen_true_boundary"]
                )[:max_sequence_length]
                true_functions[idx, : item["true_function"].shape[0]] = item[
                    "true_function"
                ][:max_sequence_length]
                true_function_lists.append(item["true_function_list"])
                msa_infos.append(item["msa_info"])
                dataset_ids.append(item["dataset_id"])
                label_id_masks.append(item["label_id_mask"])
                # =========================
                # [ADDED FOR LYRICS]
                # =========================
                if self.use_lyrics:
                    item_line_embs = item["lyrics_line_embeddings"]
                    item_line_logits = item["lyrics_line_logits"]
                    item_line_boundary_logits = item["lyrics_line_boundary_logits"]
                    item_line_time = item["lyrics_line_time_features"]
                    item_line_repeat_embs = item["lyrics_line_repeat_embeddings"]
                    item_line_repeat_feats = item["lyrics_line_repeat_features"]
                    item_line_start = item["lyrics_line_start_secs"]
                    item_line_end = item["lyrics_line_end_secs"]
                    item_frame_time = item["lyrics_frame_time_features"]
                    item_frame_line_indices = item["lyrics_frame_line_indices"]
                    item_frame_active_masks = item["lyrics_frame_active_masks"]

                    line_len = item_line_embs.shape[0]
                    seq_len = min(item_frame_time.shape[0], max_sequence_length)

                    lyrics_line_embeddings[idx, :line_len] = item_line_embs
                    lyrics_line_logits[idx, :line_len] = item_line_logits
                    lyrics_line_boundary_logits[idx, :line_len] = item_line_boundary_logits
                    lyrics_line_time_features[idx, :line_len] = item_line_time
                    lyrics_line_repeat_embeddings[idx, :line_len] = item_line_repeat_embs
                    lyrics_line_repeat_features[idx, :line_len] = item_line_repeat_feats
                    lyrics_line_start_secs[idx, :line_len] = item_line_start
                    lyrics_line_end_secs[idx, :line_len] = item_line_end
                    lyrics_line_masks[idx, :line_len] = False
                    lyrics_frame_time_features[idx, :seq_len] = item_frame_time[:seq_len]
                    lyrics_frame_line_indices[idx, :seq_len] = item_frame_line_indices[:seq_len]
                    lyrics_frame_active_masks[idx, :seq_len] = item_frame_active_masks[:seq_len]
                    has_lyrics[idx] = float(item.get("has_lyrics", False))
                if boundary_mask is not None:
                    boundary_mask[idx, : item["mask"].shape[0]] = item.get(
                        "boundary_mask", np.zeros(item["mask"].shape[0], dtype=bool)
                    )[:max_sequence_length]
                if function_mask is not None:
                    function_mask[idx, : item["mask"].shape[0]] = item.get(
                        "function_mask", np.zeros(item["mask"].shape[0], dtype=bool)
                    )[:max_sequence_length]

            # convert to torch tensors
            input_embeddings = torch.from_numpy(input_embeddings).float()
            masks = torch.from_numpy(masks).bool()
            true_boundaries = torch.from_numpy(true_boundaries).float()
            widen_true_boundaries = torch.from_numpy(widen_true_boundaries).float()
            true_functions = torch.from_numpy(true_functions).float()
            boundary_mask = torch.from_numpy(boundary_mask).bool()
            function_mask = torch.from_numpy(function_mask).bool()
            true_function_lists = [
                torch.tensor(x, dtype=torch.long) for x in true_function_lists
            ]
            dataset_ids = torch.from_numpy(np.array(dataset_ids, dtype=np.int64))

            label_id_masks = torch.from_numpy(
                np.stack(label_id_masks, axis=0, dtype=bool)[:, np.newaxis, :]
            )
            # =========================
            # [ADDED FOR LYRICS]
            # =========================
            if self.use_lyrics:
                lyrics_line_embeddings = torch.from_numpy(lyrics_line_embeddings).float()
                lyrics_line_logits = torch.from_numpy(lyrics_line_logits).float()
                lyrics_line_boundary_logits = torch.from_numpy(lyrics_line_boundary_logits).float()
                lyrics_line_time_features = torch.from_numpy(lyrics_line_time_features).float()
                lyrics_line_repeat_embeddings = torch.from_numpy(lyrics_line_repeat_embeddings).float()
                lyrics_line_repeat_features = torch.from_numpy(lyrics_line_repeat_features).float()
                lyrics_line_start_secs = torch.from_numpy(lyrics_line_start_secs).float()
                lyrics_line_end_secs = torch.from_numpy(lyrics_line_end_secs).float()
                lyrics_line_masks = torch.from_numpy(lyrics_line_masks).bool()
                lyrics_frame_time_features = torch.from_numpy(lyrics_frame_time_features).float()
                lyrics_frame_line_indices = torch.from_numpy(lyrics_frame_line_indices).long()
                lyrics_frame_active_masks = torch.from_numpy(lyrics_frame_active_masks).bool()
                has_lyrics = torch.from_numpy(has_lyrics).float()

            return_json = {
                "data_ids": data_ids,
                "input_embeddings": input_embeddings,
                "masks": masks,
                "true_boundaries": true_boundaries,
                "widen_true_boundaries": widen_true_boundaries,
                "true_functions": true_functions,
                "true_function_lists": true_function_lists,
                "msa_infos": msa_infos,
                "dataset_ids": dataset_ids,
                "label_id_masks": label_id_masks,
                "boundary_mask": boundary_mask,
                "function_mask": function_mask,
            }

            # =========================
            # [ADDED FOR LYRICS]
            # =========================
            if self.use_lyrics:
                return_json["lyrics_line_embeddings"] = lyrics_line_embeddings
                return_json["lyrics_line_logits"] = lyrics_line_logits
                return_json["lyrics_line_boundary_logits"] = lyrics_line_boundary_logits
                return_json["lyrics_line_time_features"] = lyrics_line_time_features
                return_json["lyrics_line_repeat_embeddings"] = lyrics_line_repeat_embeddings
                return_json["lyrics_line_repeat_features"] = lyrics_line_repeat_features
                return_json["lyrics_line_start_secs"] = lyrics_line_start_secs
                return_json["lyrics_line_end_secs"] = lyrics_line_end_secs
                return_json["lyrics_line_masks"] = lyrics_line_masks
                return_json["lyrics_frame_time_features"] = lyrics_frame_time_features
                return_json["lyrics_frame_line_indices"] = lyrics_frame_line_indices
                return_json["lyrics_frame_active_masks"] = lyrics_frame_active_masks
                return_json["has_lyrics"] = has_lyrics

            return return_json
        except Exception as e:
            logger.error(f"Error occurred while processing dataset: {e}")
            return None
