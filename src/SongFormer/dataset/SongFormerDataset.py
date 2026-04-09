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
    # [ADDED FOR LYRICS]
    # =========================
    def get_zero_lyrics_sequence(self, target_len: int):
        return np.zeros((target_len, self.lyrics_input_dim), dtype=np.float32)


    def _validate_lyrics_vector_dim(self, lyrics_embedding: np.ndarray, lyrics_path: Path):
        lyrics_embedding = np.asarray(lyrics_embedding, dtype=np.float32).reshape(-1)

        if lyrics_embedding.shape[0] != self.lyrics_input_dim:
            raise ValueError(
                f"Lyrics embedding dim mismatch for {lyrics_path}: "
                f"{lyrics_embedding.shape[0]} vs expected {self.lyrics_input_dim}"
            )
        return lyrics_embedding


    def _validate_lyrics_sequence_dim(self, lyrics_sequence: np.ndarray, lyrics_path: Path):
        lyrics_sequence = np.asarray(lyrics_sequence, dtype=np.float32)

        if lyrics_sequence.ndim != 2:
            raise ValueError(
                f"Lyrics sequence ndim mismatch for {lyrics_path}: "
                f"{lyrics_sequence.ndim} vs expected 2"
            )
        if lyrics_sequence.shape[1] != self.lyrics_input_dim:
            raise ValueError(
                f"Lyrics sequence dim mismatch for {lyrics_path}: "
                f"{lyrics_sequence.shape[1]} vs expected {self.lyrics_input_dim}"
            )
        return lyrics_sequence


    def _repeat_resize_sequence(self, seq: np.ndarray, target_len: int):
        """
        seq: [N, D]
        target_len: target temporal length T

        Repeat-style resizing:
        map each target index t to a source index floor(t * N / T)
        """
        seq = np.asarray(seq, dtype=np.float32)

        if seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got shape={seq.shape}")
        if seq.shape[1] != self.lyrics_input_dim:
            raise ValueError(
                f"Sequence dim mismatch: {seq.shape[1]} vs expected {self.lyrics_input_dim}"
            )
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")

        src_len = seq.shape[0]
        if src_len == 0:
            return self.get_zero_lyrics_sequence(target_len)

        if src_len == target_len:
            return seq.astype(np.float32)

        indices = np.floor(np.arange(target_len) * src_len / target_len).astype(np.int64)
        indices = np.clip(indices, 0, src_len - 1)
        return seq[indices].astype(np.float32)


    def _build_lyrics_sequence_from_npz(self, lyrics_npz, lyrics_path: Path, target_len: int):
        if "line_embs" not in lyrics_npz:
            raise KeyError(f"'line_embs' not found in {lyrics_path}")
        if "stanza_embs" not in lyrics_npz:
            raise KeyError(f"'stanza_embs' not found in {lyrics_path}")

        line_embs = np.asarray(lyrics_npz["line_embs"], dtype=np.float32)
        stanza_embs = np.asarray(lyrics_npz["stanza_embs"], dtype=np.float32)

        if line_embs.ndim != 2 or line_embs.shape[0] == 0:
            raise ValueError(f"Invalid 'line_embs' in {lyrics_path}, got shape={line_embs.shape}")
        if stanza_embs.ndim != 2 or stanza_embs.shape[0] == 0:
            raise ValueError(f"Invalid 'stanza_embs' in {lyrics_path}, got shape={stanza_embs.shape}")

        line_seq = self._repeat_resize_sequence(line_embs, target_len=target_len)       # [T, D]
        stanza_seq = self._repeat_resize_sequence(stanza_embs, target_len=target_len)   # [T, D]

        lyrics_sequence = 0.5 * line_seq + 0.5 * stanza_seq
        return self._validate_lyrics_sequence_dim(lyrics_sequence, lyrics_path)


    # =========================
    # [ADDED FOR LYRICS]
    # song_id should be the song-level stem, e.g. HX_0001_12step
    # Returns:
    #   lyrics_sequence: [T, lyrics_input_dim]
    #   has_lyrics: bool
    # Supports both:
    #   - new .npz structured embedding: line/stanza -> repeat resize -> [T, D]
    #   - old .npy global embedding: repeat same vector to [T, D]
    # =========================
    def try_load_lyrics_sequence(self, internal_tmp_id: str, song_id: str, target_len: int):
        if not self.use_lyrics:
            return None, False

        lyrics_dir = self.lyrics_embedding_dir.get(internal_tmp_id, None)
        if lyrics_dir is None or str(lyrics_dir).strip() == "":
            return self.get_zero_lyrics_sequence(target_len), False

        lyrics_dir = Path(lyrics_dir)
        lyrics_npz_path = lyrics_dir / f"{song_id}.npz"
        lyrics_npy_path = lyrics_dir / f"{song_id}.npy"

        # 1) Prefer new structured npz
        if lyrics_npz_path.exists():
            lyrics_npz = np.load(lyrics_npz_path, allow_pickle=False)
            lyrics_sequence = self._build_lyrics_sequence_from_npz(
                lyrics_npz=lyrics_npz,
                lyrics_path=lyrics_npz_path,
                target_len=target_len,
            )
            return lyrics_sequence.astype(np.float32), True

        # 2) Fallback to old single-vector npy
        if lyrics_npy_path.exists():
            lyrics_embedding = np.load(lyrics_npy_path, allow_pickle=False)
            lyrics_embedding = self._validate_lyrics_vector_dim(
                lyrics_embedding=lyrics_embedding,
                lyrics_path=lyrics_npy_path,
            )
            lyrics_sequence = np.repeat(lyrics_embedding[None, :], target_len, axis=0)
            return lyrics_sequence.astype(np.float32), True

        return self.get_zero_lyrics_sequence(target_len), False

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

                target_len = item_json["mask"].shape[0]

                lyrics_sequence, has_lyrics = self.try_load_lyrics_sequence(
                    internal_tmp_id=internal_tmp_id,
                    song_id=song_id,
                    target_len=target_len,
                )

                item_json["lyrics_embedding"] = lyrics_sequence   # [T, D]
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

            # downsampled temporal length T used by the model
            target_len = input_embedding.shape[0] // self.downsample_rates

            # =========================
            # [ADDED FOR LYRICS]
            # utt is now song-level id, e.g. HX_0001_12step
            # =========================
            lyrics_embedding, has_lyrics = self.try_load_lyrics_sequence(
                internal_tmp_id=internal_tmp_id,
                song_id=utt,
                target_len=target_len,
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
                "lyrics_embedding": lyrics_embedding,
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
            # lyrics is sequence-level, so shape is [B, T, D]
            # For samples without lyrics, keep zero sequence and has_lyrics = 0
            # =========================
            if self.use_lyrics:
                lyrics_embeddings = np.zeros(
                    (len(batch), max_sequence_length, self.lyrics_input_dim), dtype=np.float32
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
                # Hook / Private / missing-HX samples will naturally fall back to zero sequence
                # =========================
                if self.use_lyrics:
                    item_lyrics = item.get("lyrics_embedding", None)
                    if item_lyrics is not None:
                        seq_len = min(item_lyrics.shape[0], max_sequence_length)
                        lyrics_embeddings[idx, :seq_len] = item_lyrics[:seq_len]
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
                lyrics_embeddings = torch.from_numpy(lyrics_embeddings).float()
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
                return_json["lyrics_embeddings"] = lyrics_embeddings
                return_json["has_lyrics"] = has_lyrics

            return return_json
        except Exception as e:
            logger.error(f"Error occurred while processing dataset: {e}")
            return None