#!/usr/bin/env python3
"""Audit the fully annotated HookTheory pool and create a deterministic split."""

import argparse
import random
from pathlib import Path


ALLOWED_LABELS = {
    "bridge",
    "chorus",
    "inst",
    "intro",
    "outro",
    "pre-chorus",
    "silence",
    "verse",
}
AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
DEFAULT_LABEL_DIR = Path(
    "/mnt/ssd/hbli/datasets/hooktheory/labels_mannual_adjusted"
)
DEFAULT_TEST_SCP = Path(
    "/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_full/results/"
    "hooktheory_test_full_unique_636.scp"
)
DEFAULT_V3_TRAIN = Path(
    "/home/hbli/songformer/repo/SongFormer/runs/"
    "hx_train_hooktheory_v3/results/train.txt"
)
DEFAULT_AUDIO_ROOT = Path(
    "/mnt/ssd/datasets/SheetSageAudio/full/打包结果_audio"
)
DEFAULT_SSL_DIRS = [
    Path(
        "/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/"
        "ssl/musicfm/30s/layer_10"
    ),
    Path(
        "/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/"
        "ssl/muq/30s/layer_10"
    ),
    Path(
        "/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/"
        "ssl/musicfm/420s/layer_10"
    ),
    Path(
        "/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/"
        "ssl/muq/420s/layer_10"
    ),
]


def parse_args():
    run_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--test_scp", type=Path, default=DEFAULT_TEST_SCP)
    parser.add_argument("--v3_train_ids", type=Path, default=DEFAULT_V3_TRAIN)
    parser.add_argument("--audio_root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument(
        "--ssl_dirs",
        type=Path,
        nargs="+",
        default=DEFAULT_SSL_DIRS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train", type=int, default=50)
    parser.add_argument("--expected_candidates", type=int, default=100)
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalized_label_dir", type=Path)
    args = parser.parse_args()
    if args.output is None:
        suffix = "train50" if args.split == "train" else "heldout50"
        args.output = run_dir / "results" / (
            f"hooktheory_manual_full_{suffix}_seed{args.seed}.scp"
        )
    return args


def read_nonempty_lines(path):
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_id(value):
    path = Path(value)
    return path.stem if path.suffix.lower() in AUDIO_SUFFIXES else value


def read_source_scp(path):
    source_by_id = {}
    for value in read_nonempty_lines(path):
        song_id = normalize_id(value)
        if song_id in source_by_id:
            raise ValueError(f"Duplicate song id in {path}: {song_id}")
        source_by_id[song_id] = Path(value)
    return source_by_id


def validate_label_file(path):
    annotations = []
    for line_number, line in enumerate(read_nonempty_lines(path), start=1):
        fields = line.split(maxsplit=1)
        time_text = fields[0]
        if len(fields) == 1:
            if not annotations:
                raise ValueError(
                    f"{path}:{line_number}: first boundary has no label"
                )
            label = annotations[-1][1]
            print(
                f"WARNING: {path}:{line_number}: missing label; "
                f"inherit '{label}'"
            )
        else:
            label = fields[1]
        if " and " in label:
            original_label = label
            label = label.split(" and ", maxsplit=1)[0]
            print(
                f"WARNING: {path}:{line_number}: composite label "
                f"'{original_label}'; use first label '{label}'"
            )
        annotations.append((float(time_text), label))

    if len(annotations) < 2:
        raise ValueError(f"{path}: fewer than two annotations")
    if annotations[-1][1] != "end":
        raise ValueError(f"{path}: final annotation must be 'end'")
    if any(
        left[0] >= right[0]
        for left, right in zip(annotations[:-1], annotations[1:])
    ):
        raise ValueError(f"{path}: timestamps are not strictly increasing")

    unknown = sorted(
        {label for _, label in annotations[:-1] if label not in ALLOWED_LABELS}
    )
    if unknown:
        raise ValueError(f"{path}: unsupported labels: {', '.join(unknown)}")
    return annotations


def main():
    args = parse_args()
    label_paths = sorted(args.label_dir.glob("*.txt"))
    if len(label_paths) != args.expected_candidates:
        raise ValueError(
            f"Expected {args.expected_candidates} labels, found {len(label_paths)}"
        )

    source_by_id = read_source_scp(args.test_scp)
    original_train_ids = {
        normalize_id(value) for value in read_nonempty_lines(args.v3_train_ids)
    }
    candidate_ids = [path.stem for path in label_paths]

    missing_from_test = sorted(set(candidate_ids) - set(source_by_id))
    overlap_with_train = sorted(set(candidate_ids) & original_train_ids)
    if missing_from_test:
        raise ValueError(
            f"{len(missing_from_test)} annotations are absent from test SCP: "
            f"{missing_from_test[:5]}"
        )
    if overlap_with_train:
        raise ValueError(
            f"{len(overlap_with_train)} candidates overlap v3 train: "
            f"{overlap_with_train[:5]}"
        )

    failures = []
    normalized_annotations = {}
    for label_path in label_paths:
        song_id = label_path.stem
        normalized_annotations[song_id] = validate_label_file(label_path)
        audio_path = source_by_id[song_id]
        if not audio_path.is_file():
            failures.append(f"{song_id}: missing audio {audio_path}")
        elif audio_path.parent.resolve() != args.audio_root.resolve():
            failures.append(f"{song_id}: audio outside canonical root {audio_path}")
        for ssl_dir in args.ssl_dirs:
            if not any(ssl_dir.glob(f"{song_id}_*.npy")):
                failures.append(f"{song_id}: no SSL chunk in {ssl_dir}")
    if failures:
        raise ValueError("\n".join(failures))

    rng = random.Random(args.seed)
    train_ids = set(rng.sample(sorted(candidate_ids), args.num_train))
    selected_ids = (
        sorted(train_ids)
        if args.split == "train"
        else sorted(set(candidate_ids) - train_ids)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for song_id in selected_ids:
            f.write(f"{source_by_id[song_id]}\n")

    if args.normalized_label_dir is not None:
        args.normalized_label_dir.mkdir(parents=True, exist_ok=True)
        for song_id in selected_ids:
            output_path = args.normalized_label_dir / f"{song_id}.txt"
            with output_path.open("w", encoding="utf-8", newline="\n") as f:
                for time, label in normalized_annotations[song_id]:
                    f.write(f"{time} {label}\n")

    print(f"Audited candidates: {len(candidate_ids)}")
    print(f"Original-v3 overlap: {len(overlap_with_train)}")
    print(f"Selected split: {args.split} ({len(selected_ids)} songs)")
    print(f"Output: {args.output}")
    if args.normalized_label_dir is not None:
        print(f"Normalized labels: {args.normalized_label_dir}")


if __name__ == "__main__":
    main()
