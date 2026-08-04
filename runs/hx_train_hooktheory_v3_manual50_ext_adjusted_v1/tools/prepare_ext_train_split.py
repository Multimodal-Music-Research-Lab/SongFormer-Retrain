#!/usr/bin/env python3
"""Validate SongFormDB Ext metadata and write its deterministic train IDs."""

import argparse
import json
import os
from pathlib import Path


DEFAULT_METADATA = Path(
    "/mnt/ssd/hbli/datasets/songformer/songformdb/data/Ext/"
    "SongFormDB-Ext.jsonl"
)
DEFAULT_MEL_DIR = Path(
    "/mnt/ssd/hbli/datasets/songformer/songformdb/Ext"
)
DEFAULT_OUTPUT = DEFAULT_METADATA.with_name("SongFormDB-Ext_train.txt")
EXPECTED_ROWS = 4308
ALLOWED_LABELS = {
    "bridge",
    "chorus",
    "end",
    "ending",
    "inst",
    "interlude",
    "intro",
    "no-vocal-interlude",
    "no-vocal-intro",
    "no-vocal-outro",
    "outro",
    "pre-chorus",
    "silence",
    "verse",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mel_dir", type=Path, default=DEFAULT_MEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected_rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--max_end_duration_delta", type=float, default=1.0)
    return parser.parse_args()


def validate_record(record, line_number, metadata_path, max_end_duration_delta):
    prefix = f"{metadata_path}:{line_number}"
    required = {"id", "split", "subset", "duration", "labels"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{prefix}: missing fields {missing}")
    if record["split"] != "train":
        raise ValueError(f"{prefix}: expected train split, got {record['split']}")
    if record["subset"] != "Ext":
        raise ValueError(f"{prefix}: expected Ext subset, got {record['subset']}")
    if float(record["duration"]) <= 0:
        raise ValueError(f"{prefix}: duration must be positive")

    labels = record["labels"]
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError(f"{prefix}: labels must contain at least two entries")

    normalized = []
    for label_index, item in enumerate(labels):
        if not isinstance(item, dict) or "start" not in item or "label" not in item:
            raise ValueError(
                f"{prefix}: invalid label at index {label_index}: {item}"
            )
        start = float(item["start"])
        label = str(item["label"])
        if label not in ALLOWED_LABELS:
            raise ValueError(f"{prefix}: unsupported label '{label}'")
        normalized.append((start, label))

    if abs(normalized[0][0]) > 1e-6:
        raise ValueError(f"{prefix}: first boundary must start at 0")
    if normalized[-1][1] != "end":
        raise ValueError(f"{prefix}: final label must be end")
    if any(
        left[0] >= right[0]
        for left, right in zip(normalized[:-1], normalized[1:])
    ):
        raise ValueError(f"{prefix}: timestamps must be strictly increasing")
    end_duration_delta = abs(normalized[-1][0] - float(record["duration"]))
    if end_duration_delta > max_end_duration_delta:
        raise ValueError(
            f"{prefix}: end time {normalized[-1][0]} differs from duration "
            f"{record['duration']} by more than {max_end_duration_delta} seconds"
        )
    return end_duration_delta


def main():
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    if not args.mel_dir.is_dir():
        raise FileNotFoundError(args.mel_dir)

    records_by_id = {}
    max_end_duration_delta = 0.0
    with args.metadata.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            end_duration_delta = validate_record(
                record,
                line_number,
                args.metadata,
                args.max_end_duration_delta,
            )
            max_end_duration_delta = max(
                max_end_duration_delta, end_duration_delta
            )
            song_id = str(record["id"])
            if song_id in records_by_id:
                raise ValueError(f"Duplicate Ext id: {song_id}")
            records_by_id[song_id] = record

    if len(records_by_id) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} Ext rows, found {len(records_by_id)}"
        )

    mel_ids = {path.stem for path in args.mel_dir.glob("*.npy")}
    missing_mels = sorted(set(records_by_id) - mel_ids)
    if missing_mels:
        raise ValueError(
            f"{len(missing_mels)} labeled Ext songs have no mel: "
            f"{missing_mels[:10]}"
        )
    unlabeled_mels = sorted(mel_ids - set(records_by_id))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.txt")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for song_id in sorted(records_by_id):
            handle.write(f"{song_id}\n")
    os.replace(temporary, args.output)

    print(f"Validated Ext labels: {len(records_by_id)}")
    print(f"Available Ext mels: {len(mel_ids)}")
    print(f"Unlabeled mels excluded: {len(unlabeled_mels)}")
    print(f"Maximum label-end/duration delta: {max_end_duration_delta:.6f}s")
    if unlabeled_mels:
        print("Excluded IDs: " + ", ".join(unlabeled_mels))
    print(f"Train IDs: {args.output}")


if __name__ == "__main__":
    main()
