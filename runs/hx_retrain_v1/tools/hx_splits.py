from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create train/val/test ID lists from meta json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--meta",
        type=str,
        default=os.environ.get("HX_META", ""),
        help="Path to SongFormDB-HX.jsonl (or any meta jsonl that includes id & split)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.environ.get("HX_SPLIT_DIR", ""),
        help="Output directory for train.txt/val.txt/test.txt",
    )
    p.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="JSON key name for sample ID",
    )
    p.add_argument(
        "--split_key",
        type=str,
        default="split",
        help="JSON key name for split label",
    )
    p.add_argument(
        "--split_map",
        type=str,
        default="train:train,val:val,valid:val,validation:val,test:test",
        help=(
            "Mapping for raw split values -> standardized split name. "
            "Format: raw1:std1,raw2:std2,...  (values are compared lowercased)"
        ),
    )
    p.add_argument(
        "--write_test",
        action="store_true",
        help="Also write test.txt if there are any items mapped to 'test'",
    )
    return p.parse_args()


def parse_split_map(s: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad --split_map item: {item!r} (expected raw:std)")
        raw, std = item.split(":", 1)
        m[raw.strip().lower()] = std.strip().lower()
    return m


def main() -> None:
    args = parse_args()

    if not args.meta:
        raise SystemExit("ERROR: --meta is required (or set env HX_META).")
    if not args.out_dir:
        raise SystemExit("ERROR: --out_dir is required (or set env HX_SPLIT_DIR).")

    meta_path = Path(args.meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = parse_split_map(args.split_map)

    buckets = {"train": [], "val": [], "test": []}
    unknown_split_counts: Dict[str, int] = {}
    missing_key_counts = {"id": 0, "split": 0}

    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"ERROR: JSON decode failed at line {line_no}: {e}") from e

            if args.id_key not in r:
                missing_key_counts["id"] += 1
                continue
            if args.split_key not in r:
                missing_key_counts["split"] += 1
                continue

            sid = str(r[args.id_key]).strip()
            raw_split = str(r[args.split_key]).strip().lower()

            std_split = split_map.get(raw_split, "")
            if std_split in buckets:
                buckets[std_split].append(sid)
            else:
                unknown_split_counts[raw_split] = unknown_split_counts.get(raw_split, 0) + 1

    # De-dup & sort for determinism
    for k in buckets:
        buckets[k] = sorted(set(buckets[k]))

    # Write outputs
    (out_dir / "train.txt").write_text("\n".join(buckets["train"]) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(buckets["val"]) + "\n", encoding="utf-8")
    if args.write_test and buckets["test"]:
        (out_dir / "test.txt").write_text("\n".join(buckets["test"]) + "\n", encoding="utf-8")

    print("=== Split generation done ===")
    print(f"meta:    {meta_path}")
    print(f"out_dir: {out_dir}")
    print(f"train:   {len(buckets['train'])}")
    print(f"val:     {len(buckets['val'])}")
    print(f"test:    {len(buckets['test'])}  (written={args.write_test and bool(buckets['test'])})")
    if unknown_split_counts:
        print("WARNING: Unknown split values (not mapped by --split_map):")
        for k, v in sorted(unknown_split_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k!r}: {v}")
    if any(missing_key_counts.values()):
        print("WARNING: Some rows missing required keys:")
        print(f"  missing id_key({args.id_key}): {missing_key_counts['id']}")
        print(f"  missing split_key({args.split_key}): {missing_key_counts['split']}")


if __name__ == "__main__":
    main()