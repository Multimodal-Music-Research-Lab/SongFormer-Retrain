#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TXT_DIR = Path("/mnt/ssd/hbli/madmom/outputs/songformerdb/HarmonixSet_adjusted_restriction/")


def parse_txt_labels(txt_path: Path) -> List[List[float | str]]:
    """
    Parse a txt file into labels: [[time, label], ...]
    Ensures:
      - label lowercased
      - times are float
      - sorted by time
      - if no final 'end', append [duration, 'end'] later in update step
    """
    labels: List[Tuple[float, str]] = []
    for ln, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            raise ValueError(f"Bad line @ {txt_path} line {ln}: {line!r}")
        t = float(parts[0])
        lab = " ".join(parts[1:]).strip().lower()
        labels.append((t, lab))

    if not labels:
        raise ValueError(f"Empty labels: {txt_path}")

    # sort by time, and drop exact duplicate (time,label) pairs while preserving order
    labels.sort(key=lambda x: (x[0], x[1]))
    dedup: List[Tuple[float, str]] = []
    seen = set()
    for t, lab in labels:
        key = (t, lab)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((t, lab))

    return [[float(t), lab] for t, lab in dedup]


def load_base_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise ValueError(f"Bad json @ {path} line {ln}: {e}")
            if "id" not in obj:
                raise ValueError(f"Missing 'id' @ {path} line {ln}")
            items.append(obj)
    return items


def write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for obj in items:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_jsonl", type=str, required=True, help="Original model jsonl path")
    ap.add_argument("--out_jsonl", type=str, required=True, help="Output jsonl path")
    ap.add_argument(
        "--txt_dir",
        type=str,
        default=str(DEFAULT_TXT_DIR),
        help="Directory containing adjusted-window *.txt files",
    )
    ap.add_argument(
        "--txt_ext",
        type=str,
        default=".txt",
        help="Label txt extension (default: .txt)",
    )
    args = ap.parse_args()

    base_jsonl = Path(args.base_jsonl)
    out_jsonl = Path(args.out_jsonl)
    txt_dir = Path(args.txt_dir)

    if not base_jsonl.exists():
        raise FileNotFoundError(f"base_jsonl not found: {base_jsonl}")
    if not txt_dir.exists():
        raise FileNotFoundError(f"txt_dir not found: {txt_dir}")

    # 1) load all txt labels
    id2labels: Dict[str, List[List[float | str]]] = {}
    txt_files = sorted([p for p in txt_dir.rglob("*.txt") if not p.name.endswith("_db.txt")])
    for p in txt_files:
        song_id = p.stem  # assume filename stem == obj["id"]
        try:
            id2labels[song_id] = parse_txt_labels(p)
        except Exception as e:
            # fail fast (better than silently skipping)
            raise RuntimeError(f"Failed parsing {p}: {e}")

    # 2) load base jsonl and update labels
    items = load_base_jsonl(base_jsonl)

    hit = 0
    miss = 0
    fixed_end = 0

    for obj in items:
        sid = obj["id"]
        if sid not in id2labels:
            miss += 1
            continue

        new_labels = id2labels[sid]

        # Ensure last label is 'end'
        last_lab = str(new_labels[-1][1]).lower()
        if last_lab != "end":
            # append end at duration (keep duration unchanged)
            if "duration" not in obj:
                raise ValueError(f"Missing duration in base jsonl for id={sid}, cannot append end")
            new_labels.append([float(obj["duration"]), "end"])
            fixed_end += 1

        # Replace ONLY labels field
        obj["labels"] = new_labels
        hit += 1

    # 3) write output
    write_jsonl(out_jsonl, items)

    print("=== Done ===")
    print(f"base_jsonl: {base_jsonl}")
    print(f"txt_dir:    {txt_dir} (txt files: {len(txt_files)})")
    print(f"out_jsonl:  {out_jsonl}")
    print(f"updated:    {hit}")
    print(f"no_txt:     {miss}")
    print(f"append_end: {fixed_end}")


if __name__ == "__main__":
    main()
