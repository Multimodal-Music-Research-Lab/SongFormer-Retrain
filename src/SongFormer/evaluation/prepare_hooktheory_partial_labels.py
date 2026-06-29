#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


SINGLE_MAPPING = {
    "bridge": "bridge",
    "chorus": "chorus",
    "instrumental": "inst",
    "intro": "intro",
    "lead-in": "pre-chorus",
    "loop": "inst",
    "outro": "outro",
    "pre-chorus": "pre-chorus",
    "pre-outro": "outro",
    "silence": "silence",
    "solo": "inst",
    "verse": "verse",
    "end": "end",
}

COMPOSITE_MAPPING = {
    "intro and verse": ["intro", "verse"],
    "intro and chorus": ["intro", "chorus"],
    "verse and pre-chorus": ["verse", "pre-chorus"],
    "pre-chorus and chorus": ["pre-chorus", "chorus"],
    "chorus lead-out": ["chorus", "outro"],
    "lead-in alt": ["pre-chorus"],
}


def read_scp_stems(scp_path: Path) -> list[str]:
    stems = []
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace("\ufeff", "").replace("\r", "")
            if not s or s.startswith("#"):
                continue
            stems.append(Path(s.split()[-1]).stem)
    return stems


def normalize_raw(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("_", "-")
    s = " ".join(s.split())
    s = s.replace("pre chorus", "pre-chorus")
    parts = s.split()
    if parts and parts[-1].isdigit():
        s = " ".join(parts[:-1])
    return s


def choose_label(labels: list[str], policy: str) -> str:
    if not labels:
        raise ValueError("Empty label list")
    if policy == "first":
        return labels[0]
    if policy == "last":
        return labels[-1]
    if policy == "prefer_chorus":
        for label in ["chorus", "verse", "pre-chorus", "bridge", "outro", "intro", "inst", "silence"]:
            if label in labels:
                return label
    raise ValueError(f"Unknown composite policy: {policy}")


def apply_prechorus(label: str, prechorus2what: str | None) -> str:
    if label != "pre-chorus" or prechorus2what in (None, "", "none"):
        return label
    if prechorus2what not in {"verse", "chorus"}:
        raise ValueError(f"Unknown prechorus2what: {prechorus2what}")
    return prechorus2what


def normalize_label(raw: str, composite_policy: str, prechorus2what: str | None) -> str:
    s = normalize_raw(raw)
    if s in COMPOSITE_MAPPING:
        label = choose_label(COMPOSITE_MAPPING[s], composite_policy)
    elif " and " in s:
        label = choose_label([normalize_label(x, composite_policy, None) for x in s.split(" and ")], composite_policy)
    elif s in SINGLE_MAPPING:
        label = SINGLE_MAPPING[s]
    else:
        raise ValueError(f"Unknown HookTheory label: {raw!r} -> {s!r}")
    return apply_prechorus(label, prechorus2what)


def normalize_file(
    src_path: Path,
    dst_path: Path,
    composite_policy: str,
    prechorus2what: str | None,
) -> None:
    rows: list[tuple[float, str]] = []
    with src_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                raise ValueError(f"Missing label in {src_path} line {line_no}: {line!r}")
            t = float(parts[0])
            label = normalize_label(parts[1], composite_policy, prechorus2what)
            rows.append((t, label))

    if not rows:
        raise ValueError(f"No labels found in {src_path}")
    if rows[-1][1] != "end":
        raise ValueError(f"{src_path} does not end with end")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        for t, label in rows:
            f.write(f"{t:.6f} {label}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalize HookTheory partial-label txt files into SongFormer single-label MSA txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ann_dir", type=str, required=True, help="Raw HookTheory label txt directory.")
    p.add_argument("--scp", type=str, required=True, help="SCP whose stems select the labels to normalize.")
    p.add_argument("--output_dir", type=str, required=True, help="Output normalized label txt directory.")
    p.add_argument("--prechorus2what", type=str, default="verse", choices=["verse", "chorus", "none"])
    p.add_argument("--composite_policy", type=str, default="first", choices=["first", "last", "prefer_chorus"])
    p.add_argument("--allow_failed", action="store_true", help="Skip labels that cannot be normalized.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.output_dir)
    stems = read_scp_stems(Path(args.scp))
    prechorus2what = None if args.prechorus2what == "none" else args.prechorus2what

    missing = []
    failed = []
    written = 0
    for stem in stems:
        src = ann_dir / f"{stem}.txt"
        dst = out_dir / f"{stem}.txt"
        if not src.exists():
            missing.append(stem)
            continue
        try:
            normalize_file(src, dst, args.composite_policy, prechorus2what)
            written += 1
        except Exception as exc:
            failed.append((stem, str(exc)))

    print("=== HookTheory partial label normalization done ===")
    print(f"ann_dir: {ann_dir}")
    print(f"scp: {args.scp}")
    print(f"output_dir: {out_dir}")
    print(f"requested: {len(stems)}")
    print(f"written: {written}")
    print(f"missing: {len(missing)}")
    print(f"failed: {len(failed)}")
    if missing:
        print(f"missing samples: {missing[:10]}")
    if failed:
        print(f"failed samples: {failed[:10]}")
        if not args.allow_failed:
            raise RuntimeError(f"Failed to normalize {len(failed)} files")


if __name__ == "__main__":
    main()
