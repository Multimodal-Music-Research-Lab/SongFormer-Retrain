#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set


def song_key_from_stem(stem: str) -> str:
    parts = [x for x in stem.split("_") if x]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse HookTheory stem: {stem}")
    return f"{parts[0]}_{parts[1]}"


def read_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace("\ufeff", "").replace("\r", "")
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def collect_audio_variants(audio_dir: Path, audio_ext: str) -> Dict[str, List[Path]]:
    by_key: Dict[str, List[Path]] = defaultdict(list)
    for audio_path in audio_dir.rglob(f"*{audio_ext}"):
        key = song_key_from_stem(audio_path.stem)
        by_key[key].append(audio_path.resolve())
    for variants in by_key.values():
        variants.sort(key=lambda p: p.name)
    return dict(sorted(by_key.items()))


def collect_measure_sec_counts(section_dir: Path, sec_ext: str) -> Dict[str, int]:
    suffix = f"_measure{sec_ext}"
    counts: Dict[str, int] = defaultdict(int)
    for sec_path in section_dir.rglob(f"*{suffix}"):
        stem = sec_path.name[: -len(suffix)]
        counts[song_key_from_stem(stem)] += 1
    return dict(counts)


def ssl_base_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return "_".join(parts[:-1]) if parts and parts[-1].isdigit() else stem


def collect_common_ssl_bases(ssl_dirs: List[str]) -> Set[str]:
    common: Set[str] | None = None
    for ssl_dir in ssl_dirs:
        p = Path(ssl_dir)
        bases = {ssl_base_from_stem(x.stem) for x in p.glob("*.npy")}
        common = bases if common is None else common.intersection(bases)
    return common if common is not None else set()


def choose_representative_audio(
    key: str,
    variants: List[Path],
    current_train_id_set: Set[str],
    ssl_bases: Set[str] | None,
) -> Path:
    for audio_path in variants:
        if audio_path.stem in current_train_id_set:
            return audio_path
    if ssl_bases:
        for audio_path in variants:
            if audio_path.stem in ssl_bases:
                return audio_path
    return variants[0]


def write_scp(path: Path, audio_paths: Iterable[Path]) -> None:
    audio_paths = list(audio_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{str(p)}\n" for p in audio_paths),
        encoding="utf-8",
    )


def write_ids(path: Path, audio_paths: Iterable[Path]) -> None:
    audio_paths = list(audio_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{p.stem}\n" for p in audio_paths),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build HookTheory train/test scp files from audio_cut. "
            "Train keeps the current v3 exact ids; test is every remaining "
            "deduplicated <artist>_<title> song."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio_dir", type=str, required=True, help="Directory containing HookTheory cut mp3 files")
    p.add_argument("--section_dir", type=str, required=True, help="Directory containing HookTheory *_measure.sec files")
    p.add_argument("--current_train_ids", type=str, required=True, help="Existing v3 train.txt/train_hook.txt")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--audio_ext", type=str, default=".mp3")
    p.add_argument("--sec_ext", type=str, default=".sec")
    p.add_argument("--prefix", type=str, default="hooktheory")
    p.add_argument(
        "--ssl_dirs",
        nargs="*",
        default=None,
        help="Optional SSL npy dirs. Representatives are chosen from stems available in every dir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    section_dir = Path(args.section_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_key = collect_audio_variants(audio_dir, args.audio_ext)
    sec_counts = collect_measure_sec_counts(section_dir, args.sec_ext)
    common_ssl_bases = collect_common_ssl_bases(args.ssl_dirs or []) if args.ssl_dirs else None

    current_train_ids = read_ids(Path(args.current_train_ids))
    current_train_id_set: Set[str] = set(current_train_ids)
    if len(current_train_id_set) != len(current_train_ids):
        raise ValueError("current_train_ids contains duplicate exact ids")

    train_key_to_stem = {song_key_from_stem(stem): stem for stem in current_train_ids}
    train_keys = set(train_key_to_stem)
    if len(train_keys) != len(current_train_ids):
        raise ValueError("current_train_ids contains duplicate <artist>_<title> keys")

    audio_stem_to_path = {audio_path.stem: audio_path for variants in by_key.values() for audio_path in variants}
    missing_train_ids = sorted(current_train_id_set - set(audio_stem_to_path))
    if missing_train_ids:
        raise FileNotFoundError(
            f"{len(missing_train_ids)} current train ids are missing from audio_dir. "
            f"First examples: {missing_train_ids[:10]}"
        )

    all_keys = set(by_key)
    test_keys = sorted(all_keys - train_keys)
    representative_by_key = {
        key: choose_representative_audio(key, variants, current_train_id_set, common_ssl_bases)
        for key, variants in by_key.items()
    }
    train_paths = [representative_by_key[key] for key in sorted(train_keys)]
    test_paths = [representative_by_key[key] for key in test_keys]
    all_paths = train_paths + test_paths

    train_scp = out_dir / f"{args.prefix}_train_unique_9010.scp"
    test_scp = out_dir / f"{args.prefix}_test_unique_636.scp"
    all_scp = out_dir / f"{args.prefix}_all_unique_9646.scp"
    train_ids_out = out_dir / "train_hook_full.txt"
    test_ids_out = out_dir / "test_hook_full.txt"
    manifest_csv = out_dir / f"{args.prefix}_full_split_manifest.csv"

    write_scp(train_scp, train_paths)
    write_scp(test_scp, test_paths)
    write_scp(all_scp, all_paths)
    write_ids(train_ids_out, train_paths)
    write_ids(test_ids_out, test_paths)

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "song_key",
                "split",
                "representative_stem",
                "representative_audio_path",
                "num_audio_variants",
                "num_measure_sec",
            ],
        )
        writer.writeheader()
        for key in sorted(all_keys):
            split = "train" if key in train_keys else "test"
            rep_path = representative_by_key[key]
            writer.writerow(
                {
                    "song_key": key,
                    "split": split,
                    "representative_stem": rep_path.stem,
                    "representative_audio_path": str(rep_path),
                    "num_audio_variants": len(by_key[key]),
                    "num_measure_sec": sec_counts.get(key, 0),
                }
            )

    no_measure_keys = sorted(key for key in all_keys if sec_counts.get(key, 0) == 0)
    missing_ssl_rep_keys = []
    if common_ssl_bases is not None:
        missing_ssl_rep_keys = sorted(key for key, p in representative_by_key.items() if p.stem not in common_ssl_bases)

    print("=== HookTheory full split generation done ===")
    print(f"audio_dir:      {audio_dir}")
    print(f"section_dir:    {section_dir}")
    print(f"out_dir:        {out_dir}")
    print(f"all songs:      {len(all_keys)}")
    print(f"train songs:    {len(train_paths)}")
    print(f"test songs:     {len(test_paths)}")
    print(f"audio variants: {sum(len(v) for v in by_key.values())}")
    print(f"measure secs:   {sum(sec_counts.values())}")
    print(f"keys without measure sec: {len(no_measure_keys)}")
    if common_ssl_bases is not None:
        print(f"common SSL exact bases: {len(common_ssl_bases)}")
        print(f"representatives missing common SSL: {len(missing_ssl_rep_keys)}")
        if missing_ssl_rep_keys:
            print(f"first missing SSL representatives: {missing_ssl_rep_keys[:10]}")
    if no_measure_keys:
        print(f"first missing measure keys: {no_measure_keys[:10]}")
    print(f"train_scp:      {train_scp}")
    print(f"test_scp:       {test_scp}")
    print(f"all_scp:        {all_scp}")
    print(f"train_ids:      {train_ids_out}")
    print(f"test_ids:       {test_ids_out}")
    print(f"manifest:       {manifest_csv}")


if __name__ == "__main__":
    main()
