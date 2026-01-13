#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deduplicate HookTheory scp by unique song key = "<artist>_<song>".
Keep the FIRST occurrence path for each song in the original scp order.
""" 

from pathlib import Path

script_path = Path(__file__).resolve()
base_path = script_path.parent.parent

SCP_IN = (base_path / "results/hooktheory_all.scp").resolve()
SCP_OUT = (base_path / "results/hooktheory_all_unique.scp").resolve()

def read_scp_lines(p: Path) -> list[str]:
    lines: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace("\ufeff", "").replace("\r", "")
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def song_key_from_audio_path(audio_path: str) -> str:
    """
    Expect filename stem like:
      <artist>_<song>_<segmentid>
    Dedup by <artist>_<song>.
    """
    stem = Path(audio_path).stem
    parts = [x for x in stem.split("_") if x]
    return f"{parts[0]}_{parts[1]}"


def main():
    paths = read_scp_lines(SCP_IN)

    kept = {}  # key -> first path
    dup_cnt = 0

    for p in paths:
        key = song_key_from_audio_path(p)
        if key not in kept:
            kept[key] = p
        else:
            dup_cnt += 1

    unique_paths = list(kept.values())

    SCP_OUT.write_text("\n".join(unique_paths) + ("\n" if unique_paths else ""), encoding="utf-8")

    print("=== hooktheory_unique_song done ===")
    print(f"scp_in:   {SCP_IN}")
    print(f"scp_out:  {SCP_OUT}")
    print(f"total_in: {len(paths)}")
    print(f"unique:   {len(unique_paths)}")
    print(f"dups:     {dup_cnt}")
    print("first 5 unique:")
    for x in unique_paths[:5]:
        print(" ", x)


if __name__ == "__main__":
    main()

