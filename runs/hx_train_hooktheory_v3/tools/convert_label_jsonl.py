#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple


MAPPING = {
    "bridge": "bridge",
    "chorus": "chorus",
    "instrumental": "inst",
    "intro": "intro",
    "lead-in": "pre-chorus",
    "loop": "inst",
    "outro": "outro",
    "pre-chorus": "pre-chorus",
    "pre-outro": "outro",
    "solo": "inst",
    "verse": "verse",
    "silence": "silence",
}


def normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("_", "-")
    s = " ".join(s.split())
    s = s.replace("pre chorus", "pre-chorus")

    if s not in MAPPING:
        raise ValueError(f"Unknown label in .sec: '{raw}' -> normalized '{s}'")
    return MAPPING[s]


def parse_sec_file(sec_path: Path) -> List[Tuple[float, float, str]]:
    """
    Parse a .sec file. Each line: <start> <end> <Label>
    Returns list of (start, end, mapped_label).
    """
    segments: List[Tuple[float, float, str]] = []
    with sec_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Bad line in {sec_path} @line {ln}: '{line}'")
            if len(parts) > 3:
                raise ValueError(f"Multiple labels in {sec_path} @line {ln}: '{line}'")
            if not parts[2]:
                raise ValueError(f"No label in {sec_path} @line {ln}: '{line}'")
            
            start_s = float(parts[0])
            end_s = float(parts[1])
            label_raw = " ".join(parts[2:])
            
            if end_s <= start_s:
                continue

            mapped = normalize_label(label_raw)
            segments.append((start_s, end_s, mapped))

    return segments


def read_scp(scp_path: Path) -> List[Path]:
    audio_paths: List[Path] = []
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().replace("\ufeff", "").replace("\r", "")
            if not p or p.startswith("#"):
                continue
            audio_paths.append(Path(p))
    return audio_paths


def get_song_prefix_from_audio(audio_path: Path) -> str:
    """
    audio stem example:
      <artist>_<title>_<segid>
    return:
      <artist>_<title>
    """
    parts = [x for x in audio_path.stem.split("_") if x]
    return f"{parts[0]}_{parts[1]}"


def find_sec_files_for_song(
    audio_path: Path, section_dir: Path, sec_ext: str
) -> List[Path]:
    """
    Find all:
      <artist>_<title>_*_measure.sec
    under section_dir (recursive).
    """
    song_prefix = get_song_prefix_from_audio(audio_path)
    # all sec files whose artist name and song name are aligned with the audio file
    pattern = f"{song_prefix}_*_measure{sec_ext}"
    sec_files = sorted(section_dir.rglob(pattern))
    return sec_files


def build_jsonl(
    scp_path: Path,
    section_dir: Path,
    out_jsonl: Path,
    bad_sec_list_path: Optional[Path],
    sec_ext: str = ".sec",
    require_sec: bool = True,
) -> None:
    audio_paths = read_scp(scp_path)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if bad_sec_list_path:
        bad_sec_list_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(audio_paths)
    audio_missing = 0
    sec_missing = 0
    bad_sec = 0
    segments_written = 0

    bad_sec_paths = set()

    with out_jsonl.open("w", encoding="utf-8") as w:
        for audio_path in audio_paths:
            if not audio_path.exists():
                audio_missing += 1
                continue

            sec_paths = find_sec_files_for_song(audio_path, section_dir, sec_ext)
            if not sec_paths:
                sec_missing += 1
                if require_sec:
                    continue
                else:
                    continue

            all_segments: List[Tuple[float, float, str]] = []
            for sec_path in sec_paths:
                try:
                    segs = parse_sec_file(sec_path)
                    all_segments.extend(segs)
                except Exception:
                    bad_sec += 1
                    bad_sec_paths.add(str(sec_path))
                    continue

            # drop duplicate
            all_segments = sorted(set(all_segments), key=lambda x: (x[0], x[1], x[2]))

            for st, ed, lab in all_segments:
                obj = {
                    "ori_audio_path": str(audio_path),
                    "segment_start": float(st),
                    "segment_end": float(ed),
                    "label": [lab],
                }
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                segments_written += 1

    if bad_sec_list_path:
        with bad_sec_list_path.open("w", encoding="utf-8") as f:
            for p in sorted(bad_sec_paths):
                f.write(p + "\n")

    print("Done.")
    print(f"SCP entries: {total}")
    print(f"Missing audio paths skipped: {audio_missing}")
    print(f"Songs missing any .sec skipped: {sec_missing} (require_sec={require_sec})")
    print(f"Bad .sec files skipped: {bad_sec}")
    print(f"Segments written: {segments_written}")
    print(f"Output: {out_jsonl}")
    if bad_sec_list_path:
        print(f"Bad .sec list: {bad_sec_list_path} (unique={len(bad_sec_paths)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scp", type=str, required=True, help="Path to .scp file (one mp3 path per line)")
    ap.add_argument("--section_dir", type=str, required=True, help="Directory containing .sec files")
    ap.add_argument("--out", type=str, required=True, help="Output jsonl path")
    ap.add_argument("--sec_ext", type=str, default=".sec", help='Section file extension (default: ".sec")')
    ap.add_argument("--allow_missing_sec", action="store_true", help="Skip songs without any matching .sec")
    ap.add_argument(
        "--bad_sec_list",
        type=str,
        default=None,
        help="Write paths of .sec files that failed parsing to this text file",
    )
    args = ap.parse_args()

    build_jsonl(
        scp_path=Path(args.scp),
        section_dir=Path(args.section_dir),
        out_jsonl=Path(args.out),
        bad_sec_list_path=Path(args.bad_sec_list) if args.bad_sec_list else None,
        sec_ext=args.sec_ext,
        require_sec=not args.allow_missing_sec,
    )


if __name__ == "__main__":
    main()
