#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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

COMPOSITE_MAPPING = {
    "intro and verse": ["intro", "verse"],
    "intro and chorus": ["intro", "chorus"],
    "verse and pre-chorus": ["verse", "pre-chorus"],
    "pre-chorus and chorus": ["pre-chorus", "chorus"],
    "chorus lead-out": ["chorus", "outro"],
    "lead-in alt": ["pre-chorus"],
}


def normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("_", "-")
    s = " ".join(s.split())
    s = s.replace("pre chorus", "pre-chorus")
    parts = s.split()
    if parts and parts[-1].isdigit():
        s = " ".join(parts[:-1])

    if s not in MAPPING:
        raise ValueError(f"Unknown label in .sec: '{raw}' -> normalized '{s}'")
    return MAPPING[s]


def normalize_labels(raw: str) -> List[str]:
    if raw == "NO_LABEL":
        return ["NO_LABEL"]
    s = raw.strip().lower()
    s = s.replace("_", "-")
    s = " ".join(s.split())
    s = s.replace("pre chorus", "pre-chorus")
    if s in COMPOSITE_MAPPING:
        return COMPOSITE_MAPPING[s]
    return [normalize_label(raw)]


def parse_sec_file(
    sec_path: Path,
    blank_label: Optional[str] = None,
) -> List[Tuple[float, float, Tuple[str, ...]]]:
    """
    Parse a .sec file. Each line: <start> <end> <Label>
    Returns list of (start, end, mapped_labels).
    """
    segments: List[Tuple[float, float, Tuple[str, ...]]] = []
    with sec_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                if blank_label is None:
                    raise ValueError(f"Bad line in {sec_path} @line {ln}: '{line}'")
                label_raw = blank_label
            else:
                label_raw = " ".join(parts[2:])
            if not label_raw:
                raise ValueError(f"No label in {sec_path} @line {ln}: '{line}'")

            start_s = float(parts[0])
            end_s = float(parts[1])

            if end_s <= start_s:
                continue

            mapped = tuple(normalize_labels(label_raw))
            segments.append((start_s, end_s, mapped))

    return segments


def read_scp(scp_path: Path) -> List[Path]:
    audio_paths: List[Path] = []
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().replace("\ufeff", "").replace("\r", "")
            if not p or p.startswith("#"):
                continue
            audio_paths.append(Path(p.split()[-1]))
    return audio_paths


def get_song_prefix_from_stem(stem: str) -> str:
    """
    audio stem example:
      <artist>_<title>_<secid>
    return:
      <artist>_<title>
    """
    parts = [x for x in stem.split("_") if x]
    if len(parts) < 2:
        raise ValueError(f"Cannot get HookTheory song prefix from stem: {stem}")
    return f"{parts[0]}_{parts[1]}"


def get_song_prefix_from_audio(audio_path: Path) -> str:
    return get_song_prefix_from_stem(audio_path.stem)


def dedup_audio_paths_by_song(audio_paths: Iterable[Path]) -> List[Path]:
    kept: "OrderedDict[str, Path]" = OrderedDict()
    for audio_path in audio_paths:
        key = get_song_prefix_from_audio(audio_path)
        kept.setdefault(key, audio_path)
    return list(kept.values())


def build_measure_sec_index(section_dir: Path, sec_ext: str) -> Dict[str, List[Path]]:
    """
    Index only HookTheory *_measure.sec files by <artist>_<title>.
    Melody section files are intentionally ignored.
    """
    index: Dict[str, List[Path]] = defaultdict(list)
    suffix = f"_measure{sec_ext}"
    for sec_path in section_dir.rglob(f"*{suffix}"):
        name = sec_path.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        key = get_song_prefix_from_stem(stem)
        index[key].append(sec_path)
    for paths in index.values():
        paths.sort()
    return index


def find_sec_files_for_song(
    audio_path: Path,
    section_dir: Path,
    sec_ext: str,
    sec_index: Optional[Dict[str, List[Path]]] = None,
) -> List[Path]:
    """
    Find all:
      <artist>_<title>_*_measure.sec
    under section_dir.
    """
    song_prefix = get_song_prefix_from_audio(audio_path)
    if sec_index is not None:
        return list(sec_index.get(song_prefix, []))
    pattern = f"{song_prefix}_*_measure{sec_ext}"
    return sorted(section_dir.rglob(pattern))


def build_jsonl(
    scp_path: Path,
    section_dir: Path,
    out_jsonl: Path,
    bad_sec_list_path: Optional[Path],
    sec_ext: str = ".sec",
    require_sec: bool = True,
    dedup_by_song: bool = False,
    blank_label: Optional[str] = None,
) -> None:
    audio_paths = read_scp(scp_path)
    if dedup_by_song:
        audio_paths = dedup_audio_paths_by_song(audio_paths)

    sec_index = build_measure_sec_index(section_dir, sec_ext)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if bad_sec_list_path:
        bad_sec_list_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(audio_paths)
    audio_missing = 0
    sec_missing = 0
    bad_sec = 0
    segments_written = 0
    songs_written = 0
    sec_files_used = set()
    bad_sec_paths = set()

    with out_jsonl.open("w", encoding="utf-8") as w:
        for audio_path in audio_paths:
            if not audio_path.exists():
                audio_missing += 1
                continue

            sec_paths = find_sec_files_for_song(audio_path, section_dir, sec_ext, sec_index)
            if not sec_paths:
                sec_missing += 1
                if require_sec:
                    continue
                continue

            all_segments: List[Tuple[float, float, Tuple[str, ...]]] = []
            song_had_good_sec = False
            for sec_path in sec_paths:
                try:
                    segs = parse_sec_file(sec_path, blank_label=blank_label)
                    all_segments.extend(segs)
                    sec_files_used.add(str(sec_path))
                    song_had_good_sec = True
                except Exception:
                    bad_sec += 1
                    bad_sec_paths.add(str(sec_path))
                    continue

            if not song_had_good_sec:
                continue

            # Preserve the historical jsonl schema: one line per segment.
            # Multiple section files for the same duplicated audio identity are
            # merged into the representative ori_audio_path.
            all_segments = sorted(set(all_segments), key=lambda x: (x[0], x[1], x[2]))

            for st, ed, labels in all_segments:
                obj = {
                    "ori_audio_path": str(audio_path),
                    "segment_start": float(st),
                    "segment_end": float(ed),
                    "label": list(labels),
                }
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                segments_written += 1
            songs_written += 1

    if bad_sec_list_path:
        with bad_sec_list_path.open("w", encoding="utf-8") as f:
            for p in sorted(bad_sec_paths):
                f.write(p + "\n")

    print("Done.")
    print(f"SCP entries after optional dedup: {total}")
    print(f"Missing audio paths skipped: {audio_missing}")
    print(f"Songs missing any _measure.sec skipped: {sec_missing} (require_sec={require_sec})")
    print(f"Bad .sec files skipped: {bad_sec}")
    print(f"Unique songs written: {songs_written}")
    print(f"Measure .sec files used: {len(sec_files_used)}")
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
        "--dedup_by_song",
        action="store_true",
        help="Deduplicate SCP entries by <artist>_<title> before writing jsonl.",
    )
    ap.add_argument(
        "--blank_label",
        choices=["error", "no_label"],
        default="error",
        help="How to handle .sec rows that contain start/end times but no label.",
    )
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
        dedup_by_song=args.dedup_by_song,
        blank_label="NO_LABEL" if args.blank_label == "no_label" else None,
    )


if __name__ == "__main__":
    main()
