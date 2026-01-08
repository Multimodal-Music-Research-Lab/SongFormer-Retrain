from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create an ID list from an .scp file (one path per line)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scp",
        type=str,
        required=True,
        help="Path to .scp file (one absolute audio path per line)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    p.add_argument(
        "--out_name",
        type=str,
        default="train.txt",
        help="Output filename, e.g. train.txt / eval.txt / val.txt",
    )
    return p.parse_args()


def read_scp_paths(scp_path: Path) -> List[str]:
    paths: List[str] = []
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace("\ufeff", "").replace("\r", "")
            if not s or s.startswith("#"):
                continue
            paths.append(s)
    return paths


def main() -> None:
    args = parse_args()
    scp_path = Path(args.scp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = read_scp_paths(scp_path)
    ids = sorted(set(Path(p).stem for p in audio_paths))

    out_path = out_dir / args.out_name
    out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")

    print("=== ID list generation done ===")
    print(f"scp:      {scp_path}")
    print(f"out_path: {out_path}")
    print(f"count:    {len(ids)}")
    print("first 5:")
    for x in ids[:5]:
        print(" ", x)


if __name__ == "__main__":
    main()
