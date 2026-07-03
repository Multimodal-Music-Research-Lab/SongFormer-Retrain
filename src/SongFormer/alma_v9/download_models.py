from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert_model_id", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--mert_dir", required=True)
    parser.add_argument("--tokenizer_model_id", default="allenai/longformer-base-4096")
    parser.add_argument("--tokenizer_dir", required=True)
    args = parser.parse_args()

    Path(args.mert_dir).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tokenizer_dir).parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.mert_model_id,
        local_dir=args.mert_dir,
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id=args.tokenizer_model_id,
        local_dir=args.tokenizer_dir,
        local_dir_use_symlinks=False,
    )
    print(f"MERT model ready: {args.mert_dir}")
    print(f"Lyrics tokenizer ready: {args.tokenizer_dir}")


if __name__ == "__main__":
    main()

