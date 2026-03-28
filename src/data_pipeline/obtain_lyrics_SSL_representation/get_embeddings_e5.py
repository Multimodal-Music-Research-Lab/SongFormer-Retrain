import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    E5 官方 model card 推荐的 mean pooling 方式。
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_jsonl(jsonl_path: str) -> List[Dict]:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def normalize_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return False


def build_output_stem(file_field: str, prefix: str = "HX_") -> str:
    """
    0001_12step-> HX_0001_12step.npy
    """
    return f"{prefix}{file_field}"


def filter_and_prepare_records(
    rows: List[Dict],
    output_dir: str,
    prefix: str = "HX_",
) -> List[Dict]:
    """
    filter valid samples for lyrics SSL
    """
    prepared = []

    for row in rows:
        file_field = row.get("File")
        plain_lyrics = row.get("plain_lyrics")
        status = row.get("status", None)

        if not file_field:
            continue

        if status != "matched":
            continue

        if plain_lyrics is None:
            continue

        plain_lyrics = str(plain_lyrics).strip()

        output_stem = build_output_stem(file_field=file_field, prefix=prefix)
        output_path = os.path.join(output_dir, f"{output_stem}.npy")

        prepared.append(
            {
                "File": file_field,
                "Title": row.get("Title", ""),
                "Artist": row.get("Artist", ""),
                "plain_lyrics": plain_lyrics,
                "output_stem": output_stem,
                "output_path": output_path,
            }
        )

    return prepared


def get_processed_ids(output_dir: str) -> set:
    """
    和现有 audio SSL 脚本类似：支持断点续跑
    """
    if not os.path.exists(output_dir):
        return set()

    processed = set()
    for x in os.listdir(output_dir):
        if x.endswith(".npy"):
            processed.add(x.replace(".npy", ""))
    return processed


def encode_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 512,
) -> np.ndarray:
    """
    这里用 multilingual-e5-large 官方 model card 推荐的方式：
    - 前缀 query:
    - average pooling
    - L2 normalize
    官方说明里明确写了：即使是非英文，也建议带 "query:" 或 "passage:" 前缀；
    对 retrieval 以外任务，可直接用 "query:" 前缀。 :contentReference[oaicite:0]{index=0}
    """
    prefixed_texts = [f"query: {t}" for t in texts]

    batch_dict = tokenizer(
        prefixed_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.detach().cpu().float().numpy()


def save_manifest(manifest_rows: List[Dict], manifest_path: str):
    import pandas as pd

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = args.model_dir
    input_jsonl = args.input_jsonl
    output_dir = args.output_dir
    manifest_path = args.manifest_path
    prefix = args.prefix

    print(f"Loading lyrics jsonl from: {input_jsonl}")
    rows = load_jsonl(input_jsonl)
    print(f"Loaded {len(rows)} rows")

    prepared = filter_and_prepare_records(
        rows=rows,
        output_dir=output_dir,
        prefix=prefix,
    )
    print(f"Rows after filtering: {len(prepared)}")

    processed_ids = get_processed_ids(output_dir)
    print(f"Already processed: {len(processed_ids)}")

    todo = [x for x in prepared if x["output_stem"] not in processed_ids]
    print(f"To process now: {len(todo)}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    manifest_rows = []
    batch_records = []
    batch_texts = []

    for row in tqdm(todo, desc="Encoding lyrics"):
        batch_records.append(row)
        batch_texts.append(row["plain_lyrics"])

        if len(batch_records) == args.batch_size:
            batch_emb = encode_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch_texts,
                device=device,
                max_length=args.max_length,
            )

            for rec, emb in zip(batch_records, batch_emb):
                np.save(rec["output_path"], emb.astype(np.float32))
                manifest_rows.append(
                    {
                        "File": rec["File"],
                        "Title": rec["Title"],
                        "Artist": rec["Artist"],
                        "output_stem": rec["output_stem"],
                        "embedding_path": rec["output_path"],
                        "dim": int(emb.shape[0]),
                        "model_name": args.model_name,
                    }
                )

            batch_records = []
            batch_texts = []

    # flush last batch
    if batch_records:
        batch_emb = encode_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            device=device,
            max_length=args.max_length,
        )

        for rec, emb in zip(batch_records, batch_emb):
            np.save(rec["output_path"], emb.astype(np.float32))
            manifest_rows.append(
                {
                    "File": rec["File"],
                    "Title": rec["Title"],
                    "Artist": rec["Artist"],
                    "output_stem": rec["output_stem"],
                    "embedding_path": rec["output_path"],
                    "dim": int(emb.shape[0]),
                    "model_name": args.model_name,
                }
            )

    # 把旧的 npy 也写进 manifest，方便统一索引
    all_manifest_rows = []
    for rec in prepared:
        if os.path.exists(rec["output_path"]):
            emb = np.load(rec["output_path"], allow_pickle=False)
            all_manifest_rows.append(
                {
                    "File": rec["File"],
                    "Title": rec["Title"],
                    "Artist": rec["Artist"],
                    "output_stem": rec["output_stem"],
                    "embedding_path": rec["output_path"],
                    "dim": int(emb.shape[0]),
                    "model_name": args.model_name,
                }
            )

    save_manifest(all_manifest_rows, manifest_path)
    print(f"Saved manifest to: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to harmonixset_lrclib_results.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save lyrics SSL .npy files",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="CSV manifest path",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Local directory of downloaded embedding model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Model name for bookkeeping",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )

    args = parser.parse_args()
    main(args)