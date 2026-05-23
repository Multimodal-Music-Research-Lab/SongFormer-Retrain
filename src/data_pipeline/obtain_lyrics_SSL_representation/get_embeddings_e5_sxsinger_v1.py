import argparse
import json
import os
from typing import List, Dict, Tuple
from pathlib import Path

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


def load_soulx_json_paths(input_dir: str) -> List[str]:
    return sorted([str(p) for p in Path(input_dir).rglob("*.json")])


def build_output_stem(file_field: str, prefix: str = "") -> str:
    """
    BHX_0981_whenyouregone -> BHX_0981_whenyouregone
    """
    return f"{prefix}{file_field}"


def filter_and_prepare_records(
    json_paths: List[str],
    output_dir: str,
    prefix: str = "",
) -> List[Dict]:
    """
    直接读取 SoulX-Singer 输出的 json 文件列表
    """
    prepared = []

    for json_path in json_paths:
        json_path = str(json_path)
        file_field = Path(json_path).stem

        output_stem = build_output_stem(file_field=file_field, prefix=prefix)
        output_path = os.path.join(output_dir, f"{output_stem}.npz")

        prepared.append(
            {
                "json_path": json_path,
                "File": file_field,
                "Title": "",
                "Artist": "",
                "output_stem": output_stem,
                "output_path": output_path,
            }
        )

    return prepared


def get_processed_ids(output_dir: str) -> set:
    if not os.path.exists(output_dir):
        return set()

    processed = set()
    for x in os.listdir(output_dir):
        if x.endswith(".npz"):
            processed.add(x.replace(".npz", ""))
    return processed


def parse_soulx_json_to_lines(
    json_path: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    从 SoulX-Singer json 中提取 line-level 文本和时间
    返回:
        lines: List[str]
        line_start_secs: [L]
        line_end_secs: [L]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    line_records = []

    for seg in data.get("segments", []):
        for line in seg.get("lines", []):
            text = str(line.get("text", "")).strip()
            start_ms = line.get("start_ms", None)
            end_ms = line.get("end_ms", None)

            if not text:
                continue
            if start_ms is None or end_ms is None:
                continue

            start_ms = int(start_ms)
            end_ms = int(end_ms)

            if end_ms <= start_ms:
                continue

            line_records.append((start_ms, end_ms, text))

    if len(line_records) == 0:
        raise ValueError(f"No valid line records found in {json_path}")

    # 去重 + 按时间排序
    line_records = sorted(set(line_records), key=lambda x: (x[0], x[1], x[2]))

    lines = [x[2] for x in line_records]
    line_start_secs = np.asarray([x[0] / 1000.0 for x in line_records], dtype=np.float32)
    line_end_secs = np.asarray([x[1] / 1000.0 for x in line_records], dtype=np.float32)

    return lines, line_start_secs, line_end_secs


def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
    text_prefix: str = "passage",
) -> np.ndarray:
    """
    e5 encoding
    """
    if len(texts) == 0:
        raise ValueError("encode_texts got empty texts.")

    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        prefixed_texts = [f"{text_prefix}: {t}" for t in batch_texts]

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

        all_embs.append(embeddings.detach().cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


def build_song_embeddings(
    model,
    tokenizer,
    json_path: str,
    device: str,
    batch_size: int,
    line_max_length: int,
    global_max_length: int,
    text_prefix: str,
    global_mode: str = "mean_lines",
) -> Dict[str, np.ndarray]:
    """
    SoulX-Singer json -> line-level structured npz
    """
    lines, line_start_secs, line_end_secs = parse_soulx_json_to_lines(json_path=json_path)

    line_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=lines,
        device=device,
        batch_size=batch_size,
        max_length=line_max_length,
        text_prefix=text_prefix,
    ).astype(np.float32)

    cleaned_full_text = "\n".join(lines)

    if global_mode == "full_text":
        global_emb = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=[cleaned_full_text],
            device=device,
            batch_size=1,
            max_length=global_max_length,
            text_prefix=text_prefix,
        )[0].astype(np.float32)
    elif global_mode == "mean_lines":
        global_emb = line_embs.mean(axis=0)
        norm = np.linalg.norm(global_emb)
        if norm > 0:
            global_emb = (global_emb / norm).astype(np.float32)
        else:
            global_emb = global_emb.astype(np.float32)
    else:
        raise ValueError(f"Unsupported global_mode: {global_mode}")

    output = {
        "line_embs": line_embs,                              # [L, 1024]
        "line_start_secs": line_start_secs,                  # [L]
        "line_end_secs": line_end_secs,                      # [L]
        "global_emb": global_emb.astype(np.float32),         # [1024]
        "num_lines": np.asarray(len(lines), dtype=np.int32),
    }
    return output


def save_song_npz(output_path: str, arrays: Dict[str, np.ndarray]) -> None:
    """
    save npz
    """
    np.savez_compressed(output_path, **arrays)


def save_manifest(manifest_rows: List[Dict], manifest_path: str):
    import pandas as pd

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def build_manifest_row(
    rec: Dict,
    output_path: str,
    model_name: str,
    text_prefix: str,
    global_mode: str,
    line_max_length: int,
    global_max_length: int,
) -> Dict:
    data = np.load(output_path, allow_pickle=False)

    line_embs = data["line_embs"]
    global_emb = data["global_emb"]
    num_lines = int(data["num_lines"])

    row = {
        "File": rec["File"],
        "Title": rec["Title"],
        "Artist": rec["Artist"],
        "output_stem": rec["output_stem"],
        "embedding_path": output_path,
        "num_lines": num_lines,
        "line_dim": int(line_embs.shape[1]),
        "global_dim": int(global_emb.shape[0]),
        "has_line_timestamps": True,
        "text_prefix": text_prefix,
        "global_mode": global_mode,
        "line_max_length": line_max_length,
        "global_max_length": global_max_length,
        "model_name": model_name,
    }
    return row


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = args.model_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    manifest_path = args.manifest_path
    prefix = args.prefix

    print(f"Loading SoulX-Singer jsons from: {input_dir}")
    json_paths = load_soulx_json_paths(input_dir)
    print(f"Loaded {len(json_paths)} json files")

    prepared = filter_and_prepare_records(
        json_paths=json_paths,
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

    failed_rows = []

    for rec in tqdm(todo, desc="Encoding lyrics to structured npz"):
        try:
            arrays = build_song_embeddings(
                model=model,
                tokenizer=tokenizer,
                json_path=rec["json_path"],
                device=device,
                batch_size=args.batch_size,
                line_max_length=args.line_max_length,
                global_max_length=args.global_max_length,
                text_prefix=args.text_prefix,
                global_mode=args.global_mode,
            )

            save_song_npz(rec["output_path"], arrays)

        except Exception as e:
            failed_rows.append(
                {
                    "File": rec["File"],
                    "Title": rec["Title"],
                    "Artist": rec["Artist"],
                    "error": str(e),
                }
            )

    all_manifest_rows = []
    for rec in prepared:
        if os.path.exists(rec["output_path"]):
            try:
                row = build_manifest_row(
                    rec=rec,
                    output_path=rec["output_path"],
                    model_name=args.model_name,
                    text_prefix=args.text_prefix,
                    global_mode=args.global_mode,
                    line_max_length=args.line_max_length,
                    global_max_length=args.global_max_length,
                )
                all_manifest_rows.append(row)
            except Exception as e:
                failed_rows.append(
                    {
                        "File": rec["File"],
                        "Title": rec["Title"],
                        "Artist": rec["Artist"],
                        "error": f"manifest_build_failed: {str(e)}",
                    }
                )

    save_manifest(all_manifest_rows, manifest_path)
    print(f"Saved manifest to: {manifest_path}")

    if failed_rows:
        failed_path = os.path.splitext(manifest_path)[0] + "_failed.csv"
        save_manifest(failed_rows, failed_path)
        print(f"Saved failed cases to: {failed_path}")
        print(f"Failed cases: {len(failed_rows)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing SoulX-Singer lyrics json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save lyrics SSL .npz files",
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
        help="Batch size used inside text encoding",
    )
    parser.add_argument(
        "--line_max_length",
        type=int,
        default=64,
        help="Max token length for line-level encoding",
    )
    parser.add_argument(
        "--global_max_length",
        type=int,
        default=512,
        help="Max token length for global whole-lyrics encoding",
    )
    parser.add_argument(
        "--text_prefix",
        type=str,
        default="passage",
        choices=["query", "passage"],
        help="Prefix used by E5 before each text",
    )
    parser.add_argument(
        "--global_mode",
        type=str,
        default="full_text",
        choices=["full_text", "mean_lines"],
        help="How to build global_emb",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )

    args = parser.parse_args()
    main(args)