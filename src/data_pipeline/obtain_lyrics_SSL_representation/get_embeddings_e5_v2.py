import argparse
import json
import os
from typing import List, Dict, Tuple

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


def build_output_stem(file_field: str, prefix: str = "HX_") -> str:
    """
    0001_12step -> HX_0001_12step
    """
    return f"{prefix}{file_field}"


def filter_and_prepare_records(
    rows: List[Dict],
    output_dir: str,
    prefix: str = "HX_",
) -> List[Dict]:
    """
    过滤出可用于 lyrics SSL 的样本
    """
    prepared = []

    for row in rows:
        file_field = row.get("File")
        plain_lyrics = row.get("plain_lyrics")
        match_score = row.get("match_score", None)

        if not file_field:
            continue

        if match_score <= 0.5:
            continue

        if plain_lyrics is None:
            continue

        plain_lyrics = str(plain_lyrics).strip()
        if not plain_lyrics:
            continue

        output_stem = build_output_stem(file_field=file_field, prefix=prefix)
        output_path = os.path.join(output_dir, f"{output_stem}.npz")

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
    if not os.path.exists(output_dir):
        return set()

    processed = set()
    for x in os.listdir(output_dir):
        if x.endswith(".npz"):
            processed.add(x.replace(".npz", ""))
    return processed


def normalize_lyrics_text(text: str) -> str:
    """
    pre-processing for lyrics text:
    1. \\r\\n / \\r -> \\n
    2. strip every line
    3. drop empty line for start and end
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_lines = text.split("\n")

    normalized_lines = []
    prev_blank = False

    for line in raw_lines:
        line = line.strip()

        if line == "":
            if not prev_blank:
                normalized_lines.append("")
            prev_blank = True
        else:
            normalized_lines.append(line)
            prev_blank = False

    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    cleaned_text = "\n".join(normalized_lines)
    return cleaned_text


def split_lyrics_into_stanzas_and_lines(text: str) -> Tuple[List[str], List[str], List[int]]:
    """
    split into:
    - stanzas: by \n\n
    - lines: by \n
    - line_to_stanza: map line to stanza

    return：
    stanzas: List[str]
    lines: List[str]
    line_to_stanza: List[int]
    """
    if not text.strip():
        return [], [], []

    raw_stanzas = text.split("\n\n")

    stanzas = []
    lines = []
    line_to_stanza = []

    for stanza_idx, stanza in enumerate(raw_stanzas):
        stanza = stanza.strip()
        if not stanza:
            continue

        stanza_lines = [x.strip() for x in stanza.split("\n") if x.strip()]
        if not stanza_lines:
            continue

        stanza_text = "\n".join(stanza_lines)
        stanzas.append(stanza_text)

        real_stanza_idx = len(stanzas) - 1
        for line in stanza_lines:
            lines.append(line)
            line_to_stanza.append(real_stanza_idx)

    return stanzas, lines, line_to_stanza


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
    plain_lyrics: str,
    device: str,
    batch_size: int,
    line_max_length: int,
    stanza_max_length: int,
    global_max_length: int,
    text_prefix: str,
    global_mode: str = "full_text",
) -> Dict[str, np.ndarray]:
    """
    song to npz info
    """
    # pre-processing
    cleaned_text = normalize_lyrics_text(plain_lyrics)

    # split stanza and line
    stanzas, lines, line_to_stanza = split_lyrics_into_stanzas_and_lines(cleaned_text)

    if len(lines) == 0 or len(stanzas) == 0:
        raise ValueError("No valid lines/stanzas after preprocessing.")

    # line-level embeddings
    line_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=lines,
        device=device,
        batch_size=batch_size,
        max_length=line_max_length,
        text_prefix=text_prefix,
    ).astype(np.float32)

    # stanza-level embeddings
    stanza_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=stanzas,
        device=device,
        batch_size=batch_size,
        max_length=stanza_max_length,
        text_prefix=text_prefix,
    ).astype(np.float32)

    # global embedding
    if global_mode == "full_text":
        global_emb = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=[cleaned_text],
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
        "line_embs": line_embs,                                      # [L, 1024]
        "stanza_embs": stanza_embs,                                  # [S, 1024]
        "line_to_stanza": np.asarray(line_to_stanza, dtype=np.int64),# [L]
        "global_emb": global_emb.astype(np.float32),                 # [1024]
        "num_lines": np.asarray(len(lines), dtype=np.int32),
        "num_stanzas": np.asarray(len(stanzas), dtype=np.int32),
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
    stanza_max_length: int,
    global_max_length: int,
) -> Dict:
    data = np.load(output_path, allow_pickle=False)

    line_embs = data["line_embs"]
    stanza_embs = data["stanza_embs"]
    global_emb = data["global_emb"]
    num_lines = int(data["num_lines"])
    num_stanzas = int(data["num_stanzas"])

    row = {
        "File": rec["File"],
        "Title": rec["Title"],
        "Artist": rec["Artist"],
        "output_stem": rec["output_stem"],
        "embedding_path": output_path,
        "num_lines": num_lines,
        "num_stanzas": num_stanzas,
        "line_dim": int(line_embs.shape[1]),
        "stanza_dim": int(stanza_embs.shape[1]),
        "global_dim": int(global_emb.shape[0]),
        "text_prefix": text_prefix,
        "global_mode": global_mode,
        "line_max_length": line_max_length,
        "stanza_max_length": stanza_max_length,
        "global_max_length": global_max_length,
        "model_name": model_name,
    }
    return row


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

    failed_rows = []

    for rec in tqdm(todo, desc="Encoding lyrics to structured npz"):
        try:
            arrays = build_song_embeddings(
                model=model,
                tokenizer=tokenizer,
                plain_lyrics=rec["plain_lyrics"],
                device=device,
                batch_size=args.batch_size,
                line_max_length=args.line_max_length,
                stanza_max_length=args.stanza_max_length,
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
                    stanza_max_length=args.stanza_max_length,
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
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to harmonixset_lrclib_results.jsonl",
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
        "--stanza_max_length",
        type=int,
        default=192,
        help="Max token length for stanza-level encoding",
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