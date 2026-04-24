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
    过滤出可用于 sync lyrics SSL 的样本
    """
    prepared = []

    for row in rows:
        file_field = row.get("File")
        synced_lyrics = row.get("synced_lyrics")
        match_score = row.get("match_score", None)

        if not file_field:
            continue

        if match_score is None or match_score <= 0.5:
            continue

        if synced_lyrics is None:
            continue

        synced_lyrics = str(synced_lyrics).strip()
        if not synced_lyrics:
            continue

        # prefer matched_duration; fallback to original duration
        duration_sec = row.get("matched_duration", None)
        if duration_sec is None:
            duration_sec = row.get("Duration", None)

        output_stem = build_output_stem(file_field=file_field, prefix=prefix)
        output_path = os.path.join(output_dir, f"{output_stem}.npz")

        prepared.append(
            {
                "File": file_field,
                "Title": row.get("Title", ""),
                "Artist": row.get("Artist", ""),
                "synced_lyrics": synced_lyrics,
                "duration_sec": duration_sec,
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


def parse_lrc_timestamp(ts: str) -> float:
    """
    'mm:ss.xx' -> seconds
    """
    mm, ss = ts.split(":")
    return int(mm) * 60.0 + float(ss)


def parse_synced_lyrics_to_structures(
    synced_lyrics: str,
    fallback_duration_sec: float = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse LRC-style synced lyrics into:
    - line texts / line start / line end
    - stanza texts / stanza start / stanza end
    - line_to_stanza

    Stanza split rule:
    - blank synced lyric line -> stanza boundary
    """
    text = synced_lyrics.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")

    records = []
    for raw_line in raw_lines:
        raw_line = raw_line.rstrip()
        if not raw_line:
            continue
        if not raw_line.startswith("[") or "]" not in raw_line:
            continue

        right_idx = raw_line.find("]")
        ts_str = raw_line[1:right_idx].strip()
        lyric_text = raw_line[right_idx + 1 :].strip()

        try:
            ts_sec = parse_lrc_timestamp(ts_str)
        except Exception:
            continue

        records.append((ts_sec, lyric_text))

    if len(records) == 0:
        raise ValueError("No valid synced lyric records found.")

    records = sorted(records, key=lambda x: x[0])

    if fallback_duration_sec is None:
        fallback_duration_sec = records[-1][0] + 3.0
    fallback_duration_sec = max(float(fallback_duration_sec), records[-1][0] + 0.5)

    lines = []
    line_start_secs = []
    line_end_secs = []

    stanzas = []
    stanza_start_secs = []
    stanza_end_secs = []

    line_to_stanza = []

    current_stanza_lines = []
    current_stanza_start = None
    current_stanza_end = None

    for idx, (cur_t, cur_text) in enumerate(records):
        if idx + 1 < len(records):
            next_t = records[idx + 1][0]
        else:
            next_t = fallback_duration_sec

        if next_t <= cur_t:
            next_t = cur_t + 0.5

        # blank synced line => stanza separator
        if cur_text == "":
            if len(current_stanza_lines) > 0:
                stanzas.append("\n".join(current_stanza_lines))
                stanza_start_secs.append(current_stanza_start)
                stanza_end_secs.append(current_stanza_end)
                current_stanza_lines = []
                current_stanza_start = None
                current_stanza_end = None
            continue

        # line unit
        lines.append(cur_text)
        line_start_secs.append(cur_t)
        line_end_secs.append(next_t)
        line_to_stanza.append(len(stanzas))

        # current stanza accumulation
        if current_stanza_start is None:
            current_stanza_start = cur_t
        current_stanza_lines.append(cur_text)
        current_stanza_end = next_t

    if len(current_stanza_lines) > 0:
        stanzas.append("\n".join(current_stanza_lines))
        stanza_start_secs.append(current_stanza_start)
        stanza_end_secs.append(current_stanza_end)

    if len(lines) == 0 or len(stanzas) == 0:
        raise ValueError("No valid lines/stanzas after parsing synced lyrics.")

    return (
        lines,
        np.asarray(line_start_secs, dtype=np.float32),
        np.asarray(line_end_secs, dtype=np.float32),
        stanzas,
        np.asarray(line_to_stanza, dtype=np.int64),
        np.asarray(stanza_start_secs, dtype=np.float32),
        np.asarray(stanza_end_secs, dtype=np.float32),
    )


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
    synced_lyrics: str,
    song_duration_sec: float,
    device: str,
    batch_size: int,
    line_max_length: int,
    stanza_max_length: int,
    global_max_length: int,
    text_prefix: str,
    global_mode: str = "full_text",
) -> Dict[str, np.ndarray]:
    """
    synced lyrics -> structured npz
    """
    (
        lines,
        line_start_secs,
        line_end_secs,
        stanzas,
        line_to_stanza,
        stanza_start_secs,
        stanza_end_secs,
    ) = parse_synced_lyrics_to_structures(
        synced_lyrics=synced_lyrics,
        fallback_duration_sec=song_duration_sec,
    )

    line_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=lines,
        device=device,
        batch_size=batch_size,
        max_length=line_max_length,
        text_prefix=text_prefix,
    ).astype(np.float32)

    stanza_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=stanzas,
        device=device,
        batch_size=batch_size,
        max_length=stanza_max_length,
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
        "line_embs": line_embs,                                      # [L, 1024]
        "stanza_embs": stanza_embs,                                  # [S, 1024]
        "line_to_stanza": line_to_stanza,                            # [L]
        "line_start_secs": line_start_secs,                          # [L]
        "line_end_secs": line_end_secs,                              # [L]
        "stanza_start_secs": stanza_start_secs,                      # [S]
        "stanza_end_secs": stanza_end_secs,                          # [S]
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
        "has_line_timestamps": True,
        "has_stanza_timestamps": True,
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
                synced_lyrics=rec["synced_lyrics"],
                song_duration_sec=rec["duration_sec"],
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