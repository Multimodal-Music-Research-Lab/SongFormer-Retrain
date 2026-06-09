import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


LABELS = ["intro", "verse", "chorus", "bridge", "inst", "outro", "pre-chorus", "silence"]
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}
ID_TO_LABEL = {i: x for x, i in LABEL_TO_ID.items()}


def normalize_label(label: str) -> str:
    label = (label or "silence").lower().strip()
    mapping = {
        "prechorus": "pre-chorus",
        "pre chorus": "pre-chorus",
        "instrumental": "inst",
        "interlude": "inst",
        "solo": "inst",
        "end": "silence",
        "no_label": "silence",
        "no label": "silence",
    }
    label = mapping.get(label, label)
    return label if label in LABEL_TO_ID else "silence"


def read_ids(path: str) -> List[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def read_scp_ids(path: str) -> List[str]:
    return [Path(x).stem for x in read_ids(path)]


def load_msa_txt(path: Path) -> List[Tuple[float, float, str]]:
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        entries.append((float(parts[0]), normalize_label(parts[1])))
    intervals = []
    for i, (start, label) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else start
        if label == "end":
            continue
        if end > start:
            intervals.append((start, end, label))
    return intervals


def load_bench_labels(ann_dir: str) -> Dict[str, List[Tuple[float, float, str]]]:
    out = {}
    for path in Path(ann_dir).glob("*.txt"):
        out[path.stem] = load_msa_txt(path)
    return out


def load_hx_labels(path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    out: Dict[str, List[Tuple[float, float, str]]] = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            labels = obj["labels"]
            duration = float(obj.get("duration", labels[-1][0] if labels else 0.0))
            intervals = []
            for i, item in enumerate(labels):
                start = float(item[0])
                lab = normalize_label(item[1])
                end = float(labels[i + 1][0]) if i + 1 < len(labels) else duration
                if end > start and lab != "end":
                    intervals.append((start, end, lab))
            out[obj["id"]] = intervals
    return out


def load_hook_labels(paths: List[str]) -> Dict[str, List[Tuple[float, float, str]]]:
    grouped: Dict[str, List[Tuple[float, float, str]]] = {}
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                song_id = Path(obj["ori_audio_path"]).stem
                label = obj.get("label", ["silence"])
                if isinstance(label, list):
                    label = label[0] if label else "silence"
                grouped.setdefault(song_id, []).append(
                    (float(obj["segment_start"]), float(obj["segment_end"]), normalize_label(label))
                )
    for song_id in list(grouped):
        grouped[song_id] = sorted(grouped[song_id], key=lambda x: (x[0], x[1]))
    return grouped


def flatten_soulx_lines(path: Path) -> List[Dict]:
    obj = json.loads(path.read_text())
    rows = []
    for seg in obj.get("segments", []):
        lines = seg.get("lines") or []
        if not lines and seg.get("text"):
            lines = [seg]
        for line in lines:
            text = (line.get("text") or "").strip()
            start_ms = line.get("start_ms", seg.get("start_ms"))
            end_ms = line.get("end_ms", seg.get("end_ms"))
            if not text or start_ms is None or end_ms is None:
                continue
            start = float(start_ms) / 1000.0
            end = float(end_ms) / 1000.0
            if end <= start:
                continue
            rows.append({"text": text, "start": start, "end": end})
    rows.sort(key=lambda x: (x["start"], x["end"]))
    return rows


def overlap_label(line_start: float, line_end: float, intervals: List[Tuple[float, float, str]]) -> int:
    best_label = "silence"
    best_overlap = 0.0
    mid = 0.5 * (line_start + line_end)
    mid_label = None
    for start, end, label in intervals:
        ov = max(0.0, min(line_end, end) - max(line_start, start))
        if ov > best_overlap:
            best_overlap = ov
            best_label = label
        if start <= mid < end:
            mid_label = label
    if best_overlap <= 0.0 and mid_label is not None:
        best_label = mid_label
    return LABEL_TO_ID[normalize_label(best_label)]


def boundary_label(line_start: float, intervals: List[Tuple[float, float, str]], tolerance: float) -> float:
    boundaries = [x[0] for x in intervals[1:]]
    return float(any(abs(line_start - b) <= tolerance for b in boundaries))


@dataclass
class SongExample:
    song_id: str
    lines: List[Dict]
    intervals: List[Tuple[float, float, str]]
    source: str


class LyricsLineDataset(Dataset):
    def __init__(self, examples: List[SongExample], tokenizer, cfg: Dict):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = int(cfg["max_length"])
        self.boundary_tolerance = float(cfg.get("boundary_tolerance_sec", 3.0))
        self.line_token_id = tokenizer.convert_tokens_to_ids(cfg["line_token"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids = [self.tokenizer.cls_token_id]
        line_positions, labels, boundary, starts, ends, kept_text = [], [], [], [], [], []
        for line in ex.lines:
            ids = self.tokenizer.encode(line["text"], add_special_tokens=False)
            if len(input_ids) + 1 + len(ids) + 1 > self.max_length:
                break
            line_positions.append(len(input_ids))
            input_ids.append(self.line_token_id)
            input_ids.extend(ids)
            labels.append(overlap_label(line["start"], line["end"], ex.intervals))
            boundary.append(boundary_label(line["start"], ex.intervals, self.boundary_tolerance))
            starts.append(line["start"])
            ends.append(line["end"])
            kept_text.append(line["text"])
        input_ids.append(self.tokenizer.sep_token_id)
        if not line_positions:
            line_positions = [1]
            input_ids = [self.tokenizer.cls_token_id, self.line_token_id, self.tokenizer.sep_token_id]
            labels = [LABEL_TO_ID["silence"]]
            boundary = [0.0]
            starts = [0.0]
            ends = [0.01]
            kept_text = [""]
        attention_mask = [1] * len(input_ids)
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1
        for pos in line_positions:
            global_attention_mask[pos] = 1
        return {
            "song_id": ex.song_id,
            "source": ex.source,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "global_attention_mask": torch.tensor(global_attention_mask, dtype=torch.long),
            "line_positions": torch.tensor(line_positions, dtype=torch.long),
            "line_labels": torch.tensor(labels, dtype=torch.long),
            "boundary_labels": torch.tensor(boundary, dtype=torch.float),
            "line_starts": torch.tensor(starts, dtype=torch.float),
            "line_ends": torch.tensor(ends, dtype=torch.float),
            "intervals": ex.intervals,
            "texts": kept_text,
        }


def collate(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    max_lines = max(x["line_positions"].shape[0] for x in batch)
    out = {}
    for key in ["input_ids", "attention_mask", "global_attention_mask"]:
        pad_value = 0
        out[key] = torch.full((len(batch), max_len), pad_value, dtype=torch.long)
        for i, item in enumerate(batch):
            out[key][i, : item[key].shape[0]] = item[key]
    out["line_positions"] = torch.zeros((len(batch), max_lines), dtype=torch.long)
    out["line_labels"] = torch.full((len(batch), max_lines), -100, dtype=torch.long)
    out["boundary_labels"] = torch.zeros((len(batch), max_lines), dtype=torch.float)
    out["line_mask"] = torch.zeros((len(batch), max_lines), dtype=torch.bool)
    out["line_starts"] = torch.zeros((len(batch), max_lines), dtype=torch.float)
    out["line_ends"] = torch.zeros((len(batch), max_lines), dtype=torch.float)
    for i, item in enumerate(batch):
        n = item["line_positions"].shape[0]
        out["line_positions"][i, :n] = item["line_positions"]
        out["line_labels"][i, :n] = item["line_labels"]
        out["boundary_labels"][i, :n] = item["boundary_labels"]
        out["line_mask"][i, :n] = True
        out["line_starts"][i, :n] = item["line_starts"]
        out["line_ends"][i, :n] = item["line_ends"]
    out["song_id"] = [x["song_id"] for x in batch]
    out["source"] = [x["source"] for x in batch]
    out["intervals"] = [x["intervals"] for x in batch]
    out["texts"] = [x["texts"] for x in batch]
    return out


class LongformerLineClassifier(nn.Module):
    def __init__(self, model_path: str, vocab_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.encoder.resize_token_embeddings(vocab_size)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.function_head = nn.Linear(hidden, num_labels)
        self.boundary_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, global_attention_mask, line_positions):
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        ).last_hidden_state
        gather_index = line_positions.unsqueeze(-1).expand(-1, -1, enc.shape[-1])
        line_hidden = enc.gather(1, gather_index)
        line_hidden = self.dropout(line_hidden)
        return self.function_head(line_hidden), self.boundary_head(line_hidden).squeeze(-1)


def load_examples(cfg: Dict, split: str) -> List[SongExample]:
    data_cfg = cfg["data"]
    hx_labels = load_hx_labels(data_cfg["hx_label_path"])
    hook_labels = load_hook_labels(data_cfg["hook_structure_jsonl_paths"])
    examples = []

    if split in data_cfg["splits"].get("hx", {}):
        for song_id in read_ids(data_cfg["splits"]["hx"][split]):
            lyric_path = Path(data_cfg["hx_lyrics_dir"]) / f"{song_id}.json"
            if lyric_path.exists() and song_id in hx_labels:
                lines = flatten_soulx_lines(lyric_path)
                if lines:
                    examples.append(SongExample(song_id, lines, hx_labels[song_id], "hx"))

    if split in data_cfg["splits"].get("hook", {}):
        for song_id in read_ids(data_cfg["splits"]["hook"][split]):
            lyric_path = Path(data_cfg["hook_lyrics_dir"]) / f"{song_id}.json"
            if lyric_path.exists() and song_id in hook_labels:
                lines = flatten_soulx_lines(lyric_path)
                if lines:
                    examples.append(SongExample(song_id, lines, hook_labels[song_id], "hook"))
    return examples


def load_bench_examples(cfg: Dict) -> List[SongExample]:
    data_cfg = cfg["bench"]
    bench_labels = load_bench_labels(data_cfg["ann_dir"])
    examples = []
    for song_id in read_scp_ids(data_cfg["scp_path"]):
        lyric_path = Path(data_cfg["lyrics_dir"]) / f"{song_id}.json"
        if lyric_path.exists() and song_id in bench_labels:
            lines = flatten_soulx_lines(lyric_path)
            if lines:
                examples.append(SongExample(song_id, lines, bench_labels[song_id], "bench"))
    return examples


def intervals_duration(intervals: List[Tuple[float, float, str]], fallback: float = 1.0) -> float:
    return max([end for _, end, _ in intervals] + [fallback])


def line_predictions_to_segments(
    pred_labels: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    duration: float,
    min_segment_dur: float = 0.25,
) -> List[Dict]:
    if len(pred_labels) == 0:
        return [{"start": 0.0, "end": float(duration), "label": "silence"}]

    events = []
    first_start = max(0.0, float(starts[0]))
    events.append((0.0, ID_TO_LABEL[int(pred_labels[0])]))
    if first_start > 0:
        events.append((first_start, ID_TO_LABEL[int(pred_labels[0])]))
    for i, label_id in enumerate(pred_labels):
        events.append((max(0.0, float(starts[i])), ID_TO_LABEL[int(label_id)]))
        if i + 1 < len(pred_labels) and ends[i] < starts[i + 1]:
            events.append((max(0.0, float(ends[i])), ID_TO_LABEL[int(label_id)]))

    events = sorted(events, key=lambda x: x[0])
    merged_events = []
    for t, lab in events:
        t = min(max(0.0, t), float(duration))
        if not merged_events:
            merged_events.append((t, lab))
        elif lab != merged_events[-1][1]:
            if t > merged_events[-1][0]:
                merged_events.append((t, lab))
            else:
                merged_events[-1] = (merged_events[-1][0], lab)

    segments = []
    for i, (start, lab) in enumerate(merged_events):
        end = merged_events[i + 1][0] if i + 1 < len(merged_events) else float(duration)
        if end - start < min_segment_dur:
            continue
        if segments and segments[-1]["label"] == lab:
            segments[-1]["end"] = end
        else:
            segments.append({"start": float(start), "end": float(end), "label": lab})

    if not segments:
        return [{"start": 0.0, "end": float(duration), "label": ID_TO_LABEL[int(pred_labels[0])]}]
    if segments[0]["start"] > 0.0:
        segments[0]["start"] = 0.0
    segments[-1]["end"] = float(duration)
    return segments


def frame_projection(
    pred_labels: np.ndarray,
    boundary_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    intervals: List[Tuple[float, float, str]],
    frame_rate: float,
):
    duration = max([e for _, e, _ in intervals] + list(ends) + [1.0])
    n_frames = max(1, int(math.ceil(duration * frame_rate)))
    true = np.full(n_frames, LABEL_TO_ID["silence"], dtype=np.int64)
    pred = np.full(n_frames, LABEL_TO_ID["silence"], dtype=np.int64)
    for start, end, lab in intervals:
        l = max(0, min(n_frames, int(start * frame_rate)))
        r = max(l + 1, min(n_frames, int(math.ceil(end * frame_rate))))
        true[l:r] = LABEL_TO_ID[normalize_label(lab)]
    for lab, start, end in zip(pred_labels, starts, ends):
        l = max(0, min(n_frames, int(start * frame_rate)))
        r = max(l + 1, min(n_frames, int(math.ceil(end * frame_rate))))
        pred[l:r] = int(lab)
    pred_boundary = np.zeros(n_frames, dtype=np.float32)
    for score, start in zip(boundary_scores, starts):
        idx = max(0, min(n_frames - 1, int(round(start * frame_rate))))
        pred_boundary[idx] = max(pred_boundary[idx], float(score))
    for i in range(1, len(pred_labels)):
        if pred_labels[i] != pred_labels[i - 1]:
            idx = max(0, min(n_frames - 1, int(round(starts[i] * frame_rate))))
            pred_boundary[idx] = max(pred_boundary[idx], 0.75)
    true_boundaries = [start for start, _, _ in intervals[1:]]
    return true, pred, pred_boundary, true_boundaries


def hit_rate(pred_scores: np.ndarray, true_boundaries: List[float], frame_rate: float, tolerance_sec: float) -> float:
    if not true_boundaries:
        return 0.0
    k = max(1, len(true_boundaries))
    candidate_idx = np.argsort(-pred_scores)[:k]
    pred_times = sorted(float(i) / frame_rate for i in candidate_idx if pred_scores[i] > 0)
    if not pred_times:
        return 0.0
    hits = 0
    used = set()
    for tb in true_boundaries:
        best = None
        best_dist = None
        for j, pt in enumerate(pred_times):
            if j in used:
                continue
            dist = abs(pt - tb)
            if dist <= tolerance_sec and (best_dist is None or dist < best_dist):
                best = j
                best_dist = dist
        if best is not None:
            used.add(best)
            hits += 1
    return hits / len(true_boundaries)


@torch.no_grad()
def evaluate(model, loader, device, cfg: Dict, output_csv: Optional[Path] = None) -> Dict[str, float]:
    model.eval()
    frame_rate = float(cfg["eval"]["frame_rate"])
    all_true, all_pred = [], []
    hrs05, hrs3, line_true, line_pred = [], [], [], []
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        global_attention_mask = batch["global_attention_mask"].to(device)
        line_positions = batch["line_positions"].to(device)
        logits, boundary_logits = model(input_ids, attention_mask, global_attention_mask, line_positions)
        probs = torch.sigmoid(boundary_logits).cpu().numpy()
        preds = logits.argmax(-1).cpu().numpy()
        labels = batch["line_labels"].numpy()
        mask = batch["line_mask"].numpy()
        for i in range(len(batch["song_id"])):
            n = int(mask[i].sum())
            p = preds[i, :n]
            y = labels[i, :n]
            b = probs[i, :n]
            starts = batch["line_starts"][i, :n].numpy()
            ends = batch["line_ends"][i, :n].numpy()
            true_frames, pred_frames, pred_boundary, true_boundaries = frame_projection(
                p, b, starts, ends, batch["intervals"][i], frame_rate
            )
            all_true.append(true_frames)
            all_pred.append(pred_frames)
            line_true.extend(y.tolist())
            line_pred.extend(p.tolist())
            hr05 = hit_rate(pred_boundary, true_boundaries, frame_rate, 0.5)
            hr3 = hit_rate(pred_boundary, true_boundaries, frame_rate, 3.0)
            hrs05.append(hr05)
            hrs3.append(hr3)
            rows.append(
                {
                    "song_id": batch["song_id"][i],
                    "source": batch["source"][i],
                    "num_lines": n,
                    "line_acc": float((p == y).mean()) if n else 0.0,
                    "hr_0p5": hr05,
                    "hr_3": hr3,
                }
            )
    y_frame = np.concatenate(all_true) if all_true else np.array([])
    p_frame = np.concatenate(all_pred) if all_pred else np.array([])
    metrics = {
        "frame_acc": float((y_frame == p_frame).mean()) if y_frame.size else 0.0,
        "line_acc": float((np.array(line_true) == np.array(line_pred)).mean()) if line_true else 0.0,
        "line_macro_f1": float(f1_score(line_true, line_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
        if line_true
        else 0.0,
        "hr_0p5": float(np.mean(hrs05)) if hrs05 else 0.0,
        "hr_3": float(np.mean(hrs3)) if hrs3 else 0.0,
        "num_songs": len(rows),
    }
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["song_id"])
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def train(cfg: Dict):
    seed = int(cfg["train"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["train"].get("cpu", False) else "cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_path"], use_fast=True)
    line_token = cfg["model"].get("line_token", "<LINE>")
    cfg["model"]["line_token"] = line_token
    tokenizer.add_special_tokens({"additional_special_tokens": [line_token]})

    train_examples = load_examples(cfg, "train")
    val_examples = load_examples(cfg, "val")
    stats = {"train_songs": len(train_examples), "val_songs": len(val_examples), "labels": LABELS}
    (output_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))

    train_ds = LyricsLineDataset(train_examples, tokenizer, {**cfg["data"], **cfg["model"]})
    val_ds = LyricsLineDataset(val_examples, tokenizer, {**cfg["data"], **cfg["model"]})
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate,
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["eval"].get("batch_size", 1)), shuffle=False, collate_fn=collate)

    model = LongformerLineClassifier(
        cfg["model"]["pretrained_model_path"], len(tokenizer), len(LABELS), float(cfg["model"].get("dropout", 0.1))
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    total_steps = int(cfg["train"].get("max_steps", len(train_loader) * int(cfg["train"]["epochs"])))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["train"].get("warmup_steps", 0)), total_steps)

    log_path = output_dir / "training_log.csv"
    best_score = -1.0
    global_step = 0
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "loss",
                "loss_function",
                "loss_boundary",
                "val_frame_acc",
                "val_line_acc",
                "val_line_macro_f1",
                "val_hr_0p5",
                "val_hr_3",
            ],
        )
        writer.writeheader()
        for epoch in range(int(cfg["train"]["epochs"])):
            model.train()
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            for batch in pbar:
                logits, boundary_logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["global_attention_mask"].to(device),
                    batch["line_positions"].to(device),
                )
                line_labels = batch["line_labels"].to(device)
                line_mask = batch["line_mask"].to(device)
                boundary_labels = batch["boundary_labels"].to(device)
                loss_function = F.cross_entropy(logits.view(-1, len(LABELS)), line_labels.view(-1), ignore_index=-100)
                bce = F.binary_cross_entropy_with_logits(
                    boundary_logits[line_mask], boundary_labels[line_mask], reduction="mean"
                )
                loss = float(cfg["train"].get("function_loss_weight", 1.0)) * loss_function + float(
                    cfg["train"].get("boundary_loss_weight", 0.2)
                ) * bce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("max_grad_norm", 1.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(loss=float(loss.detach().cpu()))

                should_eval = global_step % int(cfg["eval"]["eval_steps"]) == 0 or global_step >= total_steps
                if should_eval:
                    metrics = evaluate(model, val_loader, device, cfg, output_dir / f"val_predictions_step_{global_step}.csv")
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(loss.detach().cpu()),
                        "loss_function": float(loss_function.detach().cpu()),
                        "loss_boundary": float(bce.detach().cpu()),
                        "val_frame_acc": metrics["frame_acc"],
                        "val_line_acc": metrics["line_acc"],
                        "val_line_macro_f1": metrics["line_macro_f1"],
                        "val_hr_0p5": metrics["hr_0p5"],
                        "val_hr_3": metrics["hr_3"],
                    }
                    writer.writerow(row)
                    f.flush()
                    score = metrics["line_macro_f1"] + metrics["hr_3"]
                    if score > best_score:
                        best_score = score
                        torch.save({"model": model.state_dict(), "cfg": cfg, "metrics": metrics}, output_dir / "best.pt")
                    model.train()
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "label_map.json").write_text(json.dumps({"labels": LABELS}, indent=2))


def run_eval(cfg: Dict, ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["train"].get("cpu", False) else "cpu")
    output_dir = Path(cfg["output_dir"])
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.exists() else cfg["model"]["pretrained_model_path"]))
    cfg["model"]["line_token"] = cfg["model"].get("line_token", "<LINE>")
    if cfg["model"]["line_token"] not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [cfg["model"]["line_token"]]})
    examples = load_examples(cfg, "val")
    ds = LyricsLineDataset(examples, tokenizer, {**cfg["data"], **cfg["model"]})
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("batch_size", 1)), shuffle=False, collate_fn=collate)
    model = LongformerLineClassifier(
        cfg["model"]["pretrained_model_path"], len(tokenizer), len(LABELS), float(cfg["model"].get("dropout", 0.1))
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    metrics = evaluate(model, loader, device, cfg, output_dir / "eval_predictions.csv")
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


@torch.no_grad()
def infer_bench(cfg: Dict, ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["train"].get("cpu", False) else "cpu")
    output_dir = Path(cfg["output_dir"])
    pred_dir = Path(cfg["bench"]["pred_json_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.exists() else cfg["model"]["pretrained_model_path"]))
    cfg["model"]["line_token"] = cfg["model"].get("line_token", "<LINE>")
    if cfg["model"]["line_token"] not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [cfg["model"]["line_token"]]})

    examples = load_bench_examples(cfg)
    dataset = LyricsLineDataset(examples, tokenizer, {**cfg["data"], **cfg["model"]})
    loader = DataLoader(dataset, batch_size=int(cfg["eval"].get("batch_size", 1)), shuffle=False, collate_fn=collate)

    model = LongformerLineClassifier(
        cfg["model"]["pretrained_model_path"], len(tokenizer), len(LABELS), float(cfg["model"].get("dropout", 0.1))
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    line_rows = []
    song_rows = []
    y_true_all, y_pred_all = [], []
    missing_ids = set(read_scp_ids(cfg["bench"]["scp_path"])) - {x.song_id for x in examples}
    bench_label_map = load_bench_labels(cfg["bench"]["ann_dir"])

    for batch in tqdm(loader, desc="bench infer"):
        logits, boundary_logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["global_attention_mask"].to(device),
            batch["line_positions"].to(device),
        )
        preds = logits.argmax(-1).cpu().numpy()
        boundary_scores = torch.sigmoid(boundary_logits).cpu().numpy()
        labels = batch["line_labels"].numpy()
        mask = batch["line_mask"].numpy()
        for i, song_id in enumerate(batch["song_id"]):
            n = int(mask[i].sum())
            pred = preds[i, :n]
            gold = labels[i, :n]
            starts = batch["line_starts"][i, :n].numpy()
            ends = batch["line_ends"][i, :n].numpy()
            intervals = batch["intervals"][i]
            duration = intervals_duration(intervals, float(ends[-1]) if n else 1.0)
            segments = line_predictions_to_segments(
                pred,
                starts,
                ends,
                duration,
                min_segment_dur=float(cfg["bench"].get("min_segment_dur", 0.25)),
            )
            (pred_dir / f"{song_id}.json").write_text(json.dumps(segments, indent=2))

            y_true_all.extend(gold.tolist())
            y_pred_all.extend(pred.tolist())
            song_rows.append(
                {
                    "song_id": song_id,
                    "num_lines": n,
                    "line_acc": float((pred == gold).mean()) if n else 0.0,
                    "duration": duration,
                }
            )
            for j in range(n):
                line_rows.append(
                    {
                        "song_id": song_id,
                        "line_idx": j,
                        "start": float(starts[j]),
                        "end": float(ends[j]),
                        "text": batch["texts"][i][j],
                        "gold_label": ID_TO_LABEL[int(gold[j])],
                        "pred_label": ID_TO_LABEL[int(pred[j])],
                        "boundary_score": float(boundary_scores[i, j]),
                    }
                )

    for song_id in sorted(missing_ids):
        duration = intervals_duration(bench_label_map.get(song_id, []), 1.0)
        fallback = [{"start": 0.0, "end": float(duration), "label": "silence"}]
        (pred_dir / f"{song_id}.json").write_text(json.dumps(fallback, indent=2))
        song_rows.append({"song_id": song_id, "num_lines": 0, "line_acc": 0.0, "duration": duration})

    line_pred_csv = output_dir / "bench_line_predictions.csv"
    with line_pred_csv.open("w", newline="") as f:
        fieldnames = ["song_id", "line_idx", "start", "end", "text", "gold_label", "pred_label", "boundary_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(line_rows)

    song_csv = output_dir / "bench_song_line_metrics.csv"
    with song_csv.open("w", newline="") as f:
        fieldnames = ["song_id", "num_lines", "line_acc", "duration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(song_rows)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="weighted", zero_division=0
    )
    metrics = {
        "num_bench_songs_requested": len(read_scp_ids(cfg["bench"]["scp_path"])),
        "num_bench_songs_with_lyrics_and_labels": len(examples),
        "num_missing_or_unusable_songs": len(missing_ids),
        "missing_or_unusable_song_ids": sorted(missing_ids),
        "num_lines": len(y_true_all),
        "line_acc": float((np.array(y_true_all) == np.array(y_pred_all)).mean()) if y_true_all else 0.0,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "pred_json_dir": str(pred_dir),
        "line_predictions_csv": str(line_pred_csv),
        "song_line_metrics_csv": str(song_csv),
    }
    (output_dir / "bench_ekaputra_line_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def label_counts_json(values: List[int]) -> str:
    counts = {label: 0 for label in LABELS}
    for v in values:
        counts[ID_TO_LABEL[int(v)]] += 1
    return json.dumps(counts, ensure_ascii=False)


def class_metric_row(prefix: str, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {f"{prefix}_{label}_f1": 0.0 for label in LABELS}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    row = {}
    for i, label in enumerate(LABELS):
        row[f"{prefix}_{label}_precision"] = float(precision[i])
        row[f"{prefix}_{label}_recall"] = float(recall[i])
        row[f"{prefix}_{label}_f1"] = float(f1[i])
        row[f"{prefix}_{label}_support"] = int(support[i])
    return row


@torch.no_grad()
def infer_split_metrics(cfg: Dict, ckpt_path: str, split: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["train"].get("cpu", False) else "cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.exists() else cfg["model"]["pretrained_model_path"]))
    cfg["model"]["line_token"] = cfg["model"].get("line_token", "<LINE>")
    if cfg["model"]["line_token"] not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [cfg["model"]["line_token"]]})

    examples = load_examples(cfg, split)
    dataset = LyricsLineDataset(examples, tokenizer, {**cfg["data"], **cfg["model"]})
    loader = DataLoader(dataset, batch_size=int(cfg["eval"].get("batch_size", 1)), shuffle=False, collate_fn=collate)

    model = LongformerLineClassifier(
        cfg["model"]["pretrained_model_path"], len(tokenizer), len(LABELS), float(cfg["model"].get("dropout", 0.1))
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    song_rows = []
    line_rows = []
    y_true_all, y_pred_all = [], []

    for batch in tqdm(loader, desc=f"{split} split metrics"):
        logits, boundary_logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["global_attention_mask"].to(device),
            batch["line_positions"].to(device),
        )
        preds = logits.argmax(-1).cpu().numpy()
        boundary_scores = torch.sigmoid(boundary_logits).cpu().numpy()
        labels = batch["line_labels"].numpy()
        mask = batch["line_mask"].numpy()

        for i, song_id in enumerate(batch["song_id"]):
            n = int(mask[i].sum())
            pred = preds[i, :n].astype(int)
            gold = labels[i, :n].astype(int)
            starts = batch["line_starts"][i, :n].numpy()
            ends = batch["line_ends"][i, :n].numpy()
            y_true = gold.tolist()
            y_pred = pred.tolist()
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            if n:
                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )
                weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                line_acc = float((pred == gold).mean())
                pred_silence_ratio = float((pred == LABEL_TO_ID["silence"]).mean())
                gold_silence_ratio = float((gold == LABEL_TO_ID["silence"]).mean())
            else:
                macro_p = macro_r = macro_f1 = weighted_p = weighted_r = weighted_f1 = 0.0
                line_acc = pred_silence_ratio = gold_silence_ratio = 0.0

            row = {
                "song_id": song_id,
                "source": batch["source"][i],
                "num_lines": n,
                "duration": intervals_duration(batch["intervals"][i], float(ends[-1]) if n else 1.0),
                "line_acc": line_acc,
                "macro_precision": float(macro_p),
                "macro_recall": float(macro_r),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_p),
                "weighted_recall": float(weighted_r),
                "weighted_f1": float(weighted_f1),
                "pred_silence_ratio": pred_silence_ratio,
                "gold_silence_ratio": gold_silence_ratio,
                "gold_label_counts": label_counts_json(y_true),
                "pred_label_counts": label_counts_json(y_pred),
            }
            row.update(class_metric_row("class", y_true, y_pred))
            song_rows.append(row)

            for j in range(n):
                line_rows.append(
                    {
                        "song_id": song_id,
                        "source": batch["source"][i],
                        "line_idx": j,
                        "start": float(starts[j]),
                        "end": float(ends[j]),
                        "text": batch["texts"][i][j],
                        "gold_label": ID_TO_LABEL[int(gold[j])],
                        "pred_label": ID_TO_LABEL[int(pred[j])],
                        "boundary_score": float(boundary_scores[i, j]),
                    }
                )

    song_csv = output_dir / f"{split}_song_line_metrics.csv"
    line_csv = output_dir / f"{split}_line_predictions.csv"
    summary_json = output_dir / f"{split}_ekaputra_line_metrics.json"

    if song_rows:
        with song_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(song_rows[0].keys()))
            writer.writeheader()
            writer.writerows(song_rows)
    if line_rows:
        with line_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(line_rows[0].keys()))
            writer.writeheader()
            writer.writerows(line_rows)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=list(range(len(LABELS))), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="weighted", zero_division=0
    )
    summary = {
        "split": split,
        "num_songs": len(song_rows),
        "num_lines": len(y_true_all),
        "line_acc": float((np.array(y_true_all) == np.array(y_pred_all)).mean()) if y_true_all else 0.0,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "song_metrics_csv": str(song_csv),
        "line_predictions_csv": str(line_csv),
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def write_msa_txt(path: Path, intervals: List[Tuple[float, float, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for start, _end, label in sorted(intervals, key=lambda x: (x[0], x[1])):
        rows.append(f"{float(start):.6f} {normalize_label(label)}")
    duration = intervals_duration(intervals, 1.0)
    rows.append(f"{duration:.6f} end")
    path.write_text("\n".join(rows) + "\n")


@torch.no_grad()
def infer_split_segments(cfg: Dict, ckpt_path: str, split: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["train"].get("cpu", False) else "cpu")
    output_dir = Path(cfg["output_dir"])
    pred_dir = output_dir / f"{split}_pred" / "json"
    ann_txt_dir = output_dir / "eval" / split / "ann_txt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ann_txt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.exists() else cfg["model"]["pretrained_model_path"]))
    cfg["model"]["line_token"] = cfg["model"].get("line_token", "<LINE>")
    if cfg["model"]["line_token"] not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [cfg["model"]["line_token"]]})

    examples = load_examples(cfg, split)
    dataset = LyricsLineDataset(examples, tokenizer, {**cfg["data"], **cfg["model"]})
    loader = DataLoader(dataset, batch_size=int(cfg["eval"].get("batch_size", 1)), shuffle=False, collate_fn=collate)

    model = LongformerLineClassifier(
        cfg["model"]["pretrained_model_path"], len(tokenizer), len(LABELS), float(cfg["model"].get("dropout", 0.1))
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    for batch in tqdm(loader, desc=f"{split} split segment infer"):
        logits, _boundary_logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["global_attention_mask"].to(device),
            batch["line_positions"].to(device),
        )
        preds = logits.argmax(-1).cpu().numpy()
        mask = batch["line_mask"].numpy()
        labels = batch["line_labels"].numpy()

        for i, song_id in enumerate(batch["song_id"]):
            n = int(mask[i].sum())
            pred = preds[i, :n]
            gold = labels[i, :n]
            starts = batch["line_starts"][i, :n].numpy()
            ends = batch["line_ends"][i, :n].numpy()
            intervals = batch["intervals"][i]
            duration = intervals_duration(intervals, float(ends[-1]) if n else 1.0)
            segments = line_predictions_to_segments(
                pred,
                starts,
                ends,
                duration,
                min_segment_dur=float(cfg.get("split_eval", {}).get("min_segment_dur", 0.25)),
            )
            (pred_dir / f"{song_id}.json").write_text(json.dumps(segments, indent=2))
            write_msa_txt(ann_txt_dir / f"{song_id}.txt", intervals)
            rows.append(
                {
                    "song_id": song_id,
                    "source": batch["source"][i],
                    "num_lines": n,
                    "line_acc": float((pred == gold).mean()) if n else 0.0,
                    "duration": duration,
                    "pred_json": str(pred_dir / f"{song_id}.json"),
                    "ann_txt": str(ann_txt_dir / f"{song_id}.txt"),
                }
            )

    summary = {
        "split": split,
        "num_songs": len(rows),
        "pred_json_dir": str(pred_dir),
        "ann_txt_dir": str(ann_txt_dir),
        "est_txt_dir": str(output_dir / "eval" / split / "est_txt"),
        "metrics_dir": str(output_dir / "eval" / split / "metrics"),
        "song_manifest_csv": str(output_dir / f"{split}_segment_infer_manifest.csv"),
    }
    manifest_path = output_dir / f"{split}_segment_infer_manifest.csv"
    if rows:
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / f"{split}_segment_infer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "infer_bench", "infer_split", "infer_split_segments"],
        default="train",
    )
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        ckpt = args.ckpt or str(Path(cfg["output_dir"]) / "best.pt")
        run_eval(cfg, ckpt)
    else:
        ckpt = args.ckpt or str(Path(cfg["output_dir"]) / "best.pt")
        if args.mode == "infer_bench":
            infer_bench(cfg, ckpt)
        elif args.mode == "infer_split_segments":
            infer_split_segments(cfg, ckpt, args.split)
        else:
            infer_split_metrics(cfg, ckpt, args.split)


if __name__ == "__main__":
    main()
