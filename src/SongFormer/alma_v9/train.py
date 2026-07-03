from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import AlmaV9Dataset, alma_v9_collate, build_hook_items, build_hx_items
from .model import AlmaV9Model, AlmaV9ModelConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def build_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.lyrics_tokenizer_dir, use_fast=True)
    line_token = cfg.model.line_token
    if line_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([line_token])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def build_dataset(cfg, split: str, tokenizer):
    if split == "train":
        items = build_hx_items(cfg.data.train_hx) + build_hook_items(cfg.data.train_hook)
        train = True
    elif split == "val":
        items = build_hx_items(cfg.data.val_hx)
        train = False
    else:
        raise ValueError(split)
    return AlmaV9Dataset(
        items=items,
        tokenizer=tokenizer,
        feature_dir=cfg.data.mert_feature_dir,
        frame_hz=float(cfg.mert.frame_hz),
        slice_dur=float(cfg.train.slice_dur),
        train=train,
        max_text_len=int(cfg.model.lyrics_max_length),
        line_max_len=int(cfg.model.lyrics_line_max_length),
        boundary_widen_sec=float(cfg.loss.boundary_widen_sec),
        line_token=cfg.model.line_token,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 0):
    model.eval()
    totals = {}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(batch)
        losses = model.compute_loss(outputs, batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
        if max_batches and count >= max_batches:
            break
    model.train()
    if count == 0:
        return {key: 0.0 for key in ["loss", "loss_boundary", "loss_function"]}
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, step, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "global_step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.train.run_dir)
    ckpt_dir = run_dir / "results" / f"train_output_{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg)
    train_set = build_dataset(cfg, "train", tokenizer)
    val_set = build_dataset(cfg, "val", tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        collate_fn=alma_v9_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=alma_v9_collate,
    )

    model_cfg = AlmaV9ModelConfig(
        audio_dim=train_set.audio_dim,
        vocab_size=len(tokenizer),
        num_classes=int(cfg.model.num_classes),
        d_model=int(cfg.model.d_model),
        lyrics_layers=int(cfg.model.lyrics_layers),
        lyrics_heads=int(cfg.model.lyrics_heads),
        lyrics_ffn_dim=int(cfg.model.lyrics_ffn_dim),
        fusion_layers=int(cfg.model.fusion_layers),
        fusion_block_type=str(cfg.model.fusion_block_type),
        dropout=float(cfg.model.dropout),
        frame_hz=float(cfg.mert.frame_hz),
        loss_weight_boundary=float(cfg.loss.weight_boundary),
        loss_weight_function=float(cfg.loss.weight_function),
        focal_weight=float(cfg.loss.focal_weight),
        focal_alpha=float(cfg.loss.focal_alpha),
        focal_gamma=float(cfg.loss.focal_gamma),
    )
    model = AlmaV9Model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        betas=tuple(cfg.train.betas),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.train.warmup_steps),
        num_training_steps=int(cfg.train.max_steps),
    )

    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = int(ckpt.get("global_step", 0))

    log_path = ckpt_dir / "training_loss.csv"
    write_header = not log_path.exists() or global_step == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "loss_boundary",
                "loss_function",
                "lyric_frame_coverage",
                "valid_loss",
                "valid_loss_boundary",
                "valid_loss_function",
            ],
        )
        if write_header:
            writer.writeheader()

        pbar = tqdm(total=int(cfg.train.max_steps), initial=global_step, desc="train_v9")
        model.train()
        while global_step < int(cfg.train.max_steps):
            for batch in train_loader:
                batch = move_to_device(batch, device)
                outputs = model(batch)
                losses = model.compute_loss(outputs, batch)
                loss = losses["loss"] / int(cfg.train.accumulation_steps)
                loss.backward()
                if (global_step + 1) % int(cfg.train.accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                row = {
                    "step": global_step,
                    "loss": float(losses["loss"].detach().cpu()),
                    "loss_boundary": float(losses["loss_boundary"].detach().cpu()),
                    "loss_function": float(losses["loss_function"].detach().cpu()),
                    "lyric_frame_coverage": float(losses["lyric_frame_coverage"].detach().cpu()),
                    "valid_loss": "",
                    "valid_loss_boundary": "",
                    "valid_loss_function": "",
                }
                if global_step % int(cfg.train.log_interval) == 0:
                    writer.writerow(row)
                    f.flush()
                if global_step % int(cfg.train.eval_interval) == 0:
                    metrics = evaluate(model, val_loader, device, max_batches=int(cfg.train.eval_max_batches))
                    row.update(
                        {
                            "valid_loss": metrics.get("loss", 0.0),
                            "valid_loss_boundary": metrics.get("loss_boundary", 0.0),
                            "valid_loss_function": metrics.get("loss_function", 0.0),
                        }
                    )
                    writer.writerow(row)
                    f.flush()
                if global_step % int(cfg.train.save_interval) == 0:
                    save_checkpoint(ckpt_dir / f"model.ckpt-{global_step}.pt", model, optimizer, scheduler, global_step, cfg)
                    (ckpt_dir / "checkpoint").write_text(f"model.ckpt-{global_step}.pt", encoding="utf-8")
                pbar.update(1)
                if global_step >= int(cfg.train.max_steps):
                    break
        pbar.close()
    save_checkpoint(ckpt_dir / f"model.ckpt-{global_step}.pt", model, optimizer, scheduler, global_step, cfg)


if __name__ == "__main__":
    main()

