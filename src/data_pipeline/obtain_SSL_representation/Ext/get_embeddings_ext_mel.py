"""Extract MuQ or MusicFM SSL features from SongFormDB Ext mel files.

SongFormDB Ext stores BigVGAN-compatible 44.1 kHz, 128-band log-mel
spectrograms rather than waveforms. This script reconstructs each waveform
in memory with the matching BigVGAN model, resamples it to 24 kHz, and then
uses the same 30 s / 420 s extraction scheme as the historical SongFormer
pipeline. No source mel or intermediate waveform file is modified or saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


SOURCE_SAMPLE_RATE = 44100
SOURCE_MEL_BANDS = 128
SOURCE_HOP_SIZE = 256
SSL_SAMPLE_RATE = 24000
SSL_DIM = 1024
SSL_LAYER = 10
WRAP_SECONDS = 420

DEFAULT_BIGVGAN_REPO = "/home/hbli/songformer/software/BigVGAN"
DEFAULT_BIGVGAN_MODEL = (
    "/home/hbli/songformer/software/bigvgan_v2_44khz_128band_256x"
)
DEFAULT_THIRD_PARTY = "/home/hbli/songformer/repo/SongFormer/src/third_party"
DEFAULT_MUSICFM_CKPT_DIR = (
    "/home/hbli/songformer/repo/SongFormer/src/SongFormer/ckpts/MusicFM"
)
DEFAULT_HF_HOME = "/home/hbli/songformer/cache/hf_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct SongFormDB Ext log-mels with BigVGAN and extract "
            "MuQ/MusicFM layer-10 SSL features."
        )
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Directory containing Ext .npy files, or a text/SCP file of paths.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output root; embeddings are written under layer_10/.",
    )
    parser.add_argument("--encoder", required=True, choices=("muq", "musicfm"))
    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
        choices=(30, 420),
        help="SSL inference window in seconds.",
    )
    parser.add_argument(
        "-gn",
        "--gpu_num",
        type=int,
        default=1,
        help="Number of visible GPUs to use.",
    )
    parser.add_argument(
        "-tn",
        "--num_thread_per_gpu",
        type=int,
        default=1,
        help="Worker processes per GPU. Keep at 1 unless VRAM permits model copies.",
    )
    parser.add_argument(
        "--bigvgan_repo",
        default=DEFAULT_BIGVGAN_REPO,
        help="Local NVIDIA BigVGAN repository.",
    )
    parser.add_argument(
        "--bigvgan_model",
        default=DEFAULT_BIGVGAN_MODEL,
        help="Local bigvgan_v2_44khz_128band_256x model directory.",
    )
    parser.add_argument(
        "--third_party_dir",
        default=DEFAULT_THIRD_PARTY,
        help="SongFormer third_party directory containing MusicFM.",
    )
    parser.add_argument(
        "--musicfm_ckpt_dir",
        default=DEFAULT_MUSICFM_CKPT_DIR,
        help="Directory containing msd_stats.json and pretrained_msd.pt.",
    )
    parser.add_argument(
        "--muq_model",
        default="OpenMuQ/MuQ-large-msd-iter",
        help="MuQ model ID or local model path.",
    )
    parser.add_argument(
        "--hf_home",
        default=DEFAULT_HF_HOME,
        help="Hugging Face cache root used to locate the existing MuQ weights.",
    )
    parser.add_argument(
        "--allow_hf_download",
        action="store_true",
        help="Allow MuQ to download missing files; default behavior is local-only.",
    )
    parser.add_argument(
        "--vocoder_chunk_seconds",
        type=float,
        default=30.0,
        help="BigVGAN target chunk length; lower this if vocoder inference OOMs.",
    )
    parser.add_argument(
        "--vocoder_context_seconds",
        type=float,
        default=0.5,
        help="Left/right mel context reconstructed and cropped at chunk edges.",
    )
    parser.add_argument(
        "--failure_csv",
        default=None,
        help="Failure manifest path (default: OUTPUT/extract_failures.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract songs even when every expected 420 s output exists.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Process at most this many pending files (for validation/debugging).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.gpu_num < 1 or args.num_thread_per_gpu < 1:
        raise ValueError("gpu_num and num_thread_per_gpu must both be positive")
    if args.vocoder_chunk_seconds <= 0:
        raise ValueError("vocoder_chunk_seconds must be positive")
    if args.vocoder_context_seconds < 0:
        raise ValueError("vocoder_context_seconds cannot be negative")
    for required_path in (
        args.input_path,
        args.bigvgan_repo,
        args.bigvgan_model,
    ):
        if not Path(required_path).exists():
            raise FileNotFoundError(required_path)
    if args.encoder == "musicfm":
        for name in ("msd_stats.json", "pretrained_msd.pt"):
            path = Path(args.musicfm_ckpt_dir) / name
            if not path.is_file():
                raise FileNotFoundError(path)


def read_inputs(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        candidates = sorted(path.rglob("*.npy"))
    elif path.suffix.lower() == ".npy":
        candidates = [path]
    else:
        candidates = []
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                full_line_path = Path(line)
                if full_line_path.is_file():
                    candidates.append(full_line_path)
                    continue
                # Also accept conventional "utt_id /absolute/path.npy" SCP lines.
                fields = line.split(maxsplit=1)
                candidate = Path(fields[-1])
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"Input line does not resolve to a file: {line}"
                    )
                candidates.append(candidate)

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.suffix.lower() != ".npy":
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise RuntimeError(f"No .npy inputs found under {input_path}")
    return unique


def mel_shape(path: Path) -> tuple[int, int]:
    mel = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = mel.shape
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 2 or shape[0] != SOURCE_MEL_BANDS:
        raise ValueError(
            f"{path}: expected ({SOURCE_MEL_BANDS}, T) mel, found {mel.shape}"
        )
    return int(shape[0]), int(shape[1])


def expected_output_paths(source: Path, output_layer_dir: Path) -> list[Path]:
    _, frames = mel_shape(source)
    total_samples = frames * SOURCE_HOP_SIZE
    samples_per_wrap = WRAP_SECONDS * SOURCE_SAMPLE_RATE
    num_wraps = max(1, math.ceil(total_samples / samples_per_wrap))
    return [
        output_layer_dir / f"{source.stem}_{index * WRAP_SECONDS}.npy"
        for index in range(num_wraps)
    ]


def is_complete(source: Path, output_layer_dir: Path) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in expected_output_paths(source, output_layer_dir)
    )


def load_bigvgan(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, args.bigvgan_repo)
    import bigvgan  # pylint: disable=import-outside-toplevel

    model = bigvgan.BigVGAN.from_pretrained(
        args.bigvgan_model,
        use_cuda_kernel=False,
        local_files_only=True,
    )
    model.remove_weight_norm()
    return model.to(device).eval()


def load_ssl_encoder(
    args: argparse.Namespace, device: torch.device
) -> torch.nn.Module:
    if args.encoder == "muq":
        from muq import MuQ  # pylint: disable=import-outside-toplevel

        model = MuQ.from_pretrained(
            args.muq_model,
            cache_dir=str(Path(args.hf_home) / "hub"),
            local_files_only=not args.allow_hf_download,
        )
        return model.to(device).eval()

    sys.path.insert(0, args.third_party_dir)
    from musicfm.model.musicfm_25hz import (  # pylint: disable=import-outside-toplevel
        MusicFM25Hz,
    )

    model = MusicFM25Hz(
        is_flash=False,
        stat_path=str(Path(args.musicfm_ckpt_dir) / "msd_stats.json"),
        model_path=str(Path(args.musicfm_ckpt_dir) / "pretrained_msd.pt"),
    )
    return model.to(device).eval()


def load_mel(path: Path) -> np.ndarray:
    mel = np.load(path, allow_pickle=False)
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]
    if mel.ndim != 2 or mel.shape[0] != SOURCE_MEL_BANDS:
        raise ValueError(
            f"{path}: expected ({SOURCE_MEL_BANDS}, T) mel, found {mel.shape}"
        )
    if not np.issubdtype(mel.dtype, np.floating):
        raise TypeError(f"{path}: expected floating-point mel, found {mel.dtype}")
    if not np.isfinite(mel).all():
        raise ValueError(f"{path}: mel contains NaN or Inf")
    return np.asarray(mel, dtype=np.float32, order="C")


@torch.inference_mode()
def reconstruct_waveform(
    mel: np.ndarray,
    vocoder: torch.nn.Module,
    device: torch.device,
    chunk_seconds: float,
    context_seconds: float,
) -> torch.Tensor:
    frames_per_second = SOURCE_SAMPLE_RATE / SOURCE_HOP_SIZE
    target_frames = max(1, round(chunk_seconds * frames_per_second))
    context_frames = max(0, round(context_seconds * frames_per_second))
    waveform_parts = []

    for target_start in range(0, mel.shape[1], target_frames):
        target_end = min(target_start + target_frames, mel.shape[1])
        context_start = max(0, target_start - context_frames)
        context_end = min(mel.shape[1], target_end + context_frames)
        mel_chunk = torch.from_numpy(
            np.ascontiguousarray(mel[:, context_start:context_end])
        )
        mel_chunk = mel_chunk.unsqueeze(0).to(device, non_blocking=True)
        generated = vocoder(mel_chunk).squeeze(0).squeeze(0)

        crop_start = (target_start - context_start) * SOURCE_HOP_SIZE
        crop_length = (target_end - target_start) * SOURCE_HOP_SIZE
        generated = generated[crop_start : crop_start + crop_length]
        if generated.numel() != crop_length:
            raise RuntimeError(
                f"BigVGAN returned {generated.numel()} cropped samples; "
                f"expected {crop_length}"
            )
        waveform_parts.append(generated.float().cpu())
        del mel_chunk, generated
        torch.cuda.empty_cache()

    waveform = torch.cat(waveform_parts)
    expected_samples = mel.shape[1] * SOURCE_HOP_SIZE
    if waveform.numel() != expected_samples:
        raise RuntimeError(
            f"Reconstructed waveform has {waveform.numel()} samples; "
            f"expected {expected_samples}"
        )
    return waveform


def resample_for_ssl(waveform: torch.Tensor) -> torch.Tensor:
    waveform = torchaudio.functional.resample(
        waveform.unsqueeze(0),
        orig_freq=SOURCE_SAMPLE_RATE,
        new_freq=SSL_SAMPLE_RATE,
    )
    return waveform.squeeze(0).contiguous()


@torch.inference_mode()
def encode_segment(
    segment: torch.Tensor,
    encoder: torch.nn.Module,
    encoder_name: str,
    device: torch.device,
) -> np.ndarray:
    segment = segment.unsqueeze(0).to(device, non_blocking=True)
    if encoder_name == "muq":
        output = encoder(segment, output_hidden_states=True)
        hidden = output["hidden_states"][SSL_LAYER]
    else:
        _, hidden_states = encoder.get_predictions(segment)
        hidden = hidden_states[SSL_LAYER]
    array = hidden.detach().cpu().float().numpy()
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != SSL_DIM:
        raise RuntimeError(
            f"Unexpected {encoder_name} layer-{SSL_LAYER} shape: {array.shape}"
        )
    return array


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def extract_ssl(
    source: Path,
    waveform: torch.Tensor,
    encoder: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_layer_dir = Path(args.output_path) / f"layer_{SSL_LAYER}"
    total_samples = waveform.numel()
    wrap_samples = WRAP_SECONDS * SSL_SAMPLE_RATE
    window_samples = args.window_size * SSL_SAMPLE_RATE

    for wrap_start_seconds in range(0, 10000, WRAP_SECONDS):
        wrap_start = wrap_start_seconds * SSL_SAMPLE_RATE
        if wrap_start >= total_samples:
            break
        wrap_end = min(wrap_start + wrap_samples, total_samples)

        embeddings = []
        if args.window_size == WRAP_SECONDS:
            segment = waveform[wrap_start:wrap_end]
            if segment.numel() >= 1025:
                embeddings.append(
                    encode_segment(segment, encoder, args.encoder, device)
                )
        else:
            for local_start_seconds in range(0, WRAP_SECONDS, args.window_size):
                segment_start = (
                    wrap_start + local_start_seconds * SSL_SAMPLE_RATE
                )
                if segment_start >= wrap_end:
                    break
                segment_end = min(segment_start + window_samples, wrap_end)
                segment = waveform[segment_start:segment_end]
                if segment.numel() < 1025:
                    break
                embeddings.append(
                    encode_segment(segment, encoder, args.encoder, device)
                )

        if not embeddings:
            raise RuntimeError(
                f"{source}: no SSL embedding produced at {wrap_start_seconds}s"
            )
        combined = np.concatenate(embeddings, axis=1)
        output_path = (
            output_layer_dir / f"{source.stem}_{wrap_start_seconds}.npy"
        )
        atomic_save(output_path, combined)
        del embeddings, combined
        torch.cuda.empty_cache()


def process_one(
    source: Path,
    vocoder: torch.nn.Module,
    encoder: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, str]:
    started = time.time()
    try:
        mel = load_mel(source)
        waveform = reconstruct_waveform(
            mel,
            vocoder,
            device,
            args.vocoder_chunk_seconds,
            args.vocoder_context_seconds,
        )
        waveform = resample_for_ssl(waveform)
        extract_ssl(source, waveform, encoder, args, device)
        return {
            "source": str(source),
            "status": "ok",
            "error": "",
            "elapsed_seconds": f"{time.time() - started:.3f}",
        }
    except Exception:
        torch.cuda.empty_cache()
        return {
            "source": str(source),
            "status": "failed",
            "error": traceback.format_exc().replace("\n", "\\n"),
            "elapsed_seconds": f"{time.time() - started:.3f}",
        }


def worker(
    rank: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    args: argparse.Namespace,
) -> None:
    device = torch.device(f"cuda:{rank}")
    try:
        torch.cuda.set_device(device)
        vocoder = load_bigvgan(args, device)
        encoder = load_ssl_encoder(args, device)
    except Exception:
        initialization_error = traceback.format_exc().replace("\n", "\\n")
        while True:
            item = input_queue.get()
            if item is None:
                return
            output_queue.put(
                {
                    "source": item,
                    "status": "failed",
                    "error": f"Worker initialization failed: {initialization_error}",
                    "elapsed_seconds": "0.000",
                }
            )

    while True:
        item = input_queue.get()
        if item is None:
            return
        result = process_one(Path(item), vocoder, encoder, args, device)
        output_queue.put(result)


def merge_failure_manifest(path: Path, results: list[dict[str, str]]) -> None:
    failures: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("source"):
                    failures[row["source"]] = row

    for result in results:
        if result["status"] == "failed":
            failures[result["source"]] = result
        else:
            failures.pop(result["source"], None)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source", "status", "error", "elapsed_seconds"),
        )
        writer.writeheader()
        writer.writerows(failures.values())
    os.replace(temporary, path)


def write_extraction_config(args: argparse.Namespace, num_inputs: int) -> None:
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload.update(
        {
            "num_inputs_discovered": num_inputs,
            "source_sample_rate": SOURCE_SAMPLE_RATE,
            "source_mel_bands": SOURCE_MEL_BANDS,
            "source_hop_size": SOURCE_HOP_SIZE,
            "ssl_sample_rate": SSL_SAMPLE_RATE,
            "ssl_layer": SSL_LAYER,
            "ssl_dim": SSL_DIM,
            "wrap_seconds": WRAP_SECONDS,
        }
    )
    path = output_path / "extraction_config.json"
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("HF_HOME", args.hf_home)
    if not args.allow_hf_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BigVGAN and SSL extraction")

    sources = read_inputs(args.input_path)
    output_layer_dir = Path(args.output_path) / f"layer_{SSL_LAYER}"
    if args.overwrite:
        all_pending = sources
    else:
        all_pending = [
            source for source in sources if not is_complete(source, output_layer_dir)
        ]
    num_complete = len(sources) - len(all_pending)
    pending = all_pending
    if args.max_items is not None:
        pending = pending[: args.max_items]

    failure_csv = Path(
        args.failure_csv or (Path(args.output_path) / "extract_failures.csv")
    )
    write_extraction_config(args, len(sources))
    print(
        f"Found {len(sources)} Ext mel files; "
        f"{num_complete} complete, {len(all_pending)} pending, "
        f"{len(pending)} scheduled in this run."
    )
    if not pending:
        print(f"Nothing to do. Failure manifest: {failure_csv}")
        return

    num_workers = args.gpu_num * args.num_thread_per_gpu
    context = mp.get_context("spawn")
    input_queue = context.Queue()
    output_queue = context.Queue()
    processes = []
    for worker_index in range(num_workers):
        rank = worker_index % args.gpu_num
        process = context.Process(
            target=worker,
            args=(rank, input_queue, output_queue, args),
            daemon=True,
        )
        process.start()
        processes.append(process)
        time.sleep(0.2)

    for source in pending:
        input_queue.put(str(source))
    for _ in processes:
        input_queue.put(None)

    results = []
    progress = tqdm(total=len(pending), desc="extract_ext_ssl")
    while len(results) < len(pending):
        try:
            result = output_queue.get(timeout=30)
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                raise RuntimeError(
                    "All extraction workers exited before returning every result"
                )
            continue
        results.append(result)
        progress.update(1)
        if result["status"] == "failed":
            tqdm.write(f"[FAILED] {result['source']}")
    progress.close()

    for process in processes:
        process.join()
    merge_failure_manifest(failure_csv, results)

    failed = sum(result["status"] == "failed" for result in results)
    print(
        f"Completed {len(results) - failed}/{len(results)} pending files; "
        f"failed={failed}. Failure manifest: {failure_csv}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
