from pathlib import Path

audio_dir = Path("/mnt/ssd/hbli/datasets/songformer/songformdb/HX/audios")
out_scp = Path("/home/hbli/songformer/repo/SongFormer/runs/hx_retrain_v1/results/hx_all.scp")

exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
paths = sorted([p.resolve().as_posix() for p in audio_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts])

if not paths:
    raise SystemExit(f"No audio files found in: {audio_dir}")

out_scp.write_text("\n".join(paths) + "\n", encoding="utf-8")
print("Wrote:", out_scp)
print("Total audio paths:", len(paths))
print("First 3:")
for p in paths[:3]:
    print(" ", p)