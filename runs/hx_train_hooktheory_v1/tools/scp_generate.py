from pathlib import Path

# label filter: delete None-label and several-label records
def extract_label_from_line(line):
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 3:
        return None
    if len(parts) > 3:
        return 'several'

    label = parts[2]
    return label if label else None

def should_drop_file(fp):
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            label = extract_label_from_line(line)
            if label is None or label == "several":
                return True
    return False


audio_dir = Path("/home/z/\u4e0b\u8f7d/\u6253\u5305\u7ed3\u679c_audio/")
sec_dir = Path("/home/z/\u4e0b\u8f7d/\u6253\u5305\u7ed3\u679c_section/")
out_scp = Path("/home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/results/hooktheory_all.scp")

SEC_SUFFIX = "_measure.sec"

# all sec files
sec_files = sorted(sec_dir.rglob(f"*{SEC_SUFFIX}"))

# delete None-label and several-label records
sec_files = [fp for fp in sec_files if not should_drop_file(fp)]
                

mp3_names = [p.name.replace("_measure.sec", ".mp3") for p in sec_files]

paths = [(audio_dir / name).resolve().as_posix() for name in mp3_names]

out_scp.write_text("\n".join(paths) + "\n", encoding="utf-8")

print("Wrote:", out_scp)
print("Total audio paths:", len(paths))
print("First 3:")
for p in paths[:3]:
    print(" ", p)