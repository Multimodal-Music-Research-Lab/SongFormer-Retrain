import json
from pathlib import Path

SRC = Path("/mnt/ssd/hbli/datasets/songformer/songformbench/data/SongFormBench.jsonl")
OUT = Path("/mnt/ssd/hbli/datasets/songformer/songformbench/data/SongFormBench_for_model.jsonl")

def convert_labels(labels):
    # [{"start":..., "label":...}, ...]  ->  [[start, label], ...]
    out = []
    for x in labels:
        if isinstance(x, dict):
            out.append([float(x["start"]), x["label"]])
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            out.append([float(x[0]), x[1]])
        else:
            raise ValueError(f"Unknown label item format: {x}")
    return out

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with SRC.open("r", encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            obj["labels"] = convert_labels(obj["labels"])
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} lines to: {OUT}")

if __name__ == "__main__":
    main()