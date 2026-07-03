from __future__ import annotations

LABELS = [
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
    "pre-chorus",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
IGNORE_INDEX = -100


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    mapping = {
        "prechorus": "pre-chorus",
        "pre chorus": "pre-chorus",
        "instrumental": "inst",
        "no_label": "NO_LABEL",
        "nolabel": "NO_LABEL",
        "end": "end",
    }
    return mapping.get(label, label)


def label_to_id(label: str) -> int:
    label = normalize_label(label)
    if label not in LABEL_TO_ID:
        raise KeyError(f"Unknown label for v9 8-class setup: {label}")
    return LABEL_TO_ID[label]


def id_to_label(idx: int) -> str:
    return ID_TO_LABEL[int(idx)]

