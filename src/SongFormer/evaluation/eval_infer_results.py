# monkey patch to fix issues in msaf
import scipy
import numpy as np
scipy.inf = np.inf

import argparse
import bisect
import os
from collections import defaultdict
from pathlib import Path
import mir_eval
import pandas as pd
from dataset.custom_types import MsaInfo
from dataset.label2id import LABEL_TO_ID
from dataset.msa_info_utils import load_msa_info
from msaf.eval import compute_results
from postprocessing.calc_acc import cal_acc
from postprocessing.calc_iou import cal_iou
from tqdm import tqdm
from loguru import logger

LEGAL_LABELS = {
    "end",
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
    "pre-chorus",
}


EVAL_LABELS = [
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
    "pre-chorus",
]


def to_inters_labels(msa_info: MsaInfo):
    label_ids = np.array([LABEL_TO_ID[x[1]] for x in msa_info[:-1]])
    times = [x[0] for x in msa_info]
    start_times = np.column_stack([np.array(times[:-1]), np.array(times[1:])])
    return start_times, label_ids


def duration_confusion(ann_info: MsaInfo, est_info: MsaInfo, post_digit: int = 3):
    ann_info_time = [
        int(round(time_, post_digit) * (10**post_digit)) for time_, _ in ann_info
    ]
    est_info_time = [
        int(round(time_, post_digit) * (10**post_digit)) for time_, _ in est_info
    ]

    common_start_time = max(ann_info_time[0], est_info_time[0])
    common_end_time = min(ann_info_time[-1], est_info_time[-1])

    time_points = {common_start_time, common_end_time}
    for time_ in ann_info_time:
        if common_start_time <= time_ <= common_end_time:
            time_points.add(time_)
    for time_ in est_info_time:
        if common_start_time <= time_ <= common_end_time:
            time_points.add(time_)

    confusion = defaultdict(float)
    sorted_time_points = sorted(time_points)
    for idx, start_time in enumerate(sorted_time_points[:-1]):
        end_time = sorted_time_points[idx + 1]
        duration = end_time - start_time
        ann_label = ann_info[bisect.bisect_right(ann_info_time, start_time) - 1][1]
        est_label = est_info[bisect.bisect_right(est_info_time, start_time) - 1][1]
        if ann_label == "end" or est_label == "end":
            continue
        confusion[(ann_label, est_label)] += duration / (10**post_digit)
    return confusion


def classification_metrics_from_confusion(confusion):
    labels = sorted(
        set(EVAL_LABELS)
        | {gold for gold, _ in confusion.keys()}
        | {pred for _, pred in confusion.keys()}
    )
    metrics = {}
    for label in labels:
        tp = confusion.get((label, label), 0.0)
        gold_dur = sum(v for (gold, _), v in confusion.items() if gold == label)
        pred_dur = sum(v for (_, pred), v in confusion.items() if pred == label)
        precision = tp / pred_dur if pred_dur > 0 else np.nan
        recall = tp / gold_dur if gold_dur > 0 else np.nan
        if gold_dur > 0 and pred_dur > 0 and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        elif gold_dur > 0:
            f1 = 0.0
        else:
            f1 = np.nan
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_dur": gold_dur,
            "pred_dur": pred_dur,
        }
    return metrics


def flatten_classification_metrics(metrics, prefix=""):
    flat = {}
    for label, values in metrics.items():
        clean_label = label.replace("-", "_")
        for key, value in values.items():
            flat[f"{prefix}{key}_{clean_label}"] = value
    return flat


def merge_continuous_segments(segments):
    """
    Merge continuous segments with the same label.

    Parameters:
    segments: List of tuples [(start_time, label), ...], where the last element is (end_time, 'end')

    Returns:
    Merged segment list in the same format [(start_time, label), ...], with the last element being (end_time, 'end')
    """
    if not segments or len(segments) < 2:
        return segments

    merged = []
    current_start = segments[0][0]
    current_label = segments[0][1]

    for i in range(1, len(segments)):
        time, label = segments[i]

        if label == "end":
            if current_label != "end":
                merged.append((current_start, current_label))
            merged.append((time, "end"))
            break

        if label != current_label:
            merged.append((current_start, current_label))
            current_start = time
            current_label = label

    return merged


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ann_dir", type=str, required=True)
    argparser.add_argument("--est_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, default="./eval_infer_results")
    argparser.add_argument("--prechorus2what", type=str, default=None)
    argparser.add_argument("--armerge_continuous_segments", action="store_true")
    args = argparser.parse_args()

    ann_dir = args.ann_dir
    est_dir = args.est_dir
    output_dir = args.output_dir
    if args.armerge_continuous_segments:
        logger.info("Merging continuous segments")
    os.makedirs(output_dir, exist_ok=True)

    ann_id_lists = [x for x in os.listdir(ann_dir) if x.endswith(".txt")]
    est_id_lists = [x for x in os.listdir(est_dir) if x.endswith(".txt")]

    common_id_lists = set(ann_id_lists) & set(est_id_lists)
    common_id_lists = list(common_id_lists)
    logger.info(f"Common number of files: {len(common_id_lists)}")

    resultes = []
    ious = {}
    confusion_total = defaultdict(float)

    for id in tqdm(common_id_lists):
        try:
            logger.info(f"Processing {id}")
            ann_msa = load_msa_info(os.path.join(ann_dir, id))
            est_msa = load_msa_info(os.path.join(est_dir, id))

            if args.prechorus2what == "verse":
                ann_msa = [
                    (t, "verse") if l == "pre-chorus" else (t, l) for t, l in ann_msa
                ]
                est_msa = [
                    (t, "verse") if l == "pre-chorus" else (t, l) for t, l in est_msa
                ]
            elif args.prechorus2what == "chorus":
                ann_msa = [
                    (t, "chorus") if l == "pre-chorus" else (t, l) for t, l in ann_msa
                ]
                est_msa = [
                    (t, "chorus") if l == "pre-chorus" else (t, l) for t, l in est_msa
                ]
            elif args.prechorus2what is not None:
                raise ValueError(f"Unknown prechorus2what: {args.prechorus2what}")
            if args.armerge_continuous_segments:
                ann_msa = merge_continuous_segments(ann_msa)
                est_msa = merge_continuous_segments(est_msa)

            ann_inter, ann_labels = to_inters_labels(ann_msa)
            est_inter, est_labels = to_inters_labels(est_msa)

            result = compute_results(
                ann_inter,
                est_inter,
                ann_labels,
                est_labels,
                bins=11,
                est_file="test.txt",
                weight=0.58,
            )
            acc = cal_acc(ann_msa, est_msa, post_digit=3)
            confusion = duration_confusion(ann_msa, est_msa, post_digit=3)
            class_metrics = classification_metrics_from_confusion(confusion)
            for key, value in confusion.items():
                confusion_total[key] += value

            ious[id] = cal_iou(ann_msa, est_msa)
            result["HitRate_1P"], result["HitRate_1R"], result["HitRate_1F"] = (
                mir_eval.segment.detection(ann_inter, est_inter, window=1, trim=False)
            )
            result.update({"id": Path(id).stem})
            result.update({"acc": acc})
            result.update(flatten_classification_metrics(class_metrics))
            for v in ious[id]:
                result.update({f"iou-{v['label']}": v["iou"]})
            del result["track_id"]
            del result["ds_name"]

            resultes.append(result)
        except Exception as e:
            logger.error(f"Error processing {id}: {e}")
            continue

    df = pd.DataFrame(resultes)
    df.to_csv(f"{output_dir}/eval_infer.csv", index=False)

    intsec_dur_total = defaultdict(float)
    uni_dur_total = defaultdict(float)

    for tid, value in ious.items():
        for item in value:
            label = item["label"]
            intsec_dur_total[label] += item.get("intsec_dur", 0)
            uni_dur_total[label] += item.get("uni_dur", 0)

    total_intsec = sum(intsec_dur_total.values())
    total_uni = sum(uni_dur_total.values())
    overall_iou = total_intsec / total_uni if total_uni > 0 else 0.0

    class_ious = {}
    for label in intsec_dur_total:
        intsec = intsec_dur_total[label]
        uni = uni_dur_total[label]
        class_ious[label] = intsec / uni if uni > 0 else 0.0

    global_class_metrics = classification_metrics_from_confusion(confusion_total)
    confusion_df = pd.DataFrame(
        0.0,
        index=sorted({gold for gold, _ in confusion_total.keys()} | set(EVAL_LABELS)),
        columns=sorted({pred for _, pred in confusion_total.keys()} | set(EVAL_LABELS)),
    )
    for (gold, pred), duration in confusion_total.items():
        confusion_df.loc[gold, pred] = duration
    confusion_df.to_csv(f"{output_dir}/duration_confusion.csv")
    row_sums = confusion_df.sum(axis=1).replace(0, np.nan)
    confusion_df.div(row_sums, axis=0).to_csv(
        f"{output_dir}/duration_confusion_recall.csv"
    )

    summary = pd.DataFrame(
        [
            {
                "num_samples": len(df),
                "HR.5F": df["HitRate_0.5F"].mean(),
                "HR3F": df["HitRate_3F"].mean(),
                "HR1F": df["HitRate_1F"].mean(),
                "PWF": df["PWF"].mean(),
                "Sf": df["Sf"].mean(),
                "acc": df["acc"].mean(),
                "iou": overall_iou,
                **{f"iou_{k}": v for k, v in class_ious.items()},
                **flatten_classification_metrics(global_class_metrics),
            }
        ]
    )
    with open(f"{output_dir}/eval_infer_summary.md", "w") as f:
        print(summary.to_markdown(), file=f)

    summary.to_csv(f"{output_dir}/eval_infer_summary.csv", index=False)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
