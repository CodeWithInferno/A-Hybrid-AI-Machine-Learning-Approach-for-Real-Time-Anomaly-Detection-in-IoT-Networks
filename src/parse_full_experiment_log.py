# parse_full_experiment_log.py
import re, sys, csv, math, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPORT_HEADER = "Classification Report:"
RESULTS_HEADER_RE = re.compile(r"=+\s*RESULTS FOR:\s*(.+?)\s*=+")
OPT_THRESH_RE   = re.compile(r"Optimal Threshold:\s*([-\d\.eE]+)")
ACCURACY_RE     = re.compile(r"Accuracy:\s*([-\d\.eE]+)")
SHAPE_RE        = re.compile(r"Combined sample created successfully!\s*Shape:\s*\((\d+),\s*(\d+)\)")
PREPROC_RE      = re.compile(r"Data preprocessing complete. Final shape:\s*\((\d+),\s*(\d+)\)")
DURATION_RE     = re.compile(r"Total experiment duration:\s*([-\d\.eE]+)\s*minutes")

def parse_classification_block(lines, start_idx):
    """
    Parse the sklearn text report that immediately follows 'Classification Report:'.
    Return dict with precision/recall/f1/support per class.
    """
    i = start_idx
    # Skip until header line with columns appears
    while i < len(lines) and ('precision' not in lines[i] or 'recall' not in lines[i]):
        i += 1
    i += 1  # move to first class row

    def grab_row(s):
        # e.g. "Benign (0)       0.9534    0.9271    0.9400    123456"
        parts = s.strip().split()
        # last token is support (int), previous three are f1, recall, precision (in that printed order)
        # But sklearn prints: label, precision, recall, f1-score, support
        # So we need to align by counting from the end.
        support = int(parts[-1])
        f1 = float(parts[-2])
        recall = float(parts[-3])
        precision = float(parts[-4])
        label = " ".join(parts[:-4])
        return label, precision, recall, f1, support

    # Two classes expected: "Benign (0)" and "Attack (1)"
    benign = grab_row(lines[i]); i += 1
    attack = grab_row(lines[i]); i += 1

    return {
        'benign_label': benign[0], 'p0': benign[1], 'r0': benign[2], 'f10': benign[3], 's0': benign[4],
        'attack_label': attack[0], 'p1': attack[1], 'r1': attack[2], 'f11': attack[3], 's1': attack[4]
    }

def reconstruct_confusion_from_pr_re_s(precision, recall, support_true):
    """
    For a single class treated as positive:
      TP = recall * support_true
      FP = TP * (1/precision - 1)
      FN = support_true - TP
    We’ll round sensibly.
    """
    TP = recall * support_true
    FN = support_true - TP
    FP = TP * (1.0 / precision - 1.0) if precision > 0 else 0.0
    return round(TP), round(FP), round(FN)

def make_confusion_binary(p0, r0, s0, p1, r1, s1):
    """
    Build 2x2 confusion matrix with positive = class 1.
    Using class-1 as positive for (TP,FP,FN), and class-0 row to estimate TN.
    """
    TP, FP_from_p1, FN = reconstruct_confusion_from_pr_re_s(p1, r1, s1)
    # TN from class-0 recall on its support (correctly predicted zeros)
    TN_est = round(r0 * s0)
    # Another estimate for FP is the number of actual 0 predicted as 1: FP = s0 - TN
    FP_alt = s0 - TN_est

    # Resolve tiny rounding mismatches by trusting supports:
    FP = FP_alt
    # Recompute precision from resolved FP for info
    prec1_est = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Sanity: all cells non-negative
    TP = max(TP, 0); FP = max(FP, 0); FN = max(FN, 0); TN_est = max(TN_est, 0)
    return (TP, FP, FN, TN_est, prec1_est)

def plot_confusion(cm, labels, title, outpath):
    plt.figure(figsize=(5,4))
    data = np.array([[cm[3], cm[1]],   # [[TN, FP],
                     [cm[2], cm[0]]])  #  [FN, TP]]
    plt.imshow(data, interpolation='nearest')
    plt.title(title)
    plt.xticks([0,1], [f'Pred 0 ({labels[0]})', f'Pred 1 ({labels[1]})'], rotation=20)
    plt.yticks([0,1], [f'True 0 ({labels[0]})', f'True 1 ({labels[1]})'])
    # annotate
    for (y,x), val in np.ndenumerate(data):
        plt.text(x, y, f'{val}', ha='center', va='center')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main(log_path):
    text = Path(log_path).read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()

    # Global info
    ds_shape = SHAPE_RE.search(text)
    preproc = PREPROC_RE.search(text)
    duration = DURATION_RE.search(text)

    dataset_rows = int(ds_shape.group(1)) if ds_shape else None
    dataset_cols = int(ds_shape.group(2)) if ds_shape else None
    final_rows  = int(preproc.group(1)) if preproc else None
    final_cols  = int(preproc.group(2)) if preproc else None
    minutes     = float(duration.group(1)) if duration else None

    # Per-model blocks
    results = []
    i = 0
    while i < len(lines):
        m = RESULTS_HEADER_RE.search(lines[i])
        if m:
            model = m.group(1).strip()
            # look forward for threshold, accuracy, and report
            j = i+1
            thr = acc = None
            report_idx = None
            while j < len(lines) and (lines[j].strip() != "" or report_idx is None):
                if thr is None:
                    tm = OPT_THRESH_RE.search(lines[j]); 
                    if tm: thr = float(tm.group(1))
                if acc is None:
                    am = ACCURACY_RE.search(lines[j]); 
                    if am: acc = float(am.group(1))
                if REPORT_HEADER in lines[j]:
                    report_idx = j
                    break
                j += 1
            if report_idx is None:
                i += 1
                continue
            stats = parse_classification_block(lines, report_idx)
            # confusion matrix (positive=Attack (1))
            TP, FP, FN, TN, p1_est = make_confusion_binary(
                stats['p0'], stats['r0'], stats['s0'],
                stats['p1'], stats['r1'], stats['s1']
            )
            results.append({
                'model': model,
                'optimal_threshold': thr,
                'accuracy': acc,
                'benign_support': stats['s0'],
                'attack_support': stats['s1'],
                'precision_0': stats['p0'], 'recall_0': stats['r0'], 'f1_0': stats['f10'],
                'precision_1': stats['p1'], 'recall_1': stats['r1'], 'f1_1': stats['f11'],
                'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
                'precision_1_recomputed': p1_est
            })
            i = j + 1
        else:
            i += 1

    # Write CSV
    out_dir = Path(log_path).parent
    csv_path = out_dir / "metrics_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "model","optimal_threshold","accuracy",
            "benign_support","attack_support",
            "precision_0","recall_0","f1_0",
            "precision_1","recall_1","f1_1",
            "TN","FP","FN","TP","precision_1_recomputed"
        ]
        writer.writerow(["dataset_rows","dataset_cols","final_rows","final_cols","duration_minutes"])
        writer.writerow([dataset_rows,dataset_cols,final_rows,final_cols,minutes])
        writer.writerow([])
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r["model"], r["optimal_threshold"], r["accuracy"],
                r["benign_support"], r["attack_support"],
                r["precision_0"], r["recall_0"], r["f1_0"],
                r["precision_1"], r["recall_1"], r["f1_1"],
                r["TN"], r["FP"], r["FN"], r["TP"], r["precision_1_recomputed"]
            ])

    # Plots
    for r in results:
        title = f'Confusion Matrix — {r["model"]} (Pos=Attack)'
        out_png = out_dir / f'confusion_matrix_{r["model"].replace(" ", "_")}.png'
        plot_confusion((r["TP"], r["FP"], r["FN"], r["TN"]),
                       labels=("Benign (0)","Attack (1)"),
                       title=title, outpath=out_png)

    # Quick summary text
    summary = out_dir / "quick_summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n")
        if dataset_rows:
            f.write(f"Sampled dataset shape: {dataset_rows} rows × {dataset_cols} columns\n")
        if final_rows:
            f.write(f"Preprocessed matrix: {final_rows} × {final_cols}\n")
        if minutes:
            f.write(f"Total duration: {minutes:.2f} minutes\n")
        f.write("\nPer-model metrics:\n")
        for r in results:
            f.write(f"\n[{r['model']}]\n")
            f.write(f"  Optimal threshold: {r['optimal_threshold']}\n")
            f.write(f"  Accuracy: {r['accuracy']}\n")
            f.write(f"  Supports — Benign: {r['benign_support']}, Attack: {r['attack_support']}\n")
            f.write(f"  Precision/Recall/F1 (Attack): {r['precision_1']:.4f}/{r['recall_1']:.4f}/{r['f1_1']:.4f}\n")
            f.write(f"  Confusion Matrix (Pos=Attack):\n")
            f.write(f"    TN={r['TN']}  FP={r['FP']}  FN={r['FN']}  TP={r['TP']}\n")

    print(f"Wrote {csv_path.name}, confusion_matrix_*.png, and quick_summary.txt to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_full_experiment_log.py <path_to_full_experiment_log.txt>")
        sys.exit(1)
    main(sys.argv[1])
