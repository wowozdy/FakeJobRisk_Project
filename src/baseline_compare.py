from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_SUMMARY_PATH = TABLE_DIR / "all_models_summary.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline comparison: random guess vs majority vote.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1000,
        help="Number of Monte Carlo runs for random guessing baseline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--positive_label",
        type=int,
        default=1,
        help="Positive class label."
    )
    return parser.parse_args()

def load_labels():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if "fraudulent" not in train_df.columns or "fraudulent" not in test_df.columns:
        raise ValueError("Both train.csv and test.csv must contain column: fraudulent")

    y_train = train_df["fraudulent"].astype(int).values
    y_test = test_df["fraudulent"].astype(int).values
    return y_train, y_test

def evaluate(y_true, y_pred, positive_label=1):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def random_guess_baseline(y_train, y_test, n_runs=1000, seed=42):
    """
    Randomly predict classes according to the class distribution in y_train.
    Repeat n_runs times and report mean/std.
    """
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()

    metrics_list = []
    for _ in range(n_runs):
        y_pred = rng.choice(classes, size=len(y_test), p=probs)
        metrics_list.append(evaluate(y_test, y_pred))

    df = pd.DataFrame(metrics_list)
    summary = {
        "model": "random_guess",
        "accuracy": df["accuracy"].mean(),
        "precision": df["precision"].mean(),
        "recall": df["recall"].mean(),
        "f1": df["f1"].mean(),
        "macro_f1": df["macro_f1"].mean(),
        "accuracy_std": df["accuracy"].std(),
        "precision_std": df["precision"].std(),
        "recall_std": df["recall"].std(),
        "f1_std": df["f1"].std(),
        "macro_f1_std": df["macro_f1"].std(),
    }
    return summary

def majority_vote_baseline(y_train, y_test):
    """
    Always predict the majority class in the training set.
    """
    majority_class = pd.Series(y_train).mode().iloc[0]
    y_pred = np.full_like(y_test, fill_value=majority_class)
    metrics = evaluate(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    summary = {
        "model": "majority_vote",
        "majority_class": int(majority_class),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        **metrics,
    }
    return summary

def load_model_summary():
    if not MODEL_SUMMARY_PATH.exists():
        print(f"⚠️ Warning: {MODEL_SUMMARY_PATH} not found. Only baseline results will be saved.")
        return pd.DataFrame()

    df = pd.read_csv(MODEL_SUMMARY_PATH)

    # 兼容你已有的 all_models_summary.csv 字段
    expected_cols = ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_macro_f1"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{MODEL_SUMMARY_PATH} missing columns: {missing}")

    df = df.rename(columns={
        "test_accuracy": "accuracy",
        "test_precision": "precision",
        "test_recall": "recall",
        "test_f1": "f1",
        "test_macro_f1": "macro_f1",
    })
    return df

def main():
    args = parse_args()
    y_train, y_test = load_labels()

    # Baselines
    random_summary = random_guess_baseline(
        y_train, y_test,
        n_runs=args.n_runs,
        seed=args.seed
    )
    majority_summary = majority_vote_baseline(y_train, y_test)

    baseline_df = pd.DataFrame([
        random_summary,
        majority_summary
    ])

    # 只保留常用展示列
    baseline_display = baseline_df[[
        "model", "accuracy", "precision", "recall", "f1", "macro_f1"
    ]].copy()

    # 读取已有模型结果
    model_df = load_model_summary()

    # 合并
    if len(model_df) > 0:
        compare_df = pd.concat([
            baseline_display,
            model_df[["model", "accuracy", "precision", "recall", "f1", "macro_f1"]]
        ], ignore_index=True)
    else:
        compare_df = baseline_display

    # 排序：按 macro_f1 从高到低
    compare_df = compare_df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)

    # 输出文件
    out_csv = TABLE_DIR / "baseline_model_comparison.csv"
    out_txt = TABLE_DIR / "baseline_model_comparison.txt"

    compare_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 文本汇总
    lines = []
    lines.append("Baseline Comparison Summary")
    lines.append("=" * 40)
    lines.append(f"Random guess runs: {args.n_runs}")
    lines.append(f"Random seed: {args.seed}")
    lines.append("")
    lines.append(compare_df.round(4).to_string(index=False))

    # baseline 额外统计
    lines.append("")
    lines.append("Baseline Details")
    lines.append("-" * 40)
    lines.append(
        f"Random guess mean metrics: "
        f"Accuracy={random_summary['accuracy']:.4f}, "
        f"Precision={random_summary['precision']:.4f}, "
        f"Recall={random_summary['recall']:.4f}, "
        f"F1={random_summary['f1']:.4f}, "
        f"Macro-F1={random_summary['macro_f1']:.4f}"
    )
    lines.append(
        f"Majority vote class: {majority_summary['majority_class']}, "
        f"TP={majority_summary['tp']}, TN={majority_summary['tn']}, "
        f"FP={majority_summary['fp']}, FN={majority_summary['fn']}, "
        f"Accuracy={majority_summary['accuracy']:.4f}, "
        f"Precision={majority_summary['precision']:.4f}, "
        f"Recall={majority_summary['recall']:.4f}, "
        f"F1={majority_summary['f1']:.4f}, "
        f"Macro-F1={majority_summary['macro_f1']:.4f}"
    )

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # 控制台输出
    print("\n✅ Baseline comparison completed.")
    print(f"Saved CSV to: {out_csv}")
    print(f"Saved TXT to: {out_txt}")
    print("\nComparison table:")
    print(compare_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()