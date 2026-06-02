from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = DATA_DIR / "test.csv"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["employment_type", "required_experience", "required_education", "industry", "function"]
NUM_COLS = ["telecommuting", "has_company_logo", "has_questions"]
TARGET_COL = "fraudulent"

def parse_args():
    parser = argparse.ArgumentParser(description="Error analysis for fake job classification.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mlp_ce_lr0001",
        help="Model prefix used in results/tables, e.g. mlp_ce_lr0001"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of misclassified samples to save as preview"
    )
    return parser.parse_args()

def load_test_data():
    df = pd.read_csv(TEST_PATH)
    return df

def load_predictions(model_name: str):
    pred_path = TABLE_DIR / f"{model_name}_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    pred_df = pd.read_csv(pred_path)
    if "y_true" not in pred_df.columns or "y_pred" not in pred_df.columns:
        raise ValueError(f"{pred_path} must contain columns: y_true, y_pred")
    return pred_df, pred_path

def combine_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["combined_text"] = (
        df[TEXT_COLS]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = combine_text(df)

    # 文本长度特征
    df["text_char_len"] = df["combined_text"].fillna("").astype(str).str.len()
    df["text_word_len"] = df["combined_text"].fillna("").astype(str).str.split().apply(len)

    # 各文本字段是否缺失
    for col in TEXT_COLS:
        df[f"{col}_missing"] = df[col].fillna("").astype(str).str.strip().eq("")

    df["missing_text_fields"] = df[[f"{col}_missing" for col in TEXT_COLS]].sum(axis=1)

    # 类别字段缺失统计（Unknown / 空值都算）
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    df["missing_cat_fields"] = 0
    for col in CAT_COLS:
        df["missing_cat_fields"] += (
            df[col].fillna("Unknown").astype(str).str.strip().eq("") |
            df[col].fillna("Unknown").astype(str).str.strip().str.lower().eq("unknown")
        ).astype(int)

    # 数值字段缺失统计（异常值按缺失处理）
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df

def safe_div(a, b):
    return 0.0 if b == 0 else a / b

def compute_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = safe_div(tp + tn, len(y_true))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def summarize_group(df: pd.DataFrame, name: str):
    # 返回适合写论文的统计量
    return {
        "group": name,
        "count": len(df),
        "avg_text_char_len": float(df["text_char_len"].mean()) if len(df) else 0.0,
        "avg_text_word_len": float(df["text_word_len"].mean()) if len(df) else 0.0,
        "avg_missing_text_fields": float(df["missing_text_fields"].mean()) if len(df) else 0.0,
        "avg_missing_cat_fields": float(df["missing_cat_fields"].mean()) if len(df) else 0.0,
        "title_missing_rate": float(df["title_missing"].mean()) if len(df) else 0.0,
        "company_profile_missing_rate": float(df["company_profile_missing"].mean()) if len(df) else 0.0,
        "description_missing_rate": float(df["description_missing"].mean()) if len(df) else 0.0,
        "requirements_missing_rate": float(df["requirements_missing"].mean()) if len(df) else 0.0,
        "benefits_missing_rate": float(df["benefits_missing"].mean()) if len(df) else 0.0,
    }

def save_plots(df_all: pd.DataFrame, model_name: str):
    # 1) 文本长度分布：正确 vs 错误
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for label, subset in [("correct", df_all[df_all["is_error"] == 0]), ("error", df_all[df_all["is_error"] == 1])]:
        plt.hist(subset["text_char_len"], bins=40, alpha=0.6, label=label)
    plt.xlabel("Text Character Length")
    plt.ylabel("Count")
    plt.title("Text Length Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, subset in [("correct", df_all[df_all["is_error"] == 0]), ("error", df_all[df_all["is_error"] == 1])]:
        plt.hist(subset["missing_text_fields"], bins=np.arange(-0.5, 6.5, 1), alpha=0.6, label=label)
    plt.xlabel("Missing Text Fields")
    plt.ylabel("Count")
    plt.title("Missing Text Fields Distribution")
    plt.legend()

    plt.tight_layout()
    out_path = FIG_DIR / f"{model_name}_error_analysis_distributions.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    # 2) FP/FN 的文本长度箱线图
    error_df = df_all[df_all["is_error"] == 1].copy()
    if len(error_df) > 0:
        plt.figure(figsize=(8, 5))
        groups = [
            error_df[error_df["error_type"] == "FP"]["text_char_len"],
            error_df[error_df["error_type"] == "FN"]["text_char_len"],
        ]
        plt.boxplot(groups, labels=["FP", "FN"], showmeans=True)
        plt.ylabel("Text Character Length")
        plt.title("Text Length for FP vs FN")
        plt.tight_layout()
        out_path2 = FIG_DIR / f"{model_name}_error_fp_fn_boxplot.png"
        plt.savefig(out_path2, dpi=200)
        plt.close()

def write_report(model_name: str, metrics: dict, summary_rows: list, out_path: Path):
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("")
    lines.append("Overall Metrics")
    lines.append(f"TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    lines.append(f"Accuracy={metrics['accuracy']:.4f}")
    lines.append(f"Precision={metrics['precision']:.4f}")
    lines.append(f"Recall={metrics['recall']:.4f}")
    lines.append(f"F1={metrics['f1']:.4f}")
    lines.append("")

    lines.append("Group Statistics")
    for row in summary_rows:
        lines.append(f"[{row['group']}]")
        lines.append(f"count={row['count']}")
        lines.append(f"avg_text_char_len={row['avg_text_char_len']:.2f}")
        lines.append(f"avg_text_word_len={row['avg_text_word_len']:.2f}")
        lines.append(f"avg_missing_text_fields={row['avg_missing_text_fields']:.2f}")
        lines.append(f"avg_missing_cat_fields={row['avg_missing_cat_fields']:.2f}")
        lines.append(f"title_missing_rate={row['title_missing_rate']:.4f}")
        lines.append(f"company_profile_missing_rate={row['company_profile_missing_rate']:.4f}")
        lines.append(f"description_missing_rate={row['description_missing_rate']:.4f}")
        lines.append(f"requirements_missing_rate={row['requirements_missing_rate']:.4f}")
        lines.append(f"benefits_missing_rate={row['benefits_missing_rate']:.4f}")
        lines.append("")

    lines.append("Suggested paper interpretation:")
    lines.append(
        "False negatives usually correspond to suspicious jobs with weaker textual cues, "
        "shorter descriptions, or incomplete company information; false positives may appear "
        "when legitimate jobs contain overly promotional language or unusually sparse field patterns."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to: {out_path}")

def main():
    args = parse_args()
    model_name = args.model_name

    test_df = load_test_data()
    pred_df, pred_path = load_predictions(model_name)

    if len(test_df) != len(pred_df):
        raise ValueError(
            f"Length mismatch: test_df={len(test_df)}, pred_df={len(pred_df)}. "
            "Predictions must align with test set row order."
        )

    # 合并原始测试集 + 预测结果
    df = test_df.reset_index(drop=True).copy()
    df["y_true"] = pred_df["y_true"].astype(int).values
    df["y_pred"] = pred_df["y_pred"].astype(int).values

    df = enrich_features(df)

    # 错误类型
    df["is_error"] = (df["y_true"] != df["y_pred"]).astype(int)
    df["error_type"] = np.where(
        (df["y_true"] == 0) & (df["y_pred"] == 1), "FP",
        np.where((df["y_true"] == 1) & (df["y_pred"] == 0), "FN", "Correct")
    )

    # 保存完整分析表
    full_path = TABLE_DIR / f"{model_name}_error_analysis_full.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")

    # 计算整体指标
    metrics = compute_basic_metrics(df["y_true"], df["y_pred"])

    # 提取错误样本
    error_df = df[df["is_error"] == 1].copy()
    fp_df = error_df[error_df["error_type"] == "FP"].copy()
    fn_df = error_df[error_df["error_type"] == "FN"].copy()

    # 保存错误样本
    fp_path = TABLE_DIR / f"{model_name}_false_positives.csv"
    fn_path = TABLE_DIR / f"{model_name}_false_negatives.csv"
    error_path = TABLE_DIR / f"{model_name}_misclassified_samples.csv"

    fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")
    fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
    error_df.to_csv(error_path, index=False, encoding="utf-8-sig")

    # 汇总统计
    summary_rows = [
        summarize_group(df[df["is_error"] == 0], "Correct"),
        summarize_group(fp_df, "False Positive"),
        summarize_group(fn_df, "False Negative"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLE_DIR / f"{model_name}_error_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 保存前 N 个错误案例，方便写论文
    top_n = args.top_n
    preview_cols = [
        "y_true", "y_pred", "error_type",
        "text_char_len", "text_word_len", "missing_text_fields", "missing_cat_fields",
        "title", "company_profile", "description", "requirements", "benefits",
        "employment_type", "required_experience", "required_education", "industry", "function"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    preview_df = error_df[preview_cols].head(top_n)
    preview_path = TABLE_DIR / f"{model_name}_error_preview_top{top_n}.csv"
    preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")

    # 图表
    save_plots(df, model_name)

    # 文本报告
    report_path = REPORT_DIR / f"{model_name}_error_report.txt"
    write_report(model_name, metrics, summary_rows, report_path)

    # 控制台输出
    print("\n========== Error Analysis Summary ==========")
    print(f"Model: {model_name}")
    print(f"Predictions file: {pred_path}")
    print(f"Full analysis saved to: {full_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"False positives saved to: {fp_path}")
    print(f"False negatives saved to: {fn_path}")
    print(f"Misclassified samples saved to: {error_path}")
    print(f"Preview saved to: {preview_path}")
    print("\nMetrics:")
    print(
        f"TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, "
        f"Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
    )
    print("\nGroup summary:")
    print(summary_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()