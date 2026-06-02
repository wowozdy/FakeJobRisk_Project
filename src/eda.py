import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def text_len(x):
    if pd.isna(x):
        return 0
    return len(str(x))

def plot_missing_values(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(12, 6))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_values_bar.png", dpi=200)
    plt.close()

def plot_label_distribution(df):
    if "fraudulent" not in df.columns:
        return
    counts = df["fraudulent"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Label Distribution")
    plt.xlabel("Fraudulent")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_distribution_bar.png", dpi=200)
    plt.close()

def plot_text_lengths(df):
    text_cols = ["title", "benefits", "description", "requirements"]
    for col in text_cols:
        if col not in df.columns:
            continue
        lengths = df[col].fillna("").astype(str).map(len)

        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=50, edgecolor="black")
        plt.title(f"Length Distribution: {col}")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_length_hist.png", dpi=200)
        plt.close()

def main():
    df = pd.read_csv(DATA_PATH)

    # 缺失值统计表
    missing_count = df.isnull().sum().sort_values(ascending=False)
    missing_ratio = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_ratio_%": missing_ratio.round(2)
    })
    missing_df.to_csv(TABLE_DIR / "missing_values.csv", encoding="utf-8-sig")

    # 标签分布表
    if "fraudulent" in df.columns:
        label_counts = df["fraudulent"].value_counts().rename_axis("fraudulent").reset_index(name="count")
        label_counts["ratio_%"] = (label_counts["count"] / label_counts["count"].sum() * 100).round(2)
        label_counts.to_csv(TABLE_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig")

    # 画图
    plot_missing_values(df)
    plot_label_distribution(df)
    plot_text_lengths(df)

    print("EDA completed successfully.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")

if __name__ == "__main__":
    main()
