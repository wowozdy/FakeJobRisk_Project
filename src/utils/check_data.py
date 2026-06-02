import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    print("===== Dataset Info =====")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\n===== Missing Values =====")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_ratio = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_ratio_%": missing_ratio.round(2)
    })
    print(missing_df)

    print("\n===== Label Distribution =====")
    if "fraudulent" in df.columns:
        print(df["fraudulent"].value_counts())
        print("\nLabel ratio:")
        print((df["fraudulent"].value_counts(normalize=True) * 100).round(2))

    # 保存缺失值统计
    missing_df.to_csv(RESULTS_DIR / "missing_values.csv", encoding="utf-8-sig")

    # 保存标签分布
    if "fraudulent" in df.columns:
        label_counts = df["fraudulent"].value_counts().rename_axis("fraudulent").reset_index(name="count")
        label_counts["ratio_%"] = (label_counts["count"] / label_counts["count"].sum() * 100).round(2)
        label_counts.to_csv(RESULTS_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
