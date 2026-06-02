import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    # 先划分 60% 训练集，40% 临时集
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df["fraudulent"]
    )

    # 再把临时集一分为二：20% 验证集，20% 测试集
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["fraudulent"]
    )

    # 保存
    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    # 打印结果
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["fraudulent"].value_counts())

    print("\nVal label distribution:")
    print(val_df["fraudulent"].value_counts())

    print("\nTest label distribution:")
    print(test_df["fraudulent"].value_counts())

    print(f"\nSaved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
