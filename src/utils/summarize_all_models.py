# summarize_all_models.py
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

# 模型列表与对应文件
model_configs = [
    ("logreg_full", "baseline_metrics_new.csv"),
    ("nb_full", "baseline_metrics_new.csv"),
    ("rf_structured", "baseline_metrics_new.csv"),
    ("mlp_ce_lr0001", "mlp_ce_lr0001_metrics.csv"),
    # 可选：添加其他 MLP 变体
    # ("mlp_ce_do05", "mlp_ce_do05_metrics.csv"),
    # ("mlp_ce_lr001", "mlp_ce_lr001_metrics.csv"),
]

rows = []
for name, filename in model_configs:
    try:
        df = pd.read_csv(TABLE_DIR / filename)
        row = df[df["model"] == name].iloc[0] if "model" in df.columns else df.iloc[0]
        rows.append({
            "model": name,
            "test_accuracy": row.get("test_accuracy"),
            "test_precision": row.get("test_precision"),
            "test_recall": row.get("test_recall"),
            "test_f1": row.get("test_f1"),
            "test_macro_f1": row.get("test_macro_f1"),
        })
    except Exception as e:
        print(f"⚠️ 未找到 {name} 的结果：{e}")

summary_df = pd.DataFrame(rows)
cols = ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_macro_f1"]
summary_df = summary_df[cols]

# 保存汇总表
summary_df.to_csv(TABLE_DIR / "all_models_summary.csv", index=False, encoding="utf-8-sig")
print("\n✅ 模型对比汇总表已生成：")
print(summary_df.round(4))