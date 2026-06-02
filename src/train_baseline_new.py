import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(r"D:\homework\FakeJobRisk_Project")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["employment_type", "required_experience", "required_education", "industry", "function"]
NUM_COLS = ["telecommuting", "has_company_logo", "has_questions"]
TARGET_COL = "fraudulent"

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df

def combine_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def make_xy(df: pd.DataFrame):
    df = combine_text(df)
    y = df[TARGET_COL].astype(int)
    X = df[["combined_text"] + CAT_COLS + NUM_COLS].copy()

    for col in CAT_COLS:
        if col not in X.columns:
            X[col] = "Unknown"
        X[col] = X[col].fillna("Unknown").astype(str)

    for col in NUM_COLS:
        if col not in X.columns:
            X[col] = 0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

    return X, y

def build_full_preprocessor():
    text_transformer = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2
        ))
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "combined_text"),
            ("cat", cat_transformer, CAT_COLS),
            ("num", num_transformer, NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return preprocessor

def build_structured_preprocessor():
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CAT_COLS),
            ("num", num_transformer, NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return preprocessor

def evaluate(y_true, y_pred, prefix=""):
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def save_confusion_matrix(y_true, y_pred, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        values_format="d",
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run_model(model_name, pipeline, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n========== {model_name} ==========")
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    val_metrics = evaluate(y_val, val_pred, prefix="val_")
    test_metrics = evaluate(y_test, test_pred, prefix="test_")

    print("\n[Validation]")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))

    print("\n[Test]")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))

    cm_path = FIG_DIR / f"{model_name}_new_test_confusion_matrix.png"
    save_confusion_matrix(y_test, test_pred, f"{model_name} - Test Confusion Matrix", cm_path)

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": test_pred
    })
    pred_df.to_csv(TABLE_DIR / f"{model_name}_new_test_predictions.csv", index=False, encoding="utf-8-sig")

    return {**val_metrics, **test_metrics}

def main():
    train_df, val_df, test_df = load_data()

    X_train, y_train = make_xy(train_df)
    X_val, y_val = make_xy(val_df)
    X_test, y_test = make_xy(test_df)

    full_preprocessor = build_full_preprocessor()
    structured_preprocessor = build_structured_preprocessor()

    models = {
        "logreg_full": Pipeline([
            ("preprocess", full_preprocessor),
            ("clf", LogisticRegression(
                max_iter=5000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "nb_full": Pipeline([
            ("preprocess", full_preprocessor),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "rf_structured": Pipeline([
            ("preprocess", structured_preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            ))
        ]),
    }

    results = []
    for name, pipe in models.items():
        metrics = run_model(name, pipe, X_train, y_train, X_val, y_val, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    cols = ["model"] + [c for c in results_df.columns if c != "model"]
    results_df = results_df[cols]

    results_df.to_csv(TABLE_DIR / "baseline_metrics_new.csv", index=False, encoding="utf-8-sig")
    print("\n========== Summary ==========")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved metrics to: {TABLE_DIR / 'baseline_metrics_new.csv'}")
    print(f"Saved figures to: {FIG_DIR}")

if __name__ == "__main__":
    main()
