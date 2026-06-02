from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD

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

SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6
LR = 1e-3
SVD_COMPONENTS = 256

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    y = df[TARGET_COL].astype(int).values
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

def build_preprocessor():
    text_transformer = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
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
        sparse_threshold=1.0
    )
    return preprocessor

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def to_dense_array(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)

def metrics_from_preds(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def predict(model, X_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().numpy()
    return preds

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

def main():
    train_df, val_df, test_df = load_data()
    X_train_df, y_train = make_xy(train_df)
    X_val_df, y_val = make_xy(val_df)
    X_test_df, y_test = make_xy(test_df)

    preprocessor = build_preprocessor()

    print("Fitting preprocessor on train data...")
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    print("Applying SVD dimensionality reduction...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
    X_train = svd.fit_transform(X_train)
    X_val = svd.transform(X_val)
    X_test = svd.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("Feature shape after SVD:", X_train.shape)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(input_dim=X_train.shape[1]).to(device)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_macro_f1 = -1
    best_state = None
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_recall": [],
        "val_accuracy": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_true = []
        val_pred = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu().numpy()
                val_pred.extend(preds.tolist())
                val_true.extend(yb.cpu().numpy().astype(int).flatten().tolist())

        val_pred = np.array(val_pred)
        val_true = np.array(val_true)

        val_metrics = metrics_from_preds(val_true, val_pred)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        scheduler.step(val_metrics["macro_f1"])

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    train_pred = predict(model, torch.tensor(X_train, dtype=torch.float32).to(device))
    val_pred = predict(model, torch.tensor(X_val, dtype=torch.float32).to(device))
    test_pred = predict(model, torch.tensor(X_test, dtype=torch.float32).to(device))

    print("\n========== MLP Validation ==========")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))

    print("\n========== MLP Test ==========")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))

    val_metrics = metrics_from_preds(y_val, val_pred)
    test_metrics = metrics_from_preds(y_test, test_pred)

    results_df = pd.DataFrame([{
        "model": "mlp_new",
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }])

    results_path = TABLE_DIR / "mlp_metrics_new.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    pred_path = TABLE_DIR / "mlp_new_test_predictions.csv"
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(pred_path, index=False, encoding="utf-8-sig")

    cm_path = FIG_DIR / "mlp_new_test_confusion_matrix.png"
    save_confusion_matrix(y_test, test_pred, "mlp_new - Test Confusion Matrix", cm_path)

    # curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_accuracy"], label="val_accuracy")
    plt.plot(history["epoch"], history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Curve")
    plt.legend()

    plt.tight_layout()
    curve_path = FIG_DIR / "mlp_new_training_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()

    history_path = TABLE_DIR / "mlp_new_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

    print("\n========== Summary ==========")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved metrics to: {results_path}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved training curve to: {curve_path}")
    print(f"Saved history to: {history_path}")

if __name__ == "__main__":
    main()
