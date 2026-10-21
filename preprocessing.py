"""
preprocessing.py
----------------
Mirrors the EXACT pipeline from the Spaceship Titanic notebook — cell by cell.

Round-2 (improved) pipeline order (matches notebook cells 54-70):
  1. Feature engineering on full df (GroupSize, Deck, CabinNumber, Side) – pre-split
  2. train_test_split(test_size=0.2, random_state=42, stratify=y)
  3. Engineer TotalSpending on train/test copies (post-split)
  4. Impute spending columns with train medians (post-split)
  5. Recompute TotalSpending; create NoSpending
  6. LabelEncoder — fit on train only — only 'object' dtype columns
     (CryoSleep and VIP are bool, NOT encoded here, as in the notebook)
  7. StandardScaler — fit_transform on ALL columns of X_train_final
  8. Impute Age and CabinNumber AFTER scaling (notebook ordering)

DO NOT change this without also changing the notebook — these match intentionally.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ── Column constants ───────────────────────────────────────────────────────────
SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

FEATURE_COLS = [
    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Deck", "CabinNumber", "Side", "GroupSize", "TotalSpending", "NoSpending",
]

# Only object-dtype columns are label-encoded (notebook cell 62 uses include=['object'])
# CryoSleep and VIP are bool — they are NOT encoded; they pass through as 0/1 bools
OBJECT_CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side"]


def load_and_preprocess(csv_path: str):
    """
    Run the full notebook Round-2 pipeline and return all objects needed by
    the dashboard.

    Returns
    -------
    X_train_np, X_test_np : np.ndarray  (scaled, ready for sklearn)
    y_train, y_test        : np.ndarray
    encoders               : dict[col -> LabelEncoder]   (object cols only)
    scaler                 : StandardScaler
    encoder_categories     : dict[col -> list[str]]      (known classes per col)
    train_spending_medians : dict[col -> float]          (pre-scale spending medians)
    raw_col_order          : list[str]                   (X_train_final column order)
    """
    df = pd.read_csv(csv_path)

    # ── Step 1: Feature engineering on FULL df (pre-split) ────────────────────
    data = df.copy()
    data[["GroupId", "GroupMember"]] = data["PassengerId"].str.split("_", expand=True)
    data[["Deck", "CabinNumber", "Side"]] = data["Cabin"].str.split("/", expand=True)
    group_sizes = data["GroupId"].value_counts()
    data["GroupSize"] = data["GroupId"].map(group_sizes)
    data["CabinNumber"] = pd.to_numeric(data["CabinNumber"], errors="coerce")
    data = data.drop(columns=["PassengerId", "GroupId", "GroupMember", "Cabin", "Name"])

    X = data.drop(columns=["Transported"])
    y = data["Transported"].astype(int).values

    # ── Step 2: Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Step 3: TotalSpending on copies (post-split) ───────────────────────────
    X_train_imp = X_train.copy()
    X_test_imp  = X_test.copy()
    X_train_imp["TotalSpending"] = X_train_imp[SPENDING_COLS].sum(axis=1)
    X_test_imp["TotalSpending"]  = X_test_imp[SPENDING_COLS].sum(axis=1)

    # ── Step 4: Impute spending cols from train medians ────────────────────────
    train_spending_medians = {}
    for col in SPENDING_COLS:
        med = X_train_imp[col].median()
        train_spending_medians[col] = med
        X_train_imp[col] = X_train_imp[col].fillna(med)
        X_test_imp[col]  = X_test_imp[col].fillna(med)

    # ── Step 5: Recompute TotalSpending; create NoSpending ────────────────────
    X_train_imp["TotalSpending"] = X_train_imp[SPENDING_COLS].sum(axis=1)
    X_test_imp["TotalSpending"]  = X_test_imp[SPENDING_COLS].sum(axis=1)
    X_train_imp["NoSpending"] = (X_train_imp["TotalSpending"] == 0).astype(int)
    X_test_imp["NoSpending"]  = (X_test_imp["TotalSpending"]  == 0).astype(int)

    # ── Step 6: LabelEncoder — only 'object' dtype columns ────────────────────
    X_train_final = X_train_imp.copy()
    X_test_final  = X_test_imp.copy()

    encoders           = {}
    encoder_categories = {}

    obj_cols = X_train_final.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        le = LabelEncoder()
        X_train_final[col] = le.fit_transform(X_train_final[col].astype(str))
        known = list(le.classes_)
        encoder_categories[col] = known
        X_test_final[col] = X_test_final[col].astype(str).apply(
            lambda v, _le=le, _known=known:
                _le.transform([v])[0] if v in _known else -1
        )
        encoders[col] = le

    # ── Step 7: StandardScaler on ALL columns ─────────────────────────────────
    all_cols = list(X_train_final.columns)
    scaler = StandardScaler()
    X_train_scaled_df = pd.DataFrame(
        scaler.fit_transform(X_train_final),
        columns=all_cols, index=X_train_final.index
    )
    X_test_scaled_df = pd.DataFrame(
        scaler.transform(X_test_final),
        columns=all_cols, index=X_test_final.index
    )

    # ── Step 8: Impute Age and CabinNumber AFTER scaling ──────────────────────
    for col in ["Age", "CabinNumber"]:
        med = X_train_scaled_df[col].median()
        X_train_scaled_df[col] = X_train_scaled_df[col].fillna(med)
        X_test_scaled_df[col]  = X_test_scaled_df[col].fillna(med)

    X_train_np = X_train_scaled_df[FEATURE_COLS].values
    X_test_np  = X_test_scaled_df[FEATURE_COLS].values

    return (
        X_train_np, X_test_np,
        y_train, y_test,
        encoders, scaler,
        encoder_categories,
        train_spending_medians,
        all_cols,
    )


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train all four models exactly as in the notebook."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN (K=9)":           KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes":         GaussianNB(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> list:
    """Return per-model metrics rounded to 4 d.p., matching the notebook output."""
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, preds, zero_division=0), 4),
        })
    return results


def get_confusion_matrices(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    return {
        name: confusion_matrix(y_test, model.predict(X_test))
        for name, model in models.items()
    }


def preprocess_single_input(
    user_input: dict,
    encoders: dict,
    encoder_categories: dict,
    train_spending_medians: dict,
    scaler: StandardScaler,
    all_cols: list,
) -> np.ndarray:
    """
    Preprocess one passenger's raw input using exactly the same steps as training.

    user_input keys expected:
        HomePlanet, CryoSleep, Destination, Age, VIP,
        RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
        Deck, CabinNumber, Side, GroupSize
    """
    row = {}

    # Spending columns — impute with train medians if missing
    for col in SPENDING_COLS:
        val = user_input.get(col, None)
        row[col] = float(val) if val is not None else train_spending_medians.get(col, 0.0)

    # TotalSpending + NoSpending
    row["TotalSpending"] = sum(row[c] for c in SPENDING_COLS)
    row["NoSpending"]    = 1 if row["TotalSpending"] == 0 else 0

    # Numeric non-spending
    for col in ["Age", "CabinNumber", "GroupSize"]:
        val = user_input.get(col, None)
        row[col] = float(val) if val is not None else np.nan

    # Object-dtype categoricals — encode same as training
    for col in OBJECT_CAT_COLS:
        val = str(user_input.get(col, ""))
        known = encoder_categories.get(col, [])
        row[col] = int(encoders[col].transform([val])[0]) if val in known else -1

    # Bool columns — pass through as 0/1 matching notebook behaviour
    row["CryoSleep"] = 1 if str(user_input.get("CryoSleep", "False")).lower() == "true" else 0
    row["VIP"]       = 1 if str(user_input.get("VIP", "False")).lower() == "true" else 0

    # Build DataFrame in the column order the scaler was fitted on
    row_df = pd.DataFrame([{col: row.get(col, np.nan) for col in all_cols}])

    # Scale
    row_scaled = scaler.transform(row_df)

    # Post-scale NaN fill (Age, CabinNumber) — mirror training step 8
    row_scaled = np.where(np.isnan(row_scaled), 0.0, row_scaled)

    # Reorder to FEATURE_COLS
    col_idx = {c: i for i, c in enumerate(all_cols)}
    final = np.array([[row_scaled[0, col_idx[c]] for c in FEATURE_COLS]])
    return final
