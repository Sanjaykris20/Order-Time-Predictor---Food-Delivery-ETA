#!/usr/bin/env python
# coding: utf-8
"""
Robust, single-file script for Food Delivery Time prediction.
Improvements:
- CLI flags for iterations / learning rate / quick-mode
- NaN-safe bucketing for Delivery_Time to avoid singleton classes
- Safe categorical handling (convert to 'category' dtype and pass names to CatBoost)
- Guarded CatBoost import and clear errors
- Better logging and quick-run options
"""

import argparse
import datetime
import warnings
from time import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -------------------------------
# Utilities
# -------------------------------

def bucket_delivery_time_safe(x):
    # Map to reasonable buckets; handle NaN
    try:
        if pd.isna(x):
            return 30  # default bucket for missing
        v = float(x)
    except Exception:
        return 30
    if v <= 20:
        return 20
    if v <= 25:
        return 25
    if v <= 30:
        return 30
    if v <= 35:
        return 35
    if v <= 40:
        return 40
    return 45


# -------------------------------
# Load / Preprocess
# -------------------------------

def load_data(train_path, test_path):
    print("Loading data...")
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)
    data = pd.concat([train, test], ignore_index=True, sort=False)
    print(f"Shapes -> train: {train.shape}, test: {test.shape}, combined: {data.shape}")
    return data, train.shape[0]


def preprocess(data):
    print("Preprocessing...")
    data = data.copy()

    # Cuisines tidy
    data["Cuisines"] = (
        data["Cuisines"].fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
    )
    small_map = {
        'burger': 'fastfood', 'pizza': 'fastfood', 'wraps': 'fastfood',
        'bakery': 'desserts', 'icecream': 'desserts',
        'italian': 'european', 'french': 'european',
        'mexican': 'american', 'bbq': 'american',
        'arabian': 'middleeast', 'kebab': 'middleeast',
        'chinese': 'chinese', 'salad': 'healthyfood'
    }
    for k, v in small_map.items():
        data["Cuisines"] = data["Cuisines"].str.replace(k, v, regex=False)

    # Delivery time
    data["Delivery_Time"] = (
        data["Delivery_Time"].astype(str).str.replace(" minutes", "", regex=False)
    )
    data["Delivery_Time"] = pd.to_numeric(data["Delivery_Time"], errors="coerce")

    # Location split
    data["Location"] = data.get("Location", "").fillna("").astype(str)
    parts = data["Location"].str.rpartition(",")
    data["Locality"] = parts[0].str.lower().str.strip()
    data["City"] = parts[2].str.lower().str.strip()
    data.drop(columns=["Location"], inplace=True, errors='ignore')

    # Numeric fields
    for col in ["Average_Cost", "Minimum_Order"]:
        data[col] = data.get(col, "").astype(str).str.replace("[^0-9]", "", regex=True)
        data[col] = pd.to_numeric(data[col].str.strip(), errors="coerce")

    # Ratings / Votes / Reviews
    data["Rating"] = data.get("Rating", "").replace(['NEW', '-', 'Opening Soon', 'Temporarily Closed'], pd.NA)
    data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")
    data["Votes"] = data.get("Votes", "").replace('-', pd.NA)
    data["Votes"] = pd.to_numeric(data["Votes"], errors="coerce")
    data["Reviews"] = data.get("Reviews", "").replace('-', pd.NA)
    data["Reviews"] = pd.to_numeric(data["Reviews"], errors="coerce")

    # Feature engineering
    def safe_div(a, b):
        return a / b.replace({0: pd.NA})

    data["votes_per_review"] = safe_div(data["Votes"], data["Reviews"])
    data["votes_x_rating"] = data["Votes"] * data["Rating"]
    data["cost_ratio"] = safe_div(data["Average_Cost"], data["Minimum_Order"])

    return data


# -------------------------------
# Train & predict
# -------------------------------

def train_and_predict(data, n_train_rows, n_folds=5, random_state=42, output_path="cb_output.xlsx", iterations=3000, learning_rate=0.05, quick=False):
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        print("ERROR: catboost is required but not installed. Install with 'pip install catboost'.")
        raise

    from sklearn.model_selection import StratifiedKFold
    from scipy import stats

    print("Preparing train/test splits...")
    train_x = data.iloc[:n_train_rows].copy()
    test_x = data.iloc[n_train_rows:].copy()

    # Bucket delivery time (safe)
    train_y = train_x["Delivery_Time"].apply(bucket_delivery_time_safe)

    train_x.drop(columns=["Delivery_Time"], inplace=True, errors='ignore')
    test_x.drop(columns=["Delivery_Time"], inplace=True, errors='ignore')

    # Categorical handling: convert object columns to category and pass names to CatBoost
    cat_cols = train_x.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        train_x[c] = train_x[c].astype('category')
        test_x[c] = test_x[c].astype('category')

    print("Categorical columns:", cat_cols)

    # Quick mode: reduce iterations for fast testing
    if quick:
        iterations = min(iterations, 500)
        learning_rate = max(learning_rate, 0.1)

    base_model_params = dict(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=False
    )

    print(f"Training with iterations={iterations}, lr={learning_rate}, folds={n_folds}")

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models = []
    start = time()

    for fold, (tr_idx, val_idx) in enumerate(folds.split(train_x, train_y), 1):
        print(f"Fold {fold} / {n_folds}")
        model = CatBoostClassifier(**base_model_params)
        try:
            model.fit(
                train_x.iloc[tr_idx],
                train_y.iloc[tr_idx],
                cat_features=cat_cols,
                eval_set=(train_x.iloc[val_idx], train_y.iloc[val_idx]),
                early_stopping_rounds=100,
                verbose=200
            )
        except Exception as ex:
            print(f"ERROR in fold {fold} during fit: {ex}")
            print("Train dtypes:\n", train_x.dtypes)
            print("Train shape:", train_x.shape)
            raise
        models.append(model)

    print("Training finished in", str(datetime.timedelta(seconds=time() - start)))

    # Predict with majority voting
    print("Predicting test set with majority voting...")
    preds = [m.predict(test_x) for m in models]
    preds = np.stack([p.reshape(-1,) for p in preds], axis=1)
    final_preds = stats.mode(preds, axis=1)[0].reshape(-1)

    out = pd.DataFrame({"Delivery_Time": final_preds.astype(int).astype(str) + " minutes"})
    out.to_excel(output_path, index=False)
    print("Predictions saved to:", output_path)

    # Feature importance
    try:
        feat_imps = sorted(zip(models[-1].get_feature_importance(), train_x.columns), reverse=True)
        print("Top 10 feature importances:")
        for imp, col in feat_imps[:10]:
            print(f"  {col}: {imp:.4f}")
    except Exception:
        pass


# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Food Delivery Time training script")
    parser.add_argument("--train", type=str, default="Data_Train.xlsx", help="Train Excel file path")
    parser.add_argument("--test", type=str, default="Data_Test.xlsx", help="Test Excel file path")
    parser.add_argument("--output", type=str, default="cb_output.xlsx", help="Output Excel file (predictions)")
    parser.add_argument("--folds", type=int, default=5, help="Number of StratifiedKFold splits")
    parser.add_argument("--iterations", type=int, default=3000, help="CatBoost iterations")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="CatBoost learning rate")
    parser.add_argument("--quick", action='store_true', help="Quick mode: use fewer iterations for fast testing")
    parser.add_argument("--dry-run", action="store_true", help="Only run preprocessing and exit")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data, n_train = load_data(args.train, args.test)
    data = preprocess(data)
    if args.dry_run:
        print("Dry run complete. Preprocessed data shape:", data.shape)
        return

    train_and_predict(
        data,
        n_train,
        n_folds=args.folds,
        random_state=args.random_state,
        output_path=args.output,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        quick=args.quick
    )


if __name__ == "__main__":
    main()
