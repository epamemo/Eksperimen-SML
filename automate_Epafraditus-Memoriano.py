"""Automated preprocessing for the Telco churn dataset (Kriteria 1 - Skilled/Advance).

Converts the manual experiment steps in the notebook into a reusable function that
returns train-ready data and (optionally) writes processed CSVs to disk.

Usage:
    python automate_Epafraditus-Memoriano.py \
        --input telco_churn_raw.csv \
        --outdir telco_churn_preprocessing

Or import it:
    from automate_Epafraditus_Memoriano import preprocess
    X_train, X_test, y_train, y_test = preprocess("telco_churn_raw.csv")
"""
import argparse
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET = "Churn"
DROP = ["customerID"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "TechSupport", "NumServices"]
CATEGORICAL = ["Contract", "PaymentMethod"]


def _build_transformer():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def preprocess(input_path, test_size=0.2, random_state=42, outdir=None):
    """Load raw CSV, clean/encode/scale, split, and return train-ready arrays.

    If outdir is given, also writes train.csv/test.csv (features + target) there.
    """
    df = pd.read_csv(input_path)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    ct = _build_transformer()
    X_train_t = ct.fit_transform(X_train)
    X_test_t = ct.transform(X_test)
    cols = list(ct.get_feature_names_out())

    X_train_df = pd.DataFrame(X_train_t, columns=cols)
    X_test_df = pd.DataFrame(X_test_t, columns=cols)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        train_out = X_train_df.copy()
        train_out[TARGET] = y_train.reset_index(drop=True)
        test_out = X_test_df.copy()
        test_out[TARGET] = y_test.reset_index(drop=True)
        train_out.to_csv(os.path.join(outdir, "train.csv"), index=False)
        test_out.to_csv(os.path.join(outdir, "test.csv"), index=False)
        print(f"Wrote processed data to {outdir}/ ({len(train_out)} train, {len(test_out)} test rows)")

    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Automated Telco churn preprocessing")
    ap.add_argument("--input", default="telco_churn_raw.csv")
    ap.add_argument("--outdir", default="telco_churn_preprocessing")
    args = ap.parse_args()
    preprocess(args.input, outdir=args.outdir)


if __name__ == "__main__":
    main()
