"""Lab 1: clean -> split -> impute/scale/encode -> save arrays, transformers, metadata."""
import json, re
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import *

TEST_SIZE = 0.2


def parse_weight_kg(x):
    """'60 kg' -> 60, '2 lbs' -> 0.907, '1' -> 1 (assume kg)."""
    if pd.isna(x):
        return np.nan
    m = re.search(r"([\d.]+)", str(x))
    if not m:
        return np.nan
    v = float(m.group(1))
    return v * 0.4536 if "lb" in str(x).lower() else v


def parse_year(x):
    m = re.search(r"(20\d{2}|19\d{2})", str(x))
    return float(m.group(1)) if m else np.nan


def clean(df):
    df = df.copy()
    n0 = len(df)
    df = df[df[TARGET] > 0].reset_index(drop=True)          # a 0-point lot is a data error
    df["Bag.Weight.kg"] = df["Bag.Weight"].map(parse_weight_kg).clip(upper=1000)
    df["Harvest.Year.num"] = df["Harvest.Year"].map(parse_year)
    df["Number.of.Bags"] = df["Number.of.Bags"].clip(upper=df["Number.of.Bags"].quantile(0.99))
    df.loc[df["altitude_mean_meters"] > 5000, "altitude_mean_meters"] = np.nan  # impossible values
    df.loc[df["Moisture"] > 0.5, "Moisture"] = np.nan
    for c in CATEGORICAL:
        df[c] = df[c].astype("object").where(df[c].notna(), "Unknown").astype(str).str.strip()
    print(f"Cleaning: {n0} -> {len(df)} rows")
    return df


def main():
    PROC.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    df = clean(pd.read_csv(RAW))
    y = df[TARGET].values.astype(float)
    bins = pd.qcut(y, 5, labels=False, duplicates="drop")   # stratify a continuous target via quantile bins
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                                      random_state=SEED, stratify=bins)
    tr, te = df.iloc[idx_tr], df.iloc[idx_te]

    imputer = SimpleImputer(strategy="median").fit(tr[NUMERIC])
    scaler = StandardScaler().fit(imputer.transform(tr[NUMERIC]))
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                        sparse_output=False).fit(tr[CATEGORICAL])

    def transform(d):
        num = scaler.transform(imputer.transform(d[NUMERIC]))
        return np.hstack([num, ohe.transform(d[CATEGORICAL])])

    Xtr, Xte = transform(tr), transform(te)
    np.save(PROC / "X_train_final.npy", Xtr); np.save(PROC / "X_test_final.npy", Xte)
    np.save(PROC / "y_train.npy", y[idx_tr]); np.save(PROC / "y_test.npy", y[idx_te])
    te.to_csv(PROC / "test_rows_clean.csv", index=False)     # for error analysis
    joblib.dump(imputer, MODELS / "imputer.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(ohe, MODELS / "ohe.pkl")

    feature_names = NUMERIC + list(ohe.get_feature_names_out(CATEGORICAL))
    meta = {
        "dataset": "Coffee Quality Institute (CQI) arabica + robusta reviews",
        "source": "https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi",
        "task": "regression", "target": TARGET,
        "raw_rows": int(pd.read_csv(RAW).shape[0]), "clean_rows": int(len(df)),
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "test_size": TEST_SIZE, "random_state": SEED, "split": "stratified on target quintiles",
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "excluded_sensory_features_(leakage)": SENSORY,
        "n_features_after_encoding": int(Xtr.shape[1]), "feature_names": feature_names,
        "X_train_shape": list(Xtr.shape), "X_test_shape": list(Xte.shape),
        "target_mean_train": float(y[idx_tr].mean()), "target_std_train": float(y[idx_tr].std()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (PROC / "dataset_metadata.json").write_text(json.dumps(meta, indent=2))
    print("X_train", Xtr.shape, "X_test", Xte.shape, "| features:", Xtr.shape[1])


if __name__ == "__main__":
    main()
