"""Monthly, point-in-time RFM prediction. Target 1 means no next-month purchase."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import urllib.request
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
FEATURES = ["Recency", "Frequency", "Monetary"]
TARGET = "NoPurchaseNextMonth"
# Explicit source coverage, not inferred from individual customers' transactions.
COVERAGE_START = pd.Timestamp("2010-12-01")
COVERAGE_END = pd.Timestamp("2011-12-10")  # exclusive; December is incomplete


def download_data(path: Path) -> Path:
    """Cache the original UCI workbook without committing it to GitHub."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / "online-retail.zip"
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as z:
        with z.open("Online Retail.xlsx") as source, path.open("wb") as dest:
            import shutil
            shutil.copyfileobj(source, dest)
    return path


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Use positive purchases; gross spend excludes refunds/cancellations."""
    columns = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = raw.drop_duplicates().copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=columns)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.loc[(df.Quantity > 0) & (df.UnitPrice > 0)
                & np.isfinite(df.Quantity) & np.isfinite(df.UnitPrice)
                & ~df.InvoiceNo.str.upper().str.startswith("C")].copy()
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["TotalPrice"] = df.Quantity * df.UnitPrice
    df = df.loc[np.isfinite(df.TotalPrice)]
    return df.sort_values("InvoiceDate").reset_index(drop=True)


def month_start(value) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if date != date.to_period("M").start_time:
        raise ValueError("Cutoff must be midnight on the first day of a month")
    return date


def build_features(transactions: pd.DataFrame, cutoff, months: int = 3) -> pd.DataFrame:
    """Only transactions in [cutoff - 3 calendar months, cutoff) are accessible.

    Eligible customers bought at least once in this window. IDs and cutoff are
    bookkeeping only and never model inputs. The same customer may recur monthly.
    """
    cutoff = month_start(cutoff)
    if months < 1:
        raise ValueError("months must be positive")
    window_start = cutoff - pd.DateOffset(months=months)
    past = transactions.loc[(transactions.InvoiceDate >= window_start)
                            & (transactions.InvoiceDate < cutoff)]
    result = past.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    # Compare calendar dates: an order on the previous day has Recency=1.
    result["Recency"] = (cutoff - result.LastPurchase.dt.normalize()).dt.days
    result = result.drop(columns="LastPurchase").reset_index()
    result["Cutoff"] = cutoff
    result["ObservationStart"] = window_start
    return result[["CustomerID", "Cutoff", "ObservationStart"] + FEATURES]


def build_samples(transactions: pd.DataFrame, cutoffs, coverage_start=COVERAGE_START,
                  coverage_end=COVERAGE_END) -> pd.DataFrame:
    """Construct labels only for fully observed future calendar months."""
    frames = []
    for value in cutoffs:
        cutoff = month_start(value)
        label_end = cutoff + pd.DateOffset(months=1)
        if cutoff - pd.DateOffset(months=3) < pd.Timestamp(coverage_start):
            raise ValueError("Incomplete observation window")
        if label_end > pd.Timestamp(coverage_end):
            raise ValueError("Incomplete future label window; do not label it as no purchase")
        features = build_features(transactions, cutoff)
        future = transactions.loc[(transactions.InvoiceDate >= cutoff)
                                  & (transactions.InvoiceDate < label_end)]
        buyers = set(future.CustomerID)
        features[TARGET] = (~features.CustomerID.isin(buyers)).astype(int)
        features["LabelEnd"] = label_end
        frames.append(features)
    if not frames:
        raise ValueError("No sample cutoffs supplied")
    return pd.concat(frames, ignore_index=True).sort_values(["Cutoff", "CustomerID"])


def temporal_split(samples: pd.DataFrame):
    train = samples.loc[samples.Cutoff < pd.Timestamp("2011-08-01")].copy()
    valid = samples.loc[samples.Cutoff == pd.Timestamp("2011-08-01")].copy()
    test = samples.loc[samples.Cutoff >= pd.Timestamp("2011-09-01")].copy()
    for part in [train, valid, test]:
        if part.empty or part[TARGET].nunique() != 2:
            raise ValueError("Each split must have data and both target classes")
    if train.LabelEnd.max() > valid.Cutoff.min() or valid.LabelEnd.max() > test.Cutoff.min():
        raise ValueError("Training/validation labels were unavailable at the next split cutoff")
    return train, valid, test


def make_model(c: float = 1.0, recency_only: bool = False) -> Pipeline:
    transformers = [("recency", "passthrough", ["Recency"])]
    if not recency_only:
        transformers.append(("log_fm", FunctionTransformer(np.log1p), ["Frequency", "Monetary"]))
    return Pipeline([
        ("features", ColumnTransformer(transformers, remainder="drop")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def select_threshold(y, probabilities):
    """Illustrative classification threshold chosen on validation only."""
    candidates = np.arange(0.05, 0.951, 0.01)
    scores = [f1_score(y, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(y, probabilities, threshold):
    predictions = np.asarray(probabilities) >= threshold
    return {"rows": len(y), "positive_rate": float(np.mean(y)), "threshold": threshold,
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "average_precision": float(average_precision_score(y, probabilities)),
            "brier_score": float(brier_score_loss(y, probabilities)),
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist()}


def rank_by_month(scored: pd.DataFrame, fraction: float = 0.2):
    """Fixed monthly budget; precision/lift are about risk, not coupon effectiveness."""
    rows = []
    for cutoff, group in scored.groupby("Cutoff", sort=True):
        k = max(1, math.ceil(len(group) * fraction))
        top = group.sort_values(["NoPurchaseProbability", "CustomerID"], ascending=[False, True]).head(k)
        rate = float(group[TARGET].mean())
        precision = float(top[TARGET].mean())
        rows.append({"month": cutoff.strftime("%Y-%m"), "eligible_customers": len(group),
                     "contact_budget": k, "positive_rate": rate,
                     "precision_at_20pct": precision,
                     "recall_at_20pct": float(top[TARGET].sum() / group[TARGET].sum()),
                     "lift_at_20pct": precision / rate})
    return pd.DataFrame(rows)


def run_project(data_path: Path, output: Path = ROOT / "output", images: Path = ROOT / "images"):
    output, images = Path(output), Path(images)
    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(data_path)
    transactions = clean_transactions(raw)
    samples = build_samples(transactions, pd.date_range("2011-03-01", "2011-11-01", freq="MS"))
    train, valid, test = temporal_split(samples)

    candidates, fitted = [], {}
    for c in [0.1, 1.0, 10.0]:
        model = make_model(c).fit(train[FEATURES], train[TARGET])
        p = model.predict_proba(valid[FEATURES])[:, 1]
        candidates.append({"C": c, "validation_roc_auc": roc_auc_score(valid[TARGET], p),
                           "validation_average_precision": average_precision_score(valid[TARGET], p)})
        fitted[c] = model
    selection = pd.DataFrame(candidates)
    best_c = float(selection.sort_values(["validation_roc_auc", "C"], ascending=[False, True]).iloc[0].C)
    model = fitted[best_c]  # frozen: test data never fit the evaluation model
    threshold = select_threshold(valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]

    baseline = make_model(recency_only=True).fit(train[FEATURES], train[TARGET])
    baseline_threshold = select_threshold(valid[TARGET], baseline.predict_proba(valid[FEATURES])[:, 1])
    dummy = DummyClassifier(strategy="prior").fit(train[FEATURES], train[TARGET])
    dummy_threshold = select_threshold(valid[TARGET], dummy.predict_proba(valid[FEATURES])[:, 1])
    results = {"rfm_logistic": evaluate(test[TARGET], probabilities, threshold),
               "recency_logistic": evaluate(test[TARGET], baseline.predict_proba(test[FEATURES])[:, 1], baseline_threshold),
               "constant_prior": evaluate(test[TARGET], dummy.predict_proba(test[FEATURES])[:, 1], dummy_threshold)}
    scored = test.copy()
    scored["NoPurchaseProbability"] = probabilities
    monthly = rank_by_month(scored)
    month_metrics = {cutoff.strftime("%Y-%m"): evaluate(g[TARGET], g.NoPurchaseProbability, threshold)
                     for cutoff, g in scored.groupby("Cutoff")}

    # Separate deployment artifact: all labels up to Nov 30 are known at Dec 1.
    deployment = make_model(best_c).fit(samples[FEATURES], samples[TARGET])
    forecast = build_features(transactions, "2011-12-01")
    forecast["PredictionMonth"] = "2011-12"
    forecast["NoPurchaseProbability"] = deployment.predict_proba(forecast[FEATURES])[:, 1]
    forecast = forecast.sort_values(["NoPurchaseProbability", "CustomerID"], ascending=[False, True])
    top = forecast.head(math.ceil(len(forecast) * 0.2))

    split_summary = []
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        split_summary.append({"split": name, "prediction_months": sorted(part.Cutoff.dt.strftime("%Y-%m").unique().tolist()),
                              "rows": len(part), "unique_customers": part.CustomerID.nunique(),
                              "positive_rate": float(part[TARGET].mean())})
    report = {"target": TARGET, "target_definition": "1=no valid purchase in next calendar month, 0=at least one",
              "observation_months": 3, "eligible_population": "customers with >=1 valid purchase in observation window",
              "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
              "raw_rows": len(raw), "clean_rows": len(transactions),
              "clean_unique_customers": transactions.CustomerID.nunique(),
              "coverage_start": str(COVERAGE_START.date()), "coverage_end_exclusive": str(COVERAGE_END.date()),
              "splits": split_summary, "selected_C": best_c,
              "validation_candidates": selection.to_dict("records"), "test_results": results,
              "test_results_by_month": month_metrics,
              "forecast_month": "2011-12", "forecast_eligible_customers": len(forecast),
              "forecast_status": "unlabeled forecast; incomplete December cannot be evaluated",
              "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    monthly.to_csv(output / "monthly_top20_metrics.csv", index=False)
    scored.to_csv(output / "test_predictions.csv", index=False)
    selection.to_csv(output / "validation_selection.csv", index=False)
    forecast.to_csv(output / "december_forecast.csv", index=False)
    top.to_csv(output / "high_risk_users.csv", index=False)
    samples.to_csv(output / "user_month_samples.csv", index=False)
    joblib.dump({"pipeline": deployment, "features": FEATURES, "trained_label_end": "2011-12-01",
                 "evaluation_model_threshold": threshold, "note": "threshold validated on earlier frozen model; deployment prioritizes top 20%"},
                output / "repurchase_model.joblib")

    cm = results["rfm_logistic"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=15)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Purchase", "No purchase"],
           yticklabels=["Purchase", "No purchase"], xlabel="Predicted", ylabel="Actual",
           title="Out-of-time test: Sep-Nov 2011")
    fig.tight_layout(); fig.savefig(images / "confusion_matrix.png", dpi=150); plt.close(fig)
    coefficients = model.named_steps["model"].coef_[0]
    pd.DataFrame({"feature": ["Recency", "log1p(Frequency)", "log1p(Monetary)"],
                  "standardized_coefficient": coefficients}).to_csv(output / "coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.barh(["Recency", "log1p(Frequency)", "log1p(Monetary)"], coefficients)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Coefficient per standard deviation", title="Association with next-month non-purchase (not causal)")
    fig.tight_layout(); fig.savefig(images / "feature_importance.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(test[TARGET], probabilities, name="RFM logistic", ax=ax)
    RocCurveDisplay.from_predictions(test[TARGET], baseline.predict_proba(test[FEATURES])[:, 1], name="Recency only", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title("Out-of-time ROC: Sep-Nov 2011")
    fig.tight_layout(); fig.savefig(images / "roc_curve.png", dpi=150); plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Online Retail.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    path = download_data(args.data)
    report = run_project(path, args.output)
    print(json.dumps(report["test_results"], indent=2))


if __name__ == "__main__":
    main()
