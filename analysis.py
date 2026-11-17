"""SECOM exploratory time-series analysis.

How to run:
  1. Download UCI SECOM dataset and place secom.data + secom_labels.data in data/raw/.
  2. python -m pip install -r requirements.txt
  3. python analysis.py

The timestamps refer to test results, not proven continuous readings from a device.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw"
IMAGES = BASE / "images"
RESULTS = BASE / "results"
WINDOW = 30  # 30 recorded products/tests; not 30 days


def load_secom() -> tuple[pd.DataFrame, list[str]]:
    sensor_file = DATA / "secom.data"
    label_file = DATA / "secom_labels.data"
    missing_files = [p.name for p in (sensor_file, label_file) if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing {missing_files} in {DATA}. Download the SECOM dataset from "
            "https://archive.ics.uci.edu/dataset/179/secom and unzip both files."
        )
    sensors = pd.read_csv(sensor_file, sep=r"\s+", header=None, na_values=["NaN"])
    names = [f"sensor_{i:03d}" for i in range(1, sensors.shape[1] + 1)]
    sensors.columns = names
    # Example label row: -1 "19/07/2008 11:55:00"
    labels = pd.read_csv(
        label_file,
        sep=r"\s+",
        header=None,
        names=["label", "timestamp_text"],
        quotechar='"',
        engine="c",
    )
    if len(sensors) != len(labels):
        raise ValueError(f"Unmatched row counts: sensors={len(sensors)}, labels={len(labels)}")
    if not labels["label"].isin([-1, 1]).all():
        raise ValueError("Unexpected label codes: expected -1=pass or 1=fail")
    labels["timestamp"] = pd.to_datetime(
        labels["timestamp_text"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    invalid_timestamps = int(labels["timestamp"].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"{invalid_timestamps} dates could not be parsed; check file format")

    # Join on original row position BEFORE sorting by time.
    df = pd.concat(
        [labels[["timestamp", "label"]].reset_index(drop=True),
         sensors.reset_index(drop=True)], axis=1
    )
    df["is_fail"] = (df["label"] == 1).astype(int)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df, names


def choose_sensor(df: pd.DataFrame, names: list[str]) -> str:
    """Transparent, label-independent choice: first well-observed variable.

    This choice is for demonstrating time-series methods, NOT identifying a
    sensor that causes defects. The anonymous sensor values have unknown units.
    """
    obs_fraction = df[names].notna().mean()
    unique_count = df[names].nunique(dropna=True)
    for completeness in (0.95, 0.80, 0.50):
        candidates = [
            name for name in names
            if obs_fraction[name] >= completeness and unique_count[name] >= 20
        ]
        if candidates:
            return candidates[0]
    raise ValueError("No sensor has enough non-missing varied values for plotting")


def save_chart(fig: plt.Figure, filename: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=0.2)


def analyze(df: pd.DataFrame, names: list[str]) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    signal = choose_sensor(df, names)
    x = df["timestamp"]
    y = df[signal]  # NaN is retained: no artificial interpolation of product records
    rolling_mean = y.rolling(WINDOW, min_periods=10).mean()
    rolling_sd = y.rolling(WINDOW, min_periods=10).std()

    # IQR flags candidates for investigation; these values are NOT removed.
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.notna(iqr) and iqr > 0:
        low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        outlier = ((y < low) | (y > high)).fillna(False)
    else:
        low = high = None
        outlier = pd.Series(False, index=df.index)

    weekly = (
        df.set_index("timestamp")
        .resample("W-SUN")
        .agg(sample_count=("is_fail", "size"), fail_count=("is_fail", "sum"))
    )
    weekly = weekly.loc[weekly["sample_count"] > 0].copy()
    weekly["fail_rate_pct"] = 100 * weekly["fail_count"] / weekly["sample_count"]
    weekly.to_csv(RESULTS / "weekly_quality.csv", encoding="utf-8-sig", index_label="week_ending")

    # Chart 1: time-sorted measurements and 30-record moving average
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, y, ".", ms=2, alpha=0.35, label="Original measurements")
    ax.plot(x, rolling_mean, linewidth=2, label=f"{WINDOW}-record moving average")
    ax.set(title=f"Chart 1: {signal} measurements over test timestamps", ylabel="Sensor value (unknown unit)", xlabel="Test date")
    ax.legend()
    time_axis(ax)
    save_chart(fig, "01_sensor_trend.png")

    # Chart 2: weekly failure rate with sample counts to avoid misleading ratios
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(weekly.index, weekly["fail_rate_pct"], width=5, alpha=0.85)
    for date, percent, n in zip(weekly.index, weekly["fail_rate_pct"], weekly["sample_count"]):
        ax.annotate(f"n={int(n)}", (date, percent), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=7, rotation=60)
    ax.set(title="Chart 2: Weekly failure rate (sample counts labeled)",
           ylabel="Failed tests (%)", xlabel="Week ending (Sunday)")
    ax.set_ylim(bottom=0)
    time_axis(ax)
    save_chart(fig, "02_weekly_fail_rate.png")

    # Chart 3: robust distributional anomaly candidates; not a formal SPC limit
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(x, y, s=9, alpha=0.35, label="Measurements")
    if bool(outlier.any()):
        ax.scatter(x.loc[outlier], y.loc[outlier], s=24, marker="x",
                   label=f"IQR outlier candidates ({int(outlier.sum())})")
    if low is not None:
        ax.axhline(low, ls="--", alpha=0.7, label="IQR bounds")
        ax.axhline(high, ls="--", alpha=0.7)
    ax.set(title=f"Chart 3: {signal} potential outliers (1.5 x IQR)",
           ylabel="Sensor value (unknown unit)", xlabel="Test date")
    ax.legend(fontsize=8)
    time_axis(ax)
    save_chart(fig, "03_sensor_outliers.png")

    # Chart 4: rolling volatility by observation count, not calendar units
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, rolling_sd, linewidth=1.8)
    ax.set(title=f"Chart 4: {WINDOW}-record rolling standard deviation of {signal}",
           ylabel="Rolling standard deviation", xlabel="Test date")
    time_axis(ax)
    save_chart(fig, "04_rolling_volatility.png")

    overall_missing = int(df[names].isna().sum().sum())
    total_cells = int(len(df) * len(names))
    stat = {
        "dataset": "UCI SECOM",
        "samples": len(df),
        "sensor_features": len(names),
        "start": x.min().isoformat(),
        "end": x.max().isoformat(),
        "pass_count": int((df["label"] == -1).sum()),
        "fail_count": int(df["is_fail"].sum()),
        "overall_fail_rate_pct": round(100 * df["is_fail"].mean(), 3),
        "sensor_missing_cells": overall_missing,
        "total_sensor_cells": total_cells,
        "sensor_missing_percent": round(100 * overall_missing / total_cells, 3),
        "selected_sensor": signal,
        "selected_sensor_missing": int(y.isna().sum()),
        "selected_sensor_mean": None if pd.isna(y.mean()) else round(float(y.mean()), 4),
        "selected_sensor_min": None if pd.isna(y.min()) else round(float(y.min()), 4),
        "selected_sensor_max": None if pd.isna(y.max()) else round(float(y.max()), 4),
        "iqr_lower": low,
        "iqr_upper": high,
        "outlier_count": int(outlier.sum()),
        "rolling_window_records": WINDOW,
        "weekly_buckets": len(weekly),
        "weekly_max_fail_rate_pct": round(float(weekly["fail_rate_pct"].max()), 3),
        "weekly_min_fail_rate_pct": round(float(weekly["fail_rate_pct"].min()), 3),
        "high_failure_week": str(weekly["fail_rate_pct"].idxmax().date()),
        "high_failure_week_n": int(weekly.loc[weekly["fail_rate_pct"].idxmax(), "sample_count"]),
    }
    (RESULTS / "summary.json").write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RESULT SUMMARY ===")
    print(json.dumps(stat, ensure_ascii=False, indent=2))
    print("\nGenerated images/01..04, results/summary.json, results/weekly_quality.csv")
    print("Open REPORT.md and replace all [직접 입력] markers with your verified interpretation.")
    return stat


if __name__ == "__main__":
    data, sensor_cols = load_secom()
    analyze(data, sensor_cols)
