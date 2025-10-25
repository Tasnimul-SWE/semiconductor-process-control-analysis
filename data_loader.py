from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"


def load_data(folder=Path("data")):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("secom.data", "secom_labels.data"):
        path = folder / filename
        if not path.exists():
            print("Downloading", filename)
            urlretrieve(BASE_URL + filename, path)

    x = pd.read_csv(folder / "secom.data", sep=r"\s+", header=None)
    x.columns = [f"sensor_{i+1:03d}" for i in range(x.shape[1])]
    y = pd.read_csv(folder / "secom_labels.data", sep=r"\s+", header=None,
                    names=["status", "date", "time"])
    y["timestamp"] = pd.to_datetime(y["date"] + " " + y["time"], errors="coerce")
    if len(x) != len(y) or y["timestamp"].isna().any():
        raise ValueError("Check data rows and timestamp format")
    if not set(y["status"].unique()).issubset({-1, 1}):
        raise ValueError("Unexpected pass/fail codes")
    df = pd.concat([y[["timestamp", "status"]], x], axis=1)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def split_data(df, fraction=0.6):
    n = int(len(df) * fraction)
    if n < 25 or len(df) - n < 10:
        raise ValueError("Not enough records for baseline and monitoring")
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def select_sensors(baseline, n=8):
    sensors = [c for c in baseline if c.startswith("sensor_")]
    missing = baseline[sensors].isna().mean()
    variance = baseline[sensors].var()
    valid = [s for s in sensors if missing[s] <= 0.05 and variance[s] > 0 and baseline[s].count() >= 30]
    return sorted(valid, key=lambda s: (missing[s], -baseline[s].count(), s))[:n]
