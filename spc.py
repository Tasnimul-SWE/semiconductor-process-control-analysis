import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def limits_from_baseline(values):
    x = values.dropna().astype(float)
    mr = x.diff().abs().dropna()
    if len(x) < 30 or mr.empty:
        return None
    sigma = mr.mean() / 1.128
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    center = x.mean()
    return center, center - 3 * sigma, center + 3 * sigma


def run(baseline, monitor, sensors, output):
    output.mkdir(parents=True, exist_ok=True)
    all_alarms = monitor[["timestamp", "status"]].copy()
    rows = []
    for sensor in sensors:
        limits = limits_from_baseline(baseline[sensor])
        if limits is None:
            continue
        center, lcl, ucl = limits
        signal = monitor[sensor].notna() & ((monitor[sensor] < lcl) | (monitor[sensor] > ucl))
        all_alarms[sensor + "_alarm"] = signal.to_numpy()
        rows.append({"sensor": sensor, "center": center, "lcl": lcl, "ucl": ucl,
                     "monitoring_count": int(monitor[sensor].count()),
                     "alarms": int(signal.sum()),
                     "alarm_rate": float(signal.sum() / max(monitor[sensor].count(), 1))})

        values = pd.concat([baseline[sensor], monitor[sensor]], ignore_index=True)
        loc = np.arange(len(values))
        abnormal = values.notna() & ((values < lcl) | (values > ucl))
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(loc, values, ".", markersize=3, alpha=0.6)
        ax.scatter(loc[abnormal], values[abnormal], s=13, color="red", label="Outside limits")
        ax.axhline(center, color="black", linewidth=1)
        ax.axhline(lcl, color="red", linestyle="--")
        ax.axhline(ucl, color="red", linestyle="--")
        ax.axvline(len(baseline) - 0.5, color="gray", linestyle=":")
        ax.set_title(f"Individuals chart: {sensor}")
        ax.set_xlabel("Record order by timestamp")
        ax.set_ylabel("Sensor measurement")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{sensor}_spc.png", dpi=150)
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "control_limits.csv", index=False)
    alarm_cols = [c for c in all_alarms if c.endswith("_alarm")]
    all_alarms["any_alarm"] = all_alarms[alarm_cols].any(axis=1) if alarm_cols else False
    all_alarms.to_csv(output / "monitoring_alarms.csv", index=False)
    pd.crosstab(all_alarms["any_alarm"], all_alarms["status"]).to_csv(output / "alarms_vs_outcomes.csv")
    return summary, all_alarms
