import matplotlib.pyplot as plt
import pandas as pd


def run(df, sensors, output):
    output.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df if c.startswith("sensor_")]
    summary = pd.DataFrame({
        "missing_fraction": df[cols].isna().mean(),
        "mean": df[cols].mean(),
        "std": df[cols].std(),
        "unique_values": df[cols].nunique()
    })
    summary.index.name = "sensor"
    summary.to_csv(output / "sensor_quality.csv")
    df["status"].map({-1: "Pass", 1: "Fail"}).value_counts().to_csv(output / "status_counts.csv")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    df["status"].map({-1: "Pass", 1: "Fail"}).value_counts().reindex(["Pass", "Fail"]).plot.bar(ax=axes[0])
    axes[0].set_ylabel("Number of records")
    axes[0].set_title("Manufacturing outcomes")
    summary["missing_fraction"].plot.hist(ax=axes[1], bins=25)
    axes[1].set_xlabel("Missing fraction per sensor")
    axes[1].set_title("Sensor data quality")
    fig.tight_layout()
    fig.savefig(output / "dataset_overview.png", dpi=150)
    plt.close(fig)

    if sensors:
        data = df[sensors].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(sensors)), sensors, rotation=45, ha="right")
        ax.set_yticks(range(len(sensors)), sensors)
        fig.colorbar(im, ax=ax, label="Pearson correlation")
        fig.tight_layout()
        fig.savefig(output / "sensor_correlations.png", dpi=150)
        plt.close(fig)
    return summary
