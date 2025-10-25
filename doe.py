import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def run(output, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for temperature in (-1, 1):
        for pressure in (-1, 1):
            for repeat in range(8):
                noise = rng.normal(0, 0.7)
                response = 12 - 1.8 * temperature + 1.1 * pressure + 1.4 * temperature * pressure + noise
                rows.append((temperature, pressure, repeat + 1, response))
    df = pd.DataFrame(rows, columns=["temperature_code", "pressure_code", "replicate", "defect_score"])
    df.to_csv(output / "simulated_doe_data.csv", index=False)

    model = smf.ols("defect_score ~ C(temperature_code) * C(pressure_code)", data=df).fit()
    anova_lm(model, typ=2).to_csv(output / "doe_anova.csv")
    with open(output / "doe_regression.txt", "w", encoding="utf-8") as handle:
        handle.write(model.summary().as_text())

    means = df.groupby(["temperature_code", "pressure_code"]).defect_score.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    for pressure in (-1, 1):
        ax.plot([-1, 1], means[pressure].to_numpy(), marker="o", label=f"Pressure {pressure:+d}")
    ax.set_xticks([-1, 1])
    ax.set_xlabel("Temperature (coded low/high)")
    ax.set_ylabel("Mean simulated defect score")
    ax.set_title("2 × 2 factorial design: interaction plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "doe_interaction.png", dpi=150)
    plt.close(fig)
    return model
