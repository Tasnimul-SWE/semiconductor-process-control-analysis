import argparse
import numpy as np
import pandas as pd


def calculate(values, lower, upper):
    x = pd.Series(values).dropna().astype(float)
    if len(x) < 30 or lower >= upper:
        return None
    sigma = x.std(ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    mean = x.mean()
    return {"mean": mean, "std": sigma, "lsl": lower, "usl": upper,
            "cp": (upper - lower) / (6 * sigma),
            "cpk": min((upper - mean) / (3 * sigma), (mean - lower) / (3 * sigma)),
            "fraction_out_of_spec": ((x < lower) | (x > upper)).mean()}


def run(baseline, monitor, output, specs_file=None):
    output.mkdir(parents=True, exist_ok=True)
    if specs_file is None:
        print("Skipping Cp/Cpk: no engineering specification limits supplied.")
        return pd.DataFrame()
    specs = pd.read_csv(specs_file)
    if not {"sensor", "lsl", "usl"}.issubset(specs.columns):
        raise ValueError("Specifications CSV needs sensor,lsl,usl")
    results = []
    for r in specs.itertuples(index=False):
        if r.sensor not in baseline.columns:
            raise ValueError(f"Unknown sensor in specification file: {r.sensor}")
        score = calculate(monitor[r.sensor], float(r.lsl), float(r.usl))
        if score:
            results.append({"sensor": r.sensor, **score})
    table = pd.DataFrame(results)
    table.to_csv(output / "capability.csv", index=False)
    return table
