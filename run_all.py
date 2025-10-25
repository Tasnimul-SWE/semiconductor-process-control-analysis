from pathlib import Path
import argparse

from data_loader import load_data, select_sensors, split_data
import explore
import spc
import capability
import defects
import doe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--specs", type=Path, default=None)
    args = parser.parse_args()

    output = args.output_dir
    data = load_data(args.data_dir)
    baseline, monitor = split_data(data)
    sensors = select_sensors(baseline)
    print(f"Records: {len(data)} | Sensors: {len([c for c in data if c.startswith('sensor_')])}")
    print(f"Pass: {(data.status == -1).sum()} | Fail: {(data.status == 1).sum()}")
    print("Monitoring sensors:", ", ".join(sensors))

    explore.run(data, sensors, output / "01_exploration")
    spc.run(baseline, monitor, sensors, output / "02_spc")
    capability.run(baseline, monitor, output / "03_capability", args.specs)
    defects.run(baseline, monitor, output / "04_defects")
    doe.run(output / "05_simulated_doe")
    print("Results saved in", output.resolve())


if __name__ == "__main__":
    main()
