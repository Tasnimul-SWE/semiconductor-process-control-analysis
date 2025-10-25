# Semiconductor Process Control Analysis

Python analysis of the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom), with process monitoring, data quality analysis, exploratory defect modeling, and a separate simulated factorial experiment.

## Files

- `data_loader.py`: downloads and loads the manufacturing records; chronological baseline/monitoring split
- `explore.py`: missing values, status counts, sensor correlations
- `spc.py`: individuals charts with moving-range-based 3-sigma limits, plus alarms in the monitoring period
- `capability.py`: Cp and Cpk **only when externally justified engineering specifications are provided**
- `defects.py`: exploratory pass/fail random forest with time-ordered holdout, fit-on-training imputation
- `doe.py`: **synthetic** 2x2 factorial design and interaction analysis for learning DOE
- `run_all.py`: runs the complete workflow

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

The program downloads `secom.data` and `secom_labels.data` the first time. If downloading fails, download them manually from UCI and place them in `data/`.

To calculate capability indices **only if you have defensible engineering specification limits**, create a CSV such as:

```csv
sensor,lsl,usl
sensor_001,LOWER_ENGINEERING_LIMIT,UPPER_ENGINEERING_LIMIT
```

Replace the placeholders with real numeric values and run `python run_all.py --specs specs.csv`. SECOM does **not** provide engineering specification limits, so Cp/Cpk are skipped by default. Do not derive specification limits from the data and present them as actual process specs.

## Interpretation and limits

- The SECOM manufacturing sensor names are anonymized. Important features are **associations, not proven physical root causes**.
- SPC limits are **exploratory** and based on the first 60% of timestamp-sorted records. They are not factory-qualified limits. A chart alarm does not necessarily mean a failed chip.
- Individuals-chart assumptions may not hold (e.g., autocorrelation and product mix); review sequential process stability before operational use.
- The defect model is exploratory. Its 40% holdout is later in time than the training period; any model performance requires careful interpretation due to imbalanced outcomes.
- The DOE example uses **generated data**, not real Intel or SECOM experimental runs. Temperature and pressure are fictional labels in this educational simulation. It demonstrates factorial design and interactions, not hands-on semiconductor fabrication.
- This project does not claim fab access, process changes, or production yield improvements.

## Dataset citation

McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305. Licensed CC BY 4.0.
