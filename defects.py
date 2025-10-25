import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def run(baseline, monitor, output):
    output.mkdir(parents=True, exist_ok=True)
    cols = [c for c in baseline if c.startswith("sensor_")]
    valid = [c for c in cols if baseline[c].notna().mean() >= 0.8 and baseline[c].nunique() > 1]
    x_train, x_test = baseline[valid], monitor[valid]
    y_train, y_test = (baseline.status == 1).astype(int), (monitor.status == 1).astype(int)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        print("Skipping classification: both classes needed in each period.")
        return
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(x_train)
    test = imputer.transform(x_test)
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(train, y_train)
    predicted = model.predict(test)
    probability = model.predict_proba(test)[:, 1]
    metrics = pd.DataFrame([{
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "average_precision": average_precision_score(y_test, probability),
        "roc_auc": roc_auc_score(y_test, probability),
        "baseline_fail_fraction": y_train.mean(),
        "monitoring_fail_fraction": y_test.mean()
    }])
    metrics.to_csv(output / "defect_model_metrics.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, predicted),
                 index=["actual_pass", "actual_fail"],
                 columns=["pred_pass", "pred_fail"]).to_csv(output / "confusion_matrix.csv")
    feature_scores = pd.DataFrame({"sensor": valid, "importance": model.feature_importances_})
    feature_scores.sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    top = feature_scores.nlargest(15, "importance").sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.sensor, top.importance)
    ax.set_title("Exploratory model feature importance")
    ax.set_xlabel("Random forest importance")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=150)
    plt.close(fig)
    return metrics
