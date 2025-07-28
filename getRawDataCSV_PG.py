import os
import json
import pandas as pd

target_metrics = [
    "Mean Precision",
    "Mean Recall",
    "mAP@0.50",
    "mAP@0.75",
    "mAP@0.5:0.95"
]

poisson_gaussian_cfg = {
    "path": r"C:\Users\Admin\source\repos\Bachelor\Noise Simulator\output\poisson_gaussian",
    "brightness_levels": [1, 2, 3, 4, 5],
    "gaussian_levels": [1, 2, 3, 4, 5]
}

brightness_mapping = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.1}
gaussian_mapping = {1: 2.5, 2: 5, 3: 10, 4: 15, 5: 22.5}

baseline_json = [
  {"Metric": "Mean Precision", "Value": 0.663869301},
  {"Metric": "Mean Recall", "Value": 0.5269839312},
  {"Metric": "mAP@0.50", "Value": 0.5682071662},
  {"Metric": "mAP@0.75", "Value": 0.4242812757},
  {"Metric": "mAP@0.5:0.95", "Value": 0.3991250608}
]
baseline_map = {e["Metric"]: e["Value"] for e in baseline_json}

def load_metric_from_file(fp, metric):
    try:
        with open(fp, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    if isinstance(data, list):
        for e in data:
            if e.get("Metric") == metric:
                return e.get("Value")
        return None
    return data.get(metric)

records = []

for metric in target_metrics:
    records.append({
        "Metric": metric,
        "Noise Type": "Poisson-Gaussian",
        "Seed": 0,
        "Value": baseline_map[metric],
        "Brightness": 1,
        "Gaussian Level": 0
    })

for metric in target_metrics:
    for dirpath, _, files in os.walk(poisson_gaussian_cfg["path"]):
        if "yolo12_validation_metrics.json" not in files:
            continue

        parts = dirpath.split(os.sep)
        try:
            seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
            brightness_raw = int([p for p in parts if p.startswith("brightness_")][0].split("_")[1])
            gaussian_raw = int([p for p in parts if p.startswith("gaussian_")][0].split("_")[1])

            brightness = brightness_mapping[brightness_raw]
            gaussian = gaussian_mapping[gaussian_raw]

        except (IndexError, ValueError):
            continue

        fp = os.path.join(dirpath, "yolo12_validation_metrics.json")
        val = load_metric_from_file(fp, metric)
        if val is None:
            continue

        records.append({
            "Metric": metric,
            "Noise Type": "Poisson-Gaussian",
            "Brightness": brightness,
            "Gaussian Level": gaussian,
            "Seed": seed,
            "Value": val,
        })

df = pd.DataFrame(records)
out_path = r"C:\Users\Admin\source\repos\Bachelor\Visualizations\Performance\poisson_gaussian\poisson_gaussian_raw_metrics.csv"
df.to_csv(out_path, index=False)