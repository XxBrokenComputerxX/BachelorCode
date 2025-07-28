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

base_dirs = {
    "Gaussian": {
        "path": r"C:\Users\Admin\source\repos\Bachelor\Noise Simulator\output\gaussian",
        "levels": [2.5, 5, 10, 15, 22.5]
    },
    "Salt-And-Pepper": {
        "path": r"C:\Users\Admin\source\repos\Bachelor\Noise Simulator\output\salt_pepper",
        "levels": [0.03, 0.06, 0.12, 0.18, 0.27]
    }
}

baseline_json = [
  {"Metric":"Mean Precision","Value":0.663869301},
  {"Metric":"Mean Recall","Value":0.5269839312},
  {"Metric":"mAP@0.50","Value":0.5682071662},
  {"Metric":"mAP@0.75","Value":0.4242812757},
  {"Metric":"mAP@0.5:0.95","Value":0.3991250608}
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
for noise_label, cfg in base_dirs.items():
    root, levels = cfg["path"], cfg["levels"]

    for metric in target_metrics:
        records.append({
            "Metric":          metric,
            "Noise Type":      noise_label,
            "Noise Intensity": 0.0,
            "Seed":            0,
            "Value":           baseline_map[metric]
        })

    for metric in target_metrics:
        by_level = {}
        for dirpath, _, files in os.walk(root):
            if "yolo12_validation_metrics.json" not in files:
                continue
            fp = os.path.join(dirpath, "yolo12_validation_metrics.json")
            val = load_metric_from_file(fp, metric)
            if val is None:
                continue

            parts = dirpath.split(os.sep)
            for i, p in enumerate(parts):
                if p.startswith("seed_"):
                    seed = int(p.split("_")[1])
                    level_name = parts[i - 1] 
                    by_level.setdefault(level_name, []).append((seed, val))
                    break

        for level_name, seed_vals in by_level.items():
            idx = int(level_name.split("_")[-1]) - 1
            intensity = levels[idx]
            for seed, val in seed_vals:
                records.append({
                    "Metric":          metric,
                    "Noise Type":      noise_label,
                    "Noise Intensity": intensity,
                    "Seed":            seed,
                    "Value":           val
                })

raw_df = pd.DataFrame(records)
out_path = r"C:\Users\Admin\source\repos\Bachelor\Visualizations\Performance\gaussian_salt_pepper\raw_metrics.csv"
raw_df.to_csv(out_path, index=False)