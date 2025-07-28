import os 
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

target_metrics = [
    "mAP@0.50",
    "mAP@0.75",
    "mAP@0.5:0.95"
]

noise_type_german = {
    "salt_pepper": "Salz-und-Pfeffer Rauschen",
    "gaussian": "AWGN",
    "poisson_gaussian": "Poisson-Gaußschem Rauschen"
}

base_dirs = [
    (r"C:\Users\Admin\source\repos\Bachelor\Noise Simulator\output\gaussian", 1),
    (r"C:\Users\Admin\source\repos\Bachelor\Noise Simulator\output\salt_pepper", 1),
    (r"C:\Users\Admin\source\repos\Bachelor\Noise Simulator\output\poisson_gaussian", 2)
]

gaussian_noise_levels       = [2.5, 5, 10, 15, 22.5]
salt_pepper_noise_levels    = [0.03, 0.06, 0.12, 0.18, 0.27]
poisson_brightness_scales   = [0.8, 0.6, 0.4, 0.2, 0.1]
cv_threshold                = 1.0 

def sanitize_filename(metric_name):
    return metric_name.replace(":", "_")\
                      .replace("/", "_")\
                      .replace("\\", "_")\
                      .replace(" ", "_")

def load_metric_from_file(file_path, target_metric):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                if entry.get("Metric") == target_metric:
                    return entry.get("Value")
        return data.get(target_metric)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def collect_results_by_level_generic(root_dir, target_metric, nested_levels=1):
    results_by_level = {}
    per_seed_data    = {}
    for dirpath, _, _ in os.walk(root_dir):
        parts = dirpath[len(root_dir):].strip(os.sep).split(os.sep)
        if len(parts) != nested_levels + 1:
            continue
        *level_parts, seed_part = parts
        if not seed_part.startswith("seed_"):
            continue
        level_key   = "/".join(level_parts)
        metric_file = os.path.join(dirpath, "yolo12_val", "yolo12_validation_metrics.json")
        value       = load_metric_from_file(metric_file, target_metric)
        if value is not None:
            results_by_level.setdefault(level_key, []).append(value)
            per_seed_data   .setdefault(level_key, {})[seed_part] = value
    return results_by_level, per_seed_data

def compute_cv_from_seed_data(per_seed_data):
    cv_by_level = {}
    for level, seed_dict in per_seed_data.items():
        vals = list(seed_dict.values())
        if len(vals) < 2:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)
        if mean != 0:
            cv_by_level[level] = (std / mean) * 100
    return cv_by_level

def get_cv_results(base_dirs, target_metrics):
    cv_results = {}
    for metric in target_metrics:
        cv_results[metric] = {}
        for base_dir, depth in base_dirs:
            noise_type = os.path.basename(base_dir)
            _, per_seed = collect_results_by_level_generic(base_dir, metric, nested_levels=depth)
            cv_by_level = compute_cv_from_seed_data(per_seed)
            cv_results[metric][noise_type] = cv_by_level
    return cv_results

def plot_cv_poisson_gaussian_save(cv_results, metric, output_dir="plots"):
    data = cv_results[metric].get("poisson_gaussian", {})
    if not data:
        return

    rows, cols, vals = [], [], []
    for level, cv in data.items():
        try:
            brightness, sigma = level.split("/")
            bi = int(brightness.split("_")[1]) - 1
            gi = int(sigma.split("_")[1]) - 1
            rows.append(gaussian_noise_levels[gi])
            cols.append(poisson_brightness_scales[bi])
            vals.append(cv)
        except:
            continue

    df = pd.DataFrame({"σ": rows, "Helligkeit": cols, "CV": vals})
    pivot = df.pivot(index="σ", columns="Helligkeit", values="CV")

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': 'CV [%]'})
    plt.title(f"Seed-Variabilität bei {noise_type_german['poisson_gaussian']} ({metric})")
    plt.xlabel("Helligkeit")
    plt.ylabel("Standardabweichung σ")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    name = sanitize_filename(metric)
    plt.savefig(f"{output_dir}/cv_poisson_gaussian_{name}.png")
    plt.close()

def plot_combined_cv_1d(cv_results, noise_type, noise_levels_map, metrics,
                        output_dir="plots", threshold=1.0):
    plt.figure(figsize=(10,6))
    for metric in metrics:
        levels = cv_results[metric].get(noise_type, {})
        items  = sorted(levels.items(),
                        key=lambda x: int(x[0].split("_")[-1]))
        x_vals = [noise_levels_map[int(lvl.split("_")[-1]) - 1]
                  for lvl,_ in items]
        y_vals = [val for _,val in items]
        plt.plot(x_vals, y_vals, marker='o', label=metric)

    plt.axhline(y=threshold, color='gray', linestyle='--',
                label=f"{threshold:.1f}% Schwelle")
    label = " (σ)" if noise_type=="gaussian" else " (p)"
    plt.title(f"Seed-Variabilität (CV) bei {noise_type_german.get(noise_type, noise_type)}")
    plt.xlabel("Rauschintensität" + label)
    plt.ylabel("CV [%]")
    plt.ylim(0,5)
    plt.grid(True)
    plt.legend(title="Metric")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/cv_combined_{noise_type}.png")
    plt.close()

def save_cv_results_to_csv(cv_results, output_dir="cv_csv"):
    os.makedirs(output_dir, exist_ok=True)
    for metric, noise_data in cv_results.items():
        rows=[]
        for noise_type, lvl_data in noise_data.items():
            for lvl, cv in lvl_data.items():
                rows.append({
                    "Metric":     metric,
                    "Noise Type": noise_type_german.get(noise_type, noise_type),
                    "Noise Level": lvl,
                    "CV [%]":      round(cv,3)
                })
        pd.DataFrame(rows).to_csv(f"{output_dir}/cv_{sanitize_filename(metric)}.csv", index=False)
        print(f"CV-Daten gespeichert: {metric}")

cv_results = get_cv_results(base_dirs, target_metrics)

for metric in target_metrics:
    plot_cv_poisson_gaussian_save(cv_results, metric, output_dir="Variability")

plot_combined_cv_1d(cv_results, "gaussian", gaussian_noise_levels, target_metrics, output_dir="Variability")
plot_combined_cv_1d(cv_results, "salt_pepper", salt_pepper_noise_levels, target_metrics, output_dir="Variability")

save_cv_results_to_csv(cv_results, output_dir="Variability")
