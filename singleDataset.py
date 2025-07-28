import os
import pickle
import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO

def run_validation(
    model_path: str = "yolo12n.pt",
    data_path: str = "dataset.yml",
    output_dir: str = "val_results"
):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_path,
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.5,
        save_json=True,
        save_txt=True,
        save_conf=True,
        plots=True,
        project=output_dir,
        name="yolo12_val",
        verbose=True
    )
    with open("yolo12_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    return metrics

def load_metrics(filepath: str = "yolo12_metrics.pkl"):
    with open(filepath, "rb") as f:
        metrics = pickle.load(f)
    return metrics

def save_summary_metrics(metrics, path: str = "yolo12_validation_metrics", format: str = "csv"):
    df = pd.DataFrame({
        "Metric": ["Mean Precision", "Mean Recall", "mAP@0.50", "mAP@0.75", "mAP@0.5:0.95"],
        "Value": [
            metrics.box.mp,
            metrics.box.mr,
            metrics.box.map50,
            metrics.box.map75,
            metrics.box.map
        ]
    })

    if format == "json":
        df.to_json(f"{path}.json", orient="records", indent=2)
    else:
        df.to_csv(f"{path}.csv", index=False)


def save_per_class_metrics(
    metrics,
    data_yaml: str = "dataset.yml",
    path: str = "yolo12_per_class_metrics",
    format: str = "csv",
    sort_by: str = None
):
    box = metrics.box

    p = np.array(box.p).flatten()
    r = np.array(box.r).flatten()
    f1 = np.array(box.f1).flatten()
    ap50 = np.array(box.ap50).flatten()
    ap = np.array(box.ap).flatten()
    class_indices = np.array(box.ap_class_index).flatten()

    df = pd.DataFrame({
        "Class Index": class_indices,
        "Precision": p,
        "Recall": r,
        "F1 Score": f1,
        "AP@0.50": ap50,
        "AP@0.5:0.95": ap
    })

    if os.path.exists(data_yaml):
        with open(data_yaml, "r") as f:
            data = yaml.safe_load(f)
        names_yaml = data.get("names", [])

        if isinstance(names_yaml, dict):
            names = [names_yaml.get(i, f"class_{i}") for i in class_indices]
        elif isinstance(names_yaml, list):
            names = [names_yaml[i] if i < len(names_yaml) else f"class_{i}" for i in class_indices]
        else:
            names = [f"class_{i}" for i in class_indices]

        df["Class Name"] = names

    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    else:
        df = df.sort_values(by="Class Index")

    if format == "json":
        df.to_json(f"{path}.json", orient="records", indent=2)
    else:
        df.to_csv(f"{path}.csv", index=False)


if __name__ == "__main__":
    metrics = run_validation()
    metrics = load_metrics()
    save_summary_metrics(metrics, path="summary_metrics", format="json")
    save_per_class_metrics(metrics, format="json")
    save_per_class_metrics(metrics, format="json", sort_by="F1 Score")
