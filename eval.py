import os
import yaml
import pandas as pd
import numpy as np
from ultralytics import YOLO
import signal
import sys
import shutil

moved_files = []

def signal_handler(sig, frame):
    for src, dst in moved_files:
        if os.path.exists(dst):
            try:
                os.rename(dst, src)
            except Exception as e:
                print(f"Failed to restore {dst}: {e}")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def delete_all_val_dirs(root_dir: str):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if "yolo12_val" in dirname.lower():
                try:
                    shutil.rmtree(full_path)
                except OSError as e:
                    print(f"Failed to delete {full_path}: {e}")


def get_leaf_subdirs(root_dir: str):
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:
            leaf_dirs.append(dirpath)
    return leaf_dirs

def move_images_to_dir(src_dir: str, dest_dir: str):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            os.rename(src_path, dest_path)
            moved_files.append((src_path, dest_path))

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
        verbose=True,
        device="cpu"
    )
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
    save_full_ap_matrix: bool = False,
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

    if save_full_ap_matrix:
        all_ap = np.array(box.all_ap)
        ap_matrix_df = pd.DataFrame(all_ap)
        ap_matrix_df.index.name = "Class Index"
        ap_matrix_df.columns = [f"IoU={iou:.2f}" for iou in np.linspace(0.5, 0.95, 10)]
        ap_matrix_path = os.path.join(os.path.dirname(f"{path}"), "yolo12_ap_matrix.csv")
        ap_matrix_df.to_csv(ap_matrix_path)

if __name__ == "__main__":

    delete_all_val_dirs(r"C:\Users\Admin\source\repos\Noise Simulator\output") # Path of the output directory of the Noise Simulator

    paths = get_leaf_subdirs(r"C:\Users\Admin\source\repos\Noise Simulator\output") # Path of the output directory of the Noise Simulator

    temp_eval_dir = r"C:\Users\Admin\source\repos\COCO YOLO Validation\temp_evaluation"
    new_yaml_path = r"C:\Users\Admin\source\repos\COCO YOLO Validation\patched_dataset.yml" 

    for i, path in enumerate(paths, 1):

        move_images_to_dir(
            src_dir=path,
            dest_dir= os.path.join(temp_eval_dir, "images")
        )

        metrics = run_validation(
            model_path="yolo12n.pt",
            data_path=new_yaml_path,
            output_dir=path
        )

        save_summary_metrics(
            metrics=metrics,
            path = os.path.join(path, "yolo12_val", "yolo12_validation_metrics"),
            format="json"
        )

        save_per_class_metrics(
            metrics=metrics,
            data_yaml=new_yaml_path,
            path=os.path.join(path, "yolo12_val", "yolo12_per_class_metrics"),
            format="json",
            save_full_ap_matrix=True,
            sort_by="AP@0.5:0.95"
        )

        move_images_to_dir(
            src_dir=os.path.join(temp_eval_dir, "images"),
            dest_dir=path
        )
