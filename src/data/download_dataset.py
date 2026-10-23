"""
Modulo de descarga y preparacion del dataset janstaffa/euro-coins-dataset desde Kaggle.
"""

import os
import shutil
import argparse
import kagglehub
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_and_sync_dataset(config: dict, force: bool = False) -> str:
    raw_images_dir = config["paths"]["raw_images"]
    raw_labels_dir = config["paths"]["raw_labels"]

    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(raw_labels_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(raw_images_dir) if f.lower().endswith((".jpg", ".png"))]
    if existing_images and not force:
        print(f"[INFO] Dataset already populated in {raw_images_dir} ({len(existing_images)} images). Skipping re-download.")
        return raw_images_dir

    print("[INFO] Downloading janstaffa/euro-coins-dataset via kagglehub...")
    kaggle_path = kagglehub.dataset_download("janstaffa/euro-coins-dataset")
    print(f"[INFO] Downloaded to cache: {kaggle_path}")

    src_images = os.path.join(kaggle_path, "images")
    src_labels = os.path.join(kaggle_path, "labels")

    if not os.path.exists(src_images) or not os.path.exists(src_labels):
        raise FileNotFoundError(f"Expected 'images' and 'labels' directories in {kaggle_path}")

    copied_img = 0
    for f in os.listdir(src_images):
        shutil.copy2(os.path.join(src_images, f), os.path.join(raw_images_dir, f))
        copied_img += 1

    copied_lbl = 0
    for f in os.listdir(src_labels):
        shutil.copy2(os.path.join(src_labels, f), os.path.join(raw_labels_dir, f))
        copied_lbl += 1

    print(f"[INFO] Successfully copied {copied_img} images and {copied_lbl} labels into {raw_images_dir} and {raw_labels_dir}.")
    return raw_images_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and copy Euro coins dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    download_and_sync_dataset(cfg, force=args.force)
