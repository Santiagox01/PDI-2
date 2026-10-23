"""
Auditoria, inventario y caracterizacion cientifica del dataset de monedas de euro.
Genera dataset_manifest.csv, resumen estadistico y figuras analiticas.
"""

import os
import hashlib
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_md5(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset(config: dict) -> pd.DataFrame:
    raw_images_dir = config["paths"]["raw_images"]
    raw_labels_dir = config["paths"]["raw_labels"]
    manifest_path = config["paths"]["manifest"]
    classes_cfg = config["classes"]
    
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    image_files = sorted([f for f in os.listdir(raw_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[INFO] Auditando {len(image_files)} imagenes originales...")

    coin_records = []
    image_records = []

    for img_name in image_files:
        img_path = os.path.join(raw_images_dir, img_name)
        img_id = os.path.splitext(img_name)[0]
        lbl_name = f"{img_id}.txt"
        lbl_path = os.path.join(raw_labels_dir, lbl_name)
        
        file_size_bytes = os.path.getsize(img_path)
        img_md5 = calculate_md5(img_path)

        # Validacion de lectura
        img = cv2.imread(img_path)
        is_valid = img is not None
        if not is_valid:
            print(f"[WARN] Imagen corrupta o ilegible: {img_path}")
            h, w, c = 0, 0, 0
        else:
            h, w, c = img.shape

        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as lf:
                for line_idx, line in enumerate(lf):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    
                    xmin = max(0, int(round((xc - bw / 2.0) * w)))
                    ymin = max(0, int(round((yc - bh / 2.0) * h)))
                    xmax = min(w, int(round((xc + bw / 2.0) * w)))
                    ymax = min(h, int(round((yc + bh / 2.0) * h)))
                    
                    crop_w = xmax - xmin
                    crop_h = ymax - ymin
                    
                    # Hash del recorte para detectar duplicados
                    crop_md5 = ""
                    if is_valid and crop_w > 0 and crop_h > 0:
                        crop_bytes = img[ymin:ymax, xmin:xmax].tobytes()
                        crop_md5 = hashlib.md5(crop_bytes).hexdigest()

                    cls_info = classes_cfg.get(cls_id, {"name": f"Class {cls_id}", "value": 0.0, "color_group": "unknown"})
                    
                    coin_id = f"{img_id}_coin_{line_idx:02d}"
                    coin_record = {
                        "coin_id": coin_id,
                        "original_image_id": img_id,
                        "image_path": img_path,
                        "label_path": lbl_path,
                        "class_id": cls_id,
                        "class_name": cls_info["name"],
                        "face_value_eur": cls_info["value"],
                        "color_group": cls_info["color_group"],
                        "image_width": w,
                        "image_height": h,
                        "image_channels": c,
                        "image_file_size_bytes": file_size_bytes,
                        "image_md5": img_md5,
                        "bbox_xc": xc,
                        "bbox_yc": yc,
                        "bbox_bw": bw,
                        "bbox_bh": bh,
                        "bbox_xmin": xmin,
                        "bbox_ymin": ymin,
                        "bbox_xmax": xmax,
                        "bbox_ymax": ymax,
                        "crop_width_px": crop_w,
                        "crop_height_px": crop_h,
                        "crop_area_px": crop_w * crop_h,
                        "crop_aspect_ratio": (crop_w / crop_h) if crop_h > 0 else 0,
                        "crop_md5": crop_md5,
                        "is_valid": is_valid and crop_w > 10 and crop_h > 10
                    }
                    coin_records.append(coin_record)
                    boxes.append(coin_record)

        image_records.append({
            "original_image_id": img_id,
            "image_path": img_path,
            "width": w,
            "height": h,
            "channels": c,
            "file_size_bytes": file_size_bytes,
            "image_md5": img_md5,
            "num_coins": len(boxes),
            "is_valid": is_valid
        })

    df_coins = pd.DataFrame(coin_records)
    df_images = pd.DataFrame(image_records)

    df_coins.to_csv(manifest_path, index=False, encoding="utf-8")
    df_images.to_csv("data/interim/image_level_manifest.csv", index=False, encoding="utf-8")
    print(f"[INFO] Manifiesto de monedas guardado en: {manifest_path} ({len(df_coins)} registros)")
    print(f"[INFO] Manifiesto a nivel de imagen guardado en: data/interim/image_level_manifest.csv ({len(df_images)} imagenes)")

    # Analisis de duplicados
    dup_images = df_images[df_images.duplicated(subset=["image_md5"], keep=False)]
    dup_crops = df_coins[df_coins.duplicated(subset=["crop_md5"], keep=False)]
    print(f"[INFO] Imagenes completas duplicadas: {len(dup_images)}")
    print(f"[INFO] Recortes de monedas duplicados: {len(dup_crops)}")

    # Generacion de tablas estadisticas
    class_summary = df_coins.groupby(["class_id", "class_name", "face_value_eur", "color_group"]).agg(
        total_coins=("coin_id", "count"),
        mean_width_px=("crop_width_px", "mean"),
        std_width_px=("crop_width_px", "std"),
        mean_height_px=("crop_height_px", "mean"),
        std_height_px=("crop_height_px", "std"),
        mean_area_px=("crop_area_px", "mean"),
        mean_aspect_ratio=("crop_aspect_ratio", "mean")
    ).reset_index()
    
    class_summary["percentage"] = (class_summary["total_coins"] / len(df_coins)) * 100
    class_summary.to_csv("outputs/tables/dataset_summary.csv", index=False, encoding="utf-8")
    print("\n=== RESUMEN POR CLASE ===")
    print(class_summary[["class_id", "class_name", "total_coins", "percentage", "mean_width_px", "mean_height_px"]].to_string(index=False))

    # Visualizaciones academicas
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Distribucion de Clases
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("viridis", len(class_summary))
    bars = ax.bar(class_summary["class_name"], class_summary["total_coins"], color=palette, edgecolor="black", linewidth=1.2)
    ax.set_title("Distribución de Frecuencia por Denominación de Moneda", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Denominación de Euro", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cantidad de Monedas (Instancias)", fontsize=12, fontweight="bold")
    ax.axhline(60, color="crimson", linestyle="--", linewidth=1.5, label="Umbral sugerido en guía (60 muestras)")

    for bar, pct in zip(bars, class_summary["percentage"]):
        height = bar.get_height()
        ax.annotate(f"{int(height)}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig("outputs/figures/class_distribution.png", dpi=300)
    plt.close()

    # 2. Distribucion de Tamaños (Diametro en Pixeles) por Denominacion
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=df_coins, x="class_name", y="crop_width_px", ax=ax, palette="Set2")
    sns.stripplot(data=df_coins, x="class_name", y="crop_width_px", ax=ax, color="black", alpha=0.35, jitter=0.2, size=5)
    ax.set_title("Distribución del Diámetro Aparente (Ancho en Píxeles) por Denominación", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Denominación de Euro", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ancho del Bounding Box (píxeles)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/figures/coin_sizes_distribution.png", dpi=300)
    plt.close()

    # 3. Galeria de muestra de imagenes originales anotadas
    sample_images = ["001.jpg", "015.jpg", "042.jpg", "078.jpg", "105.jpg", "140.jpg"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, s_name in enumerate(sample_images):
        s_path = os.path.join(raw_images_dir, s_name)
        if not os.path.exists(s_path):
            continue
        vis_img = cv2.imread(s_path)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        s_id = os.path.splitext(s_name)[0]
        s_coins = df_coins[df_coins["original_image_id"] == s_id]

        for _, c_row in s_coins.iterrows():
            xmin, ymin, xmax, ymax = int(c_row["bbox_xmin"]), int(c_row["bbox_ymin"]), int(c_row["bbox_xmax"]), int(c_row["bbox_ymax"])
            cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            label_text = f"{c_row['class_name']}"
            cv2.putText(vis_img, label_text, (xmin, max(25, ymin - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        axes[idx].imshow(vis_img)
        axes[idx].set_title(f"Imagen: {s_name} ({len(s_coins)} monedas)", fontsize=11, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("Muestra de Fotografías de Escenas con Detecciones Ground Truth", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig("outputs/figures/dataset_sample_gallery.png", dpi=300)
    plt.close()
    print("[INFO] Figuras analiticas guardadas en outputs/figures/.")

    return df_coins


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    audit_dataset(cfg)
