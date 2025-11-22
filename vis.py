import os
import torch
from PIL import Image, ImageDraw, ImageFont

def plot_rois_on_image(directory_path: str, master_index: list, unique_index: int):
    """
    Instead of plotting bounding boxes on the original image,
    this function crops the regions inside each ROI bounding box
    and arranges them side-by-side on a white canvas with numbers.

    Output:
        A .jpeg image saved in directory_path/vis/rois/<id>.jpeg
        containing all ROI crops side-by-side.
    """

    # -----------------------------
    # Find sample entry
    # -----------------------------
    sample = next((item for item in master_index if item['id'] == unique_index), None)
    if sample is None:
        raise ValueError(f"ID {unique_index} not found in master_index")

    image_path = sample['filepath']
    split = sample['split']

    # -----------------------------
    # Locate .pt ROI file
    # -----------------------------
    roi_path = os.path.join(directory_path, split, "tensors_rois", f"{unique_index}.pt")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    boxes = torch.load(roi_path)
    if boxes.numel() == 0:
        print(f"Warning: No ROIs found for sample {unique_index}")

    # -----------------------------
    # Prepare save directory
    # -----------------------------
    vis_dir = os.path.join(directory_path, "vis", "rois")
    os.makedirs(vis_dir, exist_ok=True)

    # -----------------------------
    # Load image
    # -----------------------------
    img = Image.open(image_path).convert("RGB")

    # Crop all ROIs
    crops = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        crop = img.crop((x1, y1, x2, y2))
        crops.append((i + 1, crop))  # (number, image)

    if len(crops) == 0:
        print("No ROIs to visualize.")
        return

    # -----------------------------
    # Create white canvas side-by-side
    # -----------------------------
    # Resize all crops to same height for cleaner layout
    target_height = 200
    resized = []

    for num, crop in crops:
        w, h = crop.size
        scale = target_height / h
        new_w = int(w * scale)
        crop_resized = crop.resize((new_w, target_height))
        resized.append((num, crop_resized))

    # Compute total canvas size
    total_width = sum(c.size[0] for _, c in resized) + (10 * (len(resized) - 1))
    canvas_height = target_height + 40  # extra space for numbers

    canvas = Image.new("RGB", (total_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Try default font
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # Paste crops + add numbering
    x_offset = 0
    for num, crop in resized:
        canvas.paste(crop, (x_offset, 0))
        draw.text((x_offset + 5, target_height + 5), f"{num}", fill="black", font=font)
        x_offset += crop.size[0] + 10

    # -----------------------------
    # Save output
    # -----------------------------
    save_path = os.path.join(vis_dir, f"{unique_index}.jpeg")
    canvas.save(save_path, format="JPEG")


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_per_sample_class_probabilities(directory_path: str, model_file_name: str):
    """
    Recursively scans subfolders to locate model_file_name (e.g., 'demo_5rois.pt'), 
    then creates per-sample bar plots for train, val, and test probability CSVs.
    First bar is the true label probability, followed by the top 5 probabilities.
    """

    # Recursively search for the model file
    model_folder = None
    for root, dirs, files in os.walk(directory_path):
        if model_file_name in files:
            model_folder = root
            break

    if model_folder is None:
        raise FileNotFoundError(f"Model file '{model_file_name}' not found under '{directory_path}'")

    print(f"Found model file in: {model_folder}")

    # Create output folders
    out_folders = {
        "train": os.path.join(os.getcwd(), "vis_train_per_sample_top_5_probabilities"),
        "val": os.path.join(os.getcwd(), "vis_val_per_sample_top_5_probabilities"),
        "test": os.path.join(os.getcwd(), "vis_test_per_sample_top_5_probabilities")
    }
    for folder in out_folders.values():
        os.makedirs(folder, exist_ok=True)

    # File mapping
    files = {
        "train": "train_per_sample_class_probabilities.csv",
        "val": "val_per_sample_class_probabilities.csv",
        "test": "test_per_sample_class_probabilities.csv"
    }

    for split, file_name in files.items():
        file_path = os.path.join(model_folder, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in '{model_folder}', skipping {split}.")
            continue

        df = pd.read_csv(file_path)
        last_epoch = df["epoch"].max()
        df = df[df["epoch"] == last_epoch]

        for idx, row in df.iterrows():
            true_label = row["true_label"]
            probs = row[[c for c in df.columns if "class_" in c]].values.astype(float)
            pred_label = row["pred_label"]

            # Top 5 excluding true label
            probs_excl_true = np.delete(probs, true_label)
            top5_indices = probs_excl_true.argsort()[-5:][::-1]
            top5_probs = probs_excl_true[top5_indices]

            true_prob = probs[true_label]
            bar_values = np.concatenate([[true_prob], top5_probs])

            labels = ["true_label"] + [f"top{i+1}" for i in range(5)]
            title = row["sample_index"] if split != "test" else row.get("id", idx)

            plt.figure(figsize=(6, 4))
            plt.bar(labels, bar_values, color=["green"] + ["blue"]*5)
            plt.ylim(0, 1)
            plt.title(f"Sample {title} | True Label: {true_label}, Pred: {pred_label}")
            plt.ylabel("Probability")
            plt.tight_layout()

            save_path = os.path.join(out_folders[split], f"{title}.png")
            plt.savefig(save_path)
            plt.close()
