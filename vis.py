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


