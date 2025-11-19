import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def extract_rois_fasterrcnn(directory_path: str, dataset_split: str, samples: list, num_rois: int):
    """
    Extracts ROIs from images using Faster R-CNN (ResNet50-FPN) as an RPN.
    
    Inputs:
        directory_path: str, base folder
        dataset_split: str, e.g., 'train' or 'val'
        samples: list of dicts, each with keys 'id' (unique) and 'filepath'
        num_rois: int, number of ROIs to save per image
    
    Outputs:
        Saves a .pt file per sample in tensors_rois folder containing top num_rois proposals.
    """

    if len(samples) == 0:
        raise ValueError("Sample list is empty.")

    # Create output folder
    split_dir = os.path.join(directory_path, dataset_split, "tensors_rois")
    os.makedirs(split_dir, exist_ok=True)

    # Load pretrained Faster R-CNN (eval mode)
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define basic image transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for sample in tqdm(samples, desc=f"Extracting ROIs ({dataset_split})"):
        img_path = sample['filepath']
        sample_id = sample['id']  # use this for filename

        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).to(device) # type: ignore
        img_list = [img_tensor]  # Faster R-CNN expects a list of images

        # Get proposals
        with torch.no_grad():
            predictions = model(img_list)  # list of dicts per image
            boxes = predictions[0]['boxes']  # [num_boxes, 4]

        # Select top num_rois (by default first N)
        if boxes.shape[0] > num_rois:
            boxes = boxes[:num_rois]

        # Save tensor
        save_path = os.path.join(split_dir, f"{sample_id}.pt")
        torch.save(boxes.cpu(), save_path)

    print(f"\nSaved {len(samples)} ROI tensors to: {split_dir}")

import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def extract_rois_maskrcnn(directory_path: str, dataset_split: str, samples: list, num_rois: int):
    """
    Extracts ROIs from images using Mask R-CNN (ResNet50-FPN), saving both boxes and masks.

    Inputs:
        directory_path: str, base folder
        dataset_split: str, e.g., 'train' or 'val'
        samples: list of dicts, each with keys 'id' and 'filepath'
        num_rois: int, number of ROIs to save per image

    Outputs:
        Saves a .pt file per sample in tensors_rois folder containing top `num_rois` boxes and masks.
        Saved dictionary format: {'boxes': [num_rois,4], 'masks': [num_rois, H, W]}
    """

    if len(samples) == 0:
        raise ValueError("Sample list is empty.")

    # Output directory
    split_dir = os.path.join(directory_path, dataset_split, "tensors_rois")
    os.makedirs(split_dir, exist_ok=True)

    # Load pretrained Mask R-CNN (mask head enabled)
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for sample in tqdm(samples, desc=f"Extracting ROIs + Masks ({dataset_split})"):
        img_path = sample['filepath']
        sample_id = sample['id']

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).to(device) # type: ignore

        # Inference
        with torch.no_grad():
            output = model([img_tensor])[0]

        boxes = output["boxes"]       # [num_boxes, 4]
        masks = output["masks"]       # [num_boxes, 1, H, W]

        # Squeeze mask channel
        masks = masks.squeeze(1)      # [num_boxes, H, W]

        # If no detections, save empty tensors
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4))
            masks = torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]))

        # Keep top `num_rois` based on scores
        scores = output["scores"]
        if boxes.shape[0] > num_rois:
            topk_idx = torch.topk(scores, num_rois).indices
            boxes = boxes[topk_idx]
            masks = masks[topk_idx]

        # Save dictionary
        save_path = os.path.join(split_dir, f"{sample_id}.pt")
        torch.save({"boxes": boxes.cpu(), "masks": masks.cpu()}, save_path)

    print(f"\nSaved {len(samples)} ROI tensors (with masks) to: {split_dir}")



import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

def extract_feature_vectors(directory_path: str,
                            dataset_split: str,
                            pt_file_paths: list,
                            master_index: list):
    """
    Extracts feature vectors from ROI .pt files using Faster R-CNN backbone.
    Searches the master_index by the 'id' field instead of dict keys.

    Args:
        directory_path (str): Root dataset path.
        dataset_split (str): Dataset split folder (e.g., 'train').
        pt_file_paths (list): List of .pt file paths containing ROI boxes.
        master_index (list): List of dicts with 'id' and 'filepath' entries.
    """

    # -----------------------------
    # Output directory
    # -----------------------------
    save_dir = os.path.join(directory_path, dataset_split, "tensors_rois_features")
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Load Faster R-CNN backbone
    # -----------------------------
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    backbone = model.backbone
    backbone.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    resize_roi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    # -----------------------------
    # Process each .pt file
    # -----------------------------
    for pt_path in tqdm(pt_file_paths, desc=f"Extracting feature vectors ({dataset_split})"):

        # Get sample_id from filename
        sample_id = os.path.splitext(os.path.basename(pt_path))[0].strip()

        # Load ROI boxes
        try:
            boxes = torch.load(pt_path, weights_only=True)
        except Exception as e:
            tqdm.write(f"[ERROR] Could not load {pt_path}: {e}")
            continue

        if boxes is None or boxes.numel() == 0:
            tqdm.write(f"[WARNING] No ROIs for sample_id={sample_id}")
            continue

        # -----------------------------
        # Search master_index by 'id' column
        # -----------------------------
        entry = next((x for x in master_index if x['id'] == sample_id), None)
        if entry is None:
            tqdm.write(f"[ERROR] sample_id={sample_id} not found in master_index")
            continue

        img_path = entry['filepath']
        unique_id = entry['id']

        if not os.path.exists(img_path):
            tqdm.write(f"[ERROR] Image not found: {img_path}")
            continue

        # -----------------------------
        # Load image and crop ROIs
        # -----------------------------
        img = Image.open(img_path).convert("RGB")
        roi_crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            crop = img.crop((x1, y1, x2, y2))
            roi_crops.append(resize_roi(crop))

        if len(roi_crops) == 0:
            tqdm.write(f"[WARNING] No valid ROI crops for sample_id={sample_id}")
            continue

        roi_batch = torch.stack(roi_crops).to(device)

        # -----------------------------
        # Extract feature vectors
        # -----------------------------
        with torch.no_grad():
            feat_maps = backbone(roi_batch)

        feat = feat_maps["0"] if isinstance(feat_maps, dict) else feat_maps
        feat_vectors = feat.mean(dim=[2, 3]).cpu()  # Global average pool

        # -----------------------------
        # Save feature vectors
        # -----------------------------
        save_path = os.path.join(save_dir, f"{unique_id}.pt")
        torch.save(feat_vectors, save_path)


import os
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

def construct_visual_prototypes(directory_path: str,
                                feature_pt_paths: list,
                                master_index: list,
                                n_prototypes_per_class: int = 32,
                                save_filename: str = "visual_prototypes.pt"):
    """
    Constructs visual prototypes per class using MiniBatchKMeans with L2-normalization and saves as a PyTorch .pt file.

    Args:
        directory_path (str): Root dataset path.
        feature_pt_paths (list): List of .pt file paths (extracted ROI features).
        master_index (list): List of dicts containing at least 'id' and 'label'.
        n_prototypes_per_class (int): Number of prototypes per class.
        save_filename (str): Name of the output file (saved in directory_path).

    Output:
        Saves a dictionary of visual prototypes per class to a .pt file.
        Format: {class_label: torch.Tensor of shape [n_prototypes, feature_dim]}
    """

    # -----------------------------
    # Map sample_id → label
    # -----------------------------
    id_to_label = {entry['id']: entry['label'] for entry in master_index}

    # -----------------------------
    # Collect samples per class
    # -----------------------------
    class_to_features = {}  # class_label -> list of feature vectors

    for pt_path in tqdm(feature_pt_paths, desc=f"Assigning features to classes"):
        sample_id = os.path.splitext(os.path.basename(pt_path))[0].strip()

        if sample_id not in id_to_label:
            tqdm.write(f"[WARNING] sample_id={sample_id} not found in master_index, skipping")
            continue

        label = id_to_label[sample_id]

        # Load feature vectors
        feat = torch.load(pt_path, weights_only=True)  # shape: [num_rois, feature_dim] or [feature_dim]
        if feat.ndim == 2:
            feat_mean = feat.mean(dim=0)  # average ROIs → tensor
        elif feat.ndim == 1:
            feat_mean = feat
        else:
            tqdm.write(f"[WARNING] Unexpected feature shape for {sample_id}, skipping")
            continue

        # L2-normalize the feature vector
        feat_mean = torch.tensor(normalize(feat_mean.unsqueeze(0).numpy(), axis=1)[0], dtype=feat_mean.dtype)

        if label not in class_to_features:
            class_to_features[label] = []
        class_to_features[label].append(feat_mean)

    # -----------------------------
    # Perform MiniBatchKMeans per class
    # -----------------------------
    visual_prototypes = {}

    for label, features_list in class_to_features.items():
        features_array = torch.stack(features_list)  # [num_samples, feature_dim]

        if features_array.shape[0] < n_prototypes_per_class:
            # if fewer samples than prototypes, just use all samples
            visual_prototypes[label] = features_array
            tqdm.write(f"[INFO] Fewer samples than prototypes for class {label}, using all {features_array.shape[0]}")
            continue

        # MiniBatchKMeans expects numpy array
        features_array_np = features_array.numpy()
        kmeans = MiniBatchKMeans(n_clusters=n_prototypes_per_class, batch_size=32, random_state=42)
        kmeans.fit(features_array_np)
        visual_prototypes[label] = torch.tensor(kmeans.cluster_centers_, dtype=features_array.dtype)

    # -----------------------------
    # Save visual prototypes
    # -----------------------------
    save_path = os.path.join(directory_path, save_filename)
    torch.save(visual_prototypes, save_path)

    print(f"Saved visual prototypes to: {save_path}")
