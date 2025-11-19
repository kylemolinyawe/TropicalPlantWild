import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_master_index(img_directory_path: str, exp_directory_path):
    """
    Generates master_index.csv inside the given directory.

    Scans directory_path/images and all subfolders.
    For each image found:
        - assigns a unique ID
        - records the filepath
        - extracts the label (the name of the folder containing the image)
    Performs a 70/20/10 stratified random split.
    Creates a CSV containing: id, filepath, label, split
    """

    images_root = os.path.join(img_directory_path)

    if not os.path.exists(images_root):
        raise FileNotFoundError(f"'images' not found inside {img_directory_path}")

    entries = []
    uid = 0

    # Walk recursively through images
    for root, _, files in os.walk(images_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                full_path = os.path.join(root, file)

                # Label is folder name where the image is located
                label = os.path.basename(os.path.dirname(full_path))

                entries.append((uid, full_path, label))
                uid += 1

    # Convert to DataFrame for easier splitting
    df = pd.DataFrame(entries, columns=["id", "filepath", "label"])

    # ----------------------------
    # Stratified 70/20/10 Split
    # ----------------------------
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,  # remaining 30% -> val + test
        random_state=42,
        stratify=df["label"]
    )

    # Split temp into 20% val and 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1/3),  # test = 1/3 of 30% = 10% total
        random_state=42,
        stratify=temp_df["label"]
    )

    # Add split labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Combine and sort by ID
    final_df = pd.concat([train_df, val_df, test_df]).sort_values("id")

    # Save to CSV
    output_csv = os.path.join(exp_directory_path, "master_index.csv")
    final_df.to_csv(output_csv, index=False)

    print(f"master_index.csv created at {output_csv}")
    print(f"Total samples: {len(final_df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")



import os
import csv

def load_master_index(directory_path: str):
    """
    Loads master_index.csv from the given directory.
    
    Returns:
        train_entries: list of dicts for training samples
        val_entries:   list of dicts for validation samples
        test_entries:  list of dicts for testing samples
    
    Also prints the number of samples per split.
    """

    csv_path = os.path.join(directory_path, "master_index.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"master_index.csv not found in {directory_path}")

    train_entries = []
    val_entries = []
    test_entries = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:

            # Sort into split categories
            split = row["split"].lower()

            if split == "train":
                train_entries.append(row)
            elif split == "val" or split == "validation":
                val_entries.append(row)
            elif split == "test":
                test_entries.append(row)
            else:
                raise ValueError(f"Unknown split value '{row['split']}' for id={row['id']}")

    # Print shapes
    print(f"Loaded master_index.csv from: {csv_path}")
    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")

    return train_entries, val_entries, test_entries

import os

def get_tensor_rois_file_paths(directory_path: str, dataset_split: str) -> list:
    """
    Scans the dataset split folder for 'tensors_rois' and returns all .pt file paths.

    Args:
        directory_path (str): Root path of the dataset.
        dataset_split (str): Subfolder of the dataset (e.g., 'train', 'test').

    Returns:
        list of str: Full paths to all .pt files inside tensors_rois folder.
    """

    rois_dir = os.path.join(directory_path, dataset_split, "tensors_rois")

    if not os.path.exists(rois_dir):
        print(f"[WARNING] Directory not found: {rois_dir}")
        return []

    pt_file_paths = [
        os.path.join(rois_dir, f)
        for f in os.listdir(rois_dir)
        if f.endswith(".pt")
    ]

    if not pt_file_paths:
        print(f"[WARNING] No .pt files found in: {rois_dir}")

    # Optional: sort for consistent ordering
    pt_file_paths.sort()

    return pt_file_paths

import os

def get_tensor_rois_features_file_paths(directory_path: str, dataset_split: str):
    """
    Scans the dataset split folder for 'tensors_rois_features' and returns
    a list of all .pt file paths inside it.

    Args:
        directory_path (str): Root dataset path
        dataset_split (str): Dataset split folder (e.g., 'train', 'val', 'test')

    Returns:
        List[str]: List of .pt file paths
    """

    features_dir = os.path.join(directory_path, dataset_split, "tensors_rois_features")
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"'tensors_rois_features' folder not found: {features_dir}")

    pt_file_paths = [
        os.path.join(features_dir, f)
        for f in os.listdir(features_dir)
        if f.endswith(".pt")
    ]

    return pt_file_paths


import torch

def count_tensors_rois_shape(pt_file_paths: list):
    """
    Loads each .pt file in the list and prints the shape of the tensor.
    At the end, prints a summary tally.

    Args:
        pt_file_paths (list): List of paths to .pt files.

    Returns:
        None
    """

    tally = {}
    total_files = len(pt_file_paths)

    for i, path in enumerate(pt_file_paths, 1):
        try:
            tensor = torch.load(path, weights_only=True)
            shape = tuple(tensor.shape)

            if shape in tally:
                tally[shape] += 1
            else:
                tally[shape] = 1

        except Exception as e:
            print(f"[ERROR] Could not load {path}: {e}")

    print("\n--- Final Tally of Tensor Shapes ---")
    for shape, count in tally.items():
        print(f"{shape}: {count} file(s)")


import torch

def inspect_tensor(tensor: torch.Tensor, full_print=False, max_rows=10, max_cols=10):
    """
    Prints a tensor safely with optional full print or summary stats.

    Inputs:
        tensor: torch.Tensor to inspect
        full_print: bool, if True prints the entire tensor (can be very large)
        max_rows: int, max number of rows to print when full_print=False
        max_cols: int, max number of columns to print when full_print=False

    Output:
        None (prints to console)
    """
    print(f"Tensor shape: {tensor.shape}")
    print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}")
    print(f"Mean: {tensor.mean().item():.4f}, Zeros: {(tensor == 0).sum().item()}")

    if full_print:
        # Temporarily disable truncation
        torch.set_printoptions(profile="full")
        print(tensor)
        torch.set_printoptions(profile="default")
    else:
        # Print a subset
        rows = min(max_rows, tensor.shape[0])
        cols = min(max_cols, tensor.shape[1]) if tensor.ndim > 1 else 1
        print("Tensor preview:")
        if tensor.ndim == 1:
            print(tensor[:rows])
        elif tensor.ndim == 2:
            print(tensor[:rows, :cols])
        else:
            print(tensor[:rows, :cols, ...])

import os
import torch

def inspect_visual_prototypes(directory_path: str):
    """
    Inspect the contents of visual_prototypes.pt inside the given directory.

    Args:
        directory_path (str): Path to the directory containing visual_prototypes.pt

    Returns:
        None. Prints out information about the file.
    """
    prototype_file = os.path.join(directory_path, 'visual_prototypes.pt')
    
    if not os.path.exists(prototype_file):
        print(f"No file named 'visual_prototypes.pt' found in {directory_path}")
        return
    
    # Load the file
    data = torch.load(prototype_file)
    
    # Inspect type
    print(f"Type of loaded data: {type(data)}")
    
    # If tensor or numpy array, print shape
    if isinstance(data, torch.Tensor):
        print(f"Tensor shape: {data.shape}")
        print(f"Tensor sample data (first 5 rows):\n{data[:5]}")
    elif isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        # Inspect each key if numeric
        for key, value in data.items():
            print(f"\nKey: '{key}', Type: {type(value)}")
            if isinstance(value, torch.Tensor):
                print(f"Shape: {value.shape}, Sample:\n{value[:5]}")
            elif isinstance(value, list):
                print(f"Length: {len(value)}, Sample: {value[:5]}")
            else:
                print(f"Value (sample): {value}")
    elif isinstance(data, list):
        print(f"List length: {len(data)}, Sample: {data[:5]}")
    else:
        print(f"Data sample: {data}")

import json
import os

def load_text_prompts():
    """
    Loads class-specific text prompts from a JSON file named 'prompts_demo.json'
    located in the same directory as this script.

    Returns:
        dict: {class_name: [list of 50 prompts]}
    """

    # Get directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Expected JSON path
    json_path = os.path.join(current_dir, "prompts_demo.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"'prompts_demo.json' not found at: {json_path}")

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if not isinstance(data, dict):
        raise ValueError("JSON must contain a dictionary of {class_name: prompts_list}")

    for cls, prompts in data.items():
        if not isinstance(prompts, list):
            raise ValueError(f"Prompts for class '{cls}' must be a list.")
        if len(prompts) != 50:
            raise ValueError(
                f"Class '{cls}' must have exactly 50 prompts, found {len(prompts)}."
            )
        if not all(isinstance(p, str) for p in prompts):
            raise ValueError(f"All prompts for class '{cls}' must be strings.")

    return data

import pandas as pd
import os

def get_labels(directory_path: str):
    """
    Scans for a master_index.csv in the specified directory,
    extracts all unique labels, and prints them.

    Args:
        directory_path (str): Path to the directory containing master_index.csv

    Returns:
        List[str]: List of unique labels
    """
    file_path = os.path.join(directory_path, "master_index.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    # Load CSV
    df = pd.read_csv(file_path)
    
    if 'label' not in df.columns:
        print("Column 'label' not found in master_index.csv")
        return []

    # Get unique labels
    unique_labels = df['label'].unique().tolist()

    # Print the final list
    print("Unique labels:", unique_labels)

    return unique_labels

def inspect_tokenized_prompts(directory_path: str):
    """
    Inspects the 'tokenized_text_prompts.pt' file that stores:
        {
            class_name: {
                "input_ids": tensor([seq_len]),
                "attention_mask": tensor([seq_len])
            }
        }

    Prints the number of classes and shape/dtype information for the first 3 classes.
    """
    import os
    import torch

    file_path = os.path.join(directory_path, "tokenized_text_prompts.pt")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    tokenized = torch.load(file_path, weights_only=True)

    if not isinstance(tokenized, dict):
        print("Unexpected format: Expected dict[class_name -> {input_ids, attention_mask}]")
        return

    print(f"\nLoaded 'tokenized_text_prompts.pt' from {directory_path}")
    print(f"Number of classes: {len(tokenized)}")
    print("\nInspecting first 3 classes:\n")

    for idx, (cls_name, entry) in enumerate(tokenized.items()):
        if idx >= 3:
            break

        print(f"[{idx}] Class: {cls_name}")

        if not isinstance(entry, dict):
            print("  ERROR: Entry is not a dict containing 'input_ids' and 'attention_mask'.")
            continue

        if "input_ids" not in entry or "attention_mask" not in entry:
            print("  ERROR: Missing keys in entry.")
            continue

        ids = entry["input_ids"]
        mask = entry["attention_mask"]

        print(f"  input_ids shape: {ids.shape}, dtype={ids.dtype}")
        print(f"  attention_mask shape: {mask.shape}, dtype={mask.dtype}")
        print()



