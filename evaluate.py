import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import VisualBERTWithClassifier  # replace with your actual model class

def per_sample_class_probabilities(
    directory_path,
    visualbert_input,
    label_to_int,
    model_to_load,
    master_index,
    num_classes: int = 8
):
    """
    Computes per-sample class probabilities with predicted labels and per-class metrics.

    Args:
        directory_path: str, base directory
        visualbert_input: dict, output from prepare_visualbert_input
        label_to_int: dict, mapping label strings -> integers
        model_to_load: str, filename of saved model
        master_index: list of dicts with 'id' for each sample (in same order as visualbert_input)
        num_classes: int, number of classes

    Returns:
        tuple: (df_probs, df_metrics)
            df_probs: per-sample probabilities with predicted label
            df_metrics: per-class metrics
    """
    # -----------------------------
    # 1. Locate model
    # -----------------------------
    found_model_path = None
    for root, dirs, files in os.walk(directory_path):
        if model_to_load in files:
            found_model_path = os.path.join(root, model_to_load)
            break
    if found_model_path is None:
        raise FileNotFoundError(f"Model '{model_to_load}' not found in: {directory_path}")
    print(f"Found model at: {found_model_path}")

    # -----------------------------
    # 2. Load model
    # -----------------------------
    model = VisualBERTWithClassifier(num_classes)
    state_dict = torch.load(found_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------
    # 3. Prepare inputs
    # -----------------------------
    input_ids = visualbert_input["input_ids"]
    attention_mask = visualbert_input["attention_mask"]
    visual_embeds = visualbert_input["visual_embeds"]
    visual_attention_mask = visualbert_input["visual_attention_mask"]
    labels = visualbert_input["labels"]

    int_to_label = {v: k for k, v in label_to_int.items()}

    all_probs = []
    y_true = []
    y_pred = []

    # -----------------------------
    # 4. Forward pass per sample
    # -----------------------------
    for i, label in enumerate(labels):
        # Skip samples with missing ROIs
        if i >= len(visual_embeds):
            continue

        visual = visual_embeds[i].unsqueeze(0)
        text_ids = input_ids[i].unsqueeze(0)
        text_mask = attention_mask[i].unsqueeze(0)
        visual_mask = visual_attention_mask[i].unsqueeze(0)
        sample_id = master_index[i]["id"] if i < len(master_index) else f"sample_{i}"

        with torch.no_grad():
            logits = model(
                input_ids=text_ids,
                attention_mask=text_mask,
                visual_embeds=visual,
                visual_attention_mask=visual_mask
            )
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(probs))

        # -----------------------------
        # Per-sample probabilities with predicted label
        # -----------------------------
        sample_dict = {
            "id": sample_id,
            "true_label": int_to_label[label],
            "pred_label": int_to_label[pred_class]
        }
        for class_int, prob in enumerate(probs):
            sample_dict[int_to_label[class_int]] = prob
        all_probs.append(sample_dict)

        # -----------------------------
        # For per-class metrics
        # -----------------------------
        y_true.append(label)
        y_pred.append(pred_class)

    # -----------------------------
    # 5. Save per-sample probabilities
    # -----------------------------
    metrics_dir = os.path.dirname(found_model_path)
    df_probs = pd.DataFrame(all_probs)
    output_probs_path = os.path.join(metrics_dir, "test_per_sample_class_probabilities.csv")
    df_probs.to_csv(output_probs_path, index=False)
    print(f"Per-sample class probabilities with predicted labels saved to: {output_probs_path}")


import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def class_metrics(directory_path: str, model_name: str):
    """
    Computes per-class metrics AND overall model metrics by loading the
    per-sample class probabilities CSV for a given model.

    Saves two CSVs:
    - class_metrics.csv
    - model_metrics.csv
    """
    # -----------------------------
    # 1. Locate the model folder
    # -----------------------------
    found_model_path = None
    for root, dirs, files in os.walk(directory_path):
        if model_name in files:
            found_model_path = os.path.join(root, model_name)
            break

    if found_model_path is None:
        raise FileNotFoundError(f"Model '{model_name}' not found under: {directory_path}")

    metrics_dir = os.path.dirname(found_model_path)
    csv_path = os.path.join(metrics_dir, "test_per_sample_class_probabilities.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Per-sample class probabilities CSV not found at: {csv_path}")

    # -----------------------------
    # 2. Load per-sample CSV
    # -----------------------------
    df = pd.read_csv(csv_path)

    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise ValueError("CSV must contain 'true_label' and 'pred_label' columns.")

    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()

    # -----------------------------
    # 3. Identify all classes
    # -----------------------------
    classes = sorted(df["true_label"].unique())
    class_metrics = []

    # -----------------------------
    # 4. Compute per-class metrics
    # -----------------------------
    for class_name in classes:
        idx = (y_true == class_name)
        if idx.sum() == 0:
            class_metrics.append([class_name, np.nan, np.nan, np.nan, np.nan])
            continue

        acc = accuracy_score(y_true[idx], y_pred[idx])
        recall = recall_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        precision = precision_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        f1 = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)

        class_metrics.append([class_name, acc, recall, precision, f1])

    df_metrics = pd.DataFrame(
        class_metrics,
        columns=["class_name", "accuracy", "recall", "precision", "f1"]
    )

    # -----------------------------
    # 5. Save per-class metrics
    # -----------------------------
    class_metrics_path = os.path.join(metrics_dir, "class_metrics.csv")
    df_metrics.to_csv(class_metrics_path, index=False)
    print(f"Per-class metrics saved to: {class_metrics_path}")

    # -----------------------------
    # 6. Compute overall model metrics
    # -----------------------------
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    overall_macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    overall_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    df_model_metrics = pd.DataFrame(
        [[overall_accuracy, overall_macro_recall, overall_macro_precision, overall_macro_f1]],
        columns=["accuracy", "macro_recall", "macro_precision", "macro_f1"]
    )

    # -----------------------------
    # 7. Save overall model metrics
    # -----------------------------
    model_metrics_path = os.path.join(metrics_dir, "model_metrics.csv")
    df_model_metrics.to_csv(model_metrics_path, index=False)
    print(f"Model-level metrics saved to: {model_metrics_path}")
