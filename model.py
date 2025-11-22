def map_labels_to_int(unique_labels: list):
    """
    Creates a dictionary mapping each label to an integer (label -> int),
    prints it, and returns it.

    Args:
        unique_labels (list): List of unique labels

    Returns:
        dict: Dictionary mapping labels to integers
    """
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    print("Label to integer mapping:", label_to_int)

    return label_to_int


import os
import torch
from transformers import BertTokenizer

def tokenize_text_prompts(directory_path: str, class_prompts: dict):
    """
    Tokenizes prompts per class using the BERT tokenizer and stores 
    averaged token IDs + attention masks per class.

    Output:
        {
            class_name: {
                "input_ids": tensor([512]),
                "attention_mask": tensor([512])
            }
        }
    """

    # Ensure directory
    os.makedirs(directory_path, exist_ok=True)
    save_path = os.path.join(directory_path, "tokenized_text_prompts.pt")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    per_class = {}

    for cls, prompts in class_prompts.items():
        ids_list = []
        mask_list = []

        for prompt in prompts:
            enc = tokenizer(prompt, max_length=512, padding="max_length",
                            truncation=True, return_tensors="pt")
            ids_list.append(enc["input_ids"][0])
            mask_list.append(enc["attention_mask"][0])

        ids_avg = torch.stack(ids_list).mode(dim=0).values  # pick most common token
        mask_avg = torch.stack(mask_list).float().mean(dim=0).round().long()

        per_class[cls] = {
            "input_ids": ids_avg,
            "attention_mask": mask_avg
        }

    torch.save(per_class, save_path)
    print("Saved:", save_path)
    return save_path



def prepare_visualbert_input(
    directory_path: str,
    master_index,
    roi_paths: list,
    label_to_int: dict,
    num_rois: int,
    visualbert_hidden_dim: int = 2048
):
    """
    Prepares VisualBERT inputs using tokenized_text_prompts.pt and ROI features.

    Inputs:
        directory_path: path containing tokenized_text_prompts.pt
        master_index: list/dict containing keys ["id", "label"]
        roi_paths: list of .pt ROI feature paths
        label_to_int: mapping of label names to integers
        num_rois: number of regions of interest to pad/truncate
        visualbert_hidden_dim: VisualBERT expected feature size (default: 768)

    Returns:
        dict containing:
            input_ids
            attention_mask
            visual_embeds
            visual_attention_mask
            labels
    """
    import os
    import torch
    import torch.nn as nn

    # -------------------------------------------------
    # Load tokenized prompts
    # -------------------------------------------------
    prompt_path = os.path.join(directory_path, "tokenized_text_prompts.pt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"{prompt_path} not found.")

    tokenized = torch.load(prompt_path, weights_only=True)

    # -------------------------------------------------
    # Build mapping: id → label from master_index
    # -------------------------------------------------
    id_to_label = {row["id"]: row["label"] for row in master_index}

    # -------------------------------------------------
    # Prepare output lists
    # -------------------------------------------------
    input_ids_list = []
    attention_mask_list = []
    visual_embeds_list = []
    visual_attention_mask_list = []
    labels_list = []

    projector = None  # Only create when needed

    # -------------------------------------------------
    # Iterate through each ROI feature file
    # -------------------------------------------------
    for roi_path in roi_paths:

        # Extract filename without extension → matches master_index['id']
        file_id = os.path.splitext(os.path.basename(roi_path))[0]

        if file_id not in id_to_label:
            raise ValueError(f"ID {file_id} not found in master_index['id'].")

        label_name = id_to_label[file_id]
        if label_name not in label_to_int:
            raise ValueError(f"Label {label_name} missing in label_to_int mapping.")

        # Load the ROI tensor
        roi_tensor = torch.load(roi_path, weights_only=True)    # [num_actual_rois, feat_dim]

        actual_rois = roi_tensor.shape[0]  # number of real ROIs before padding

        # Project to 768 if needed
        if roi_tensor.shape[-1] != visualbert_hidden_dim:
            if projector is None:
                projector = nn.Linear(roi_tensor.shape[-1], visualbert_hidden_dim)
            roi_tensor = projector(roi_tensor)

        # Pad or truncate ROIs
        if actual_rois < num_rois:
            padding = torch.zeros((num_rois - actual_rois, visualbert_hidden_dim))
            roi_tensor = torch.cat([roi_tensor, padding], dim=0)
        elif actual_rois > num_rois:
            roi_tensor = roi_tensor[:num_rois, :]
            actual_rois = num_rois  # adjust actual_rois after truncation

        visual_embeds_list.append(roi_tensor)

        # -----------------------------
        # Correct visual_attention_mask
        # -----------------------------
        mask_real = torch.ones(actual_rois, dtype=torch.long)
        mask_pad = torch.zeros(num_rois - actual_rois, dtype=torch.long)
        mask = torch.cat([mask_real, mask_pad], dim=0)
        visual_attention_mask_list.append(mask)

        # -----------------------------
        # Append text embedding + mask
        # -----------------------------
        token_entry = tokenized[label_name]
        input_ids_list.append(token_entry["input_ids"])
        attention_mask_list.append(token_entry["attention_mask"])

        # -----------------------------
        # Label integer
        # -----------------------------
        labels_list.append(label_to_int[label_name])

    # -------------------------------------------------
    # Build final dictionary
    # -------------------------------------------------
    output = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "visual_embeds": visual_embeds_list,
        "visual_attention_mask": visual_attention_mask_list,
        "labels": labels_list
    }

    # -------------------------------------------------
    # Print shapes
    # -------------------------------------------------
    print("\nPrepared VisualBERT Input:")
    print("input_ids:", [x.shape for x in input_ids_list[:3]])
    print("attention_mask:", [x.shape for x in attention_mask_list[:3]])
    print("visual_embeds:", [x.shape for x in visual_embeds_list[:3]])
    print("visual_attention_mask:", [x.shape for x in visual_attention_mask_list[:3]])
    print("labels:", labels_list[:3])
    print(f"Total samples: {len(labels_list)}")

    return output


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import VisualBertModel

class VisualBERTDataset(Dataset):
    """Dataset for VisualBERT inputs from prepare_visualbert_input"""
    def __init__(self, visualbert_input):
        self.input_ids = visualbert_input["input_ids"]
        self.visual_embeds = visualbert_input["visual_embeds"]
        self.attention_mask = visualbert_input["attention_mask"]
        self.visual_attention_mask = visualbert_input["visual_attention_mask"]
        self.labels = visualbert_input["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "visual_embeds": self.visual_embeds[idx],
            "attention_mask": self.attention_mask[idx],
            "visual_attention_mask": self.visual_attention_mask[idx],
            "label": self.labels[idx]
        }

class VisualBERTWithClassifier(nn.Module):
    """VisualBERT backbone + classifier head"""
    def __init__(self, num_classes, pretrained_name="uclanlp/visualbert-vqa-coco-pre", hidden_dim=768):
        super().__init__()
        self.visualbert = VisualBertModel.from_pretrained(pretrained_name, use_safetensors=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, visual_embeds, attention_mask=None, visual_attention_mask=None):
        outputs = self.visualbert(
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask
        )
        pooled_output = outputs.pooler_output  # [batch, hidden_dim]
        logits = self.classifier(pooled_output)  # [batch, num_classes]
        return logits

import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import random

def train_visualbert(
    directory_path: str,
    visualbert_input,
    val_input=None,
    filename: str = "visualbert_model.pt",
    num_epochs=10,
    batch_size=4,
    lr=1e-5,
    log_probabilities: bool = False,
    seed: int = 42
):
    """
    Train VisualBERT + classifier with logging capability using visual_attention_mask,
    and log per-sample probabilities including true labels and predicted labels.
    Deterministic behavior is enforced using the provided seed.
    """

    # -------------------------------
    # Seed everything for determinism
    # -------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    num_classes = max(visualbert_input["labels"]) + 1
    print(f"Inferred number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisualBERTWithClassifier(num_classes)
    model.to(device)

    train_dataset = VisualBERTDataset(visualbert_input)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_input:
        val_dataset = VisualBERTDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def tensorize(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.stack(x).to(device)

    # ----------------------------------------------------
    # Setup logging files
    # ----------------------------------------------------
    model_name = os.path.splitext(filename)[0]
    model_folder = os.path.join(directory_path, model_name)
    os.makedirs(model_folder, exist_ok=True)

    metrics_path = os.path.join(model_folder, "training_metrics.csv")
    train_prob_path = os.path.join(model_folder, "train_per_sample_class_probabilities.csv")
    val_prob_path = os.path.join(model_folder, "val_per_sample_class_probabilities.csv")

    # Write CSV header (metrics)
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    # Write CSV header (probabilities) if enabled
    if log_probabilities:
        header = ["epoch", "sample_index", "true_label", "pred_label"] + [f"class_{i}_prob" for i in range(num_classes)]
        with open(train_prob_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        if val_input:
            with open(val_prob_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    # ----------------------------------------------------
    # Training Loop
    # ----------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        # -------- Train Phase --------
        for batch in loop:
            optimizer.zero_grad()

            input_ids = tensorize(batch["input_ids"]).long()
            attention_mask = tensorize(batch["attention_mask"]).long()
            visual_embeds = tensorize(batch["visual_embeds"]).float().detach()
            visual_attention_mask = tensorize(batch["visual_attention_mask"]).long()
            labels = tensorize(batch["label"]).long()

            logits = model(
                input_ids=input_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask
            )
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # -------- Validation Phase --------
        val_loss = 0
        val_acc = 0

        if val_input:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = tensorize(batch["input_ids"]).long()
                    attention_mask = tensorize(batch["attention_mask"]).long()
                    visual_embeds = tensorize(batch["visual_embeds"]).float().detach()
                    visual_attention_mask = tensorize(batch["visual_attention_mask"]).long()
                    labels = tensorize(batch["label"]).long()

                    logits = model(
                        input_ids=input_ids,
                        visual_embeds=visual_embeds,
                        visual_attention_mask=visual_attention_mask
                    )
                    loss = criterion(logits, labels)

                    val_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # -------- Write Metrics --------
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_loss, val_acc])

        # -------- Write per-sample probabilities --------
        if log_probabilities:
            model.eval()
            sample_index = 0

            # --- Train probabilities ---
            with torch.no_grad(), open(train_prob_path, "a", newline="") as f:
                writer = csv.writer(f)
                for batch in train_loader:
                    input_ids = tensorize(batch["input_ids"]).long()
                    attention_mask = tensorize(batch["attention_mask"]).long()
                    visual_embeds = tensorize(batch["visual_embeds"]).float().detach()
                    visual_attention_mask = tensorize(batch["visual_attention_mask"]).long()
                    labels = batch["label"]

                    logits = model(
                        input_ids=input_ids,
                        visual_embeds=visual_embeds,
                        visual_attention_mask=visual_attention_mask
                    )
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                    for i, p in enumerate(probs.cpu().numpy()):
                        row = [epoch+1, sample_index, int(labels[i].cpu().item()), int(preds[i].cpu().item())] + list(p)
                        writer.writerow(row)
                        sample_index += 1

            # --- Validation probabilities ---
            if val_input:
                sample_index = 0
                with torch.no_grad(), open(val_prob_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for batch in val_loader:
                        input_ids = tensorize(batch["input_ids"]).long()
                        attention_mask = tensorize(batch["attention_mask"]).long()
                        visual_embeds = tensorize(batch["visual_embeds"]).float().detach()
                        visual_attention_mask = tensorize(batch["visual_attention_mask"]).long()
                        labels = batch["label"]

                        logits = model(
                            input_ids=input_ids,
                            visual_embeds=visual_embeds,
                            visual_attention_mask=visual_attention_mask
                        )
                        probs = torch.softmax(logits, dim=1)
                        preds = logits.argmax(dim=1)

                        for i, p in enumerate(probs.cpu().numpy()):
                            row = [epoch+1, sample_index, int(labels[i].cpu().item()), int(preds[i].cpu().item())] + list(p)
                            writer.writerow(row)
                            sample_index += 1

    # ----------------------------------------------------
    # Save model
    # ----------------------------------------------------
    save_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
