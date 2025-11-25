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
    print("Prepared VisualBERT Input:")
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
import warnings

def evaluate_and_log(
    model,
    data_loader,
    criterion,
    device,
    num_classes,
    epoch,
    metrics_path,
    prob_path=None,
    log_probabilities=False
):
    """
    Evaluate the model on a dataset and log metrics to CSV. Optionally log per-sample probabilities.
    """
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    model.eval()
    running_loss = 0.0
    true_labels = []
    preds_labels = []

    # Compute metrics helper
    def compute_metrics(true_labels, preds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            acc = accuracy_score(true_labels, preds)
            macro_recall = recall_score(true_labels, preds, average="macro", zero_division=0)
            macro_precision = precision_score(true_labels, preds, average="macro", zero_division=0)
            macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        return acc, macro_recall, macro_precision, macro_f1

    sample_index = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).long()
            visual_embeds = batch["visual_embeds"].to(device).float().detach()
            visual_attention_mask = batch["visual_attention_mask"].to(device).long()
            labels = batch["label"].to(device).long()

            logits = model(
                input_ids=input_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask
            )

            loss = criterion(logits, labels)
            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            true_labels.extend(labels.cpu().tolist())
            preds_labels.extend(preds.cpu().tolist())

            # Per-sample probability logging
            if log_probabilities and prob_path:
                probs = torch.softmax(logits, dim=1)
                with open(prob_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for i, p in enumerate(probs.cpu().numpy()):
                        row = [epoch + 1, sample_index, int(labels[i].cpu().item()), int(preds[i].cpu().item())] + list(p)
                        writer.writerow(row)
                        sample_index += 1

    avg_loss = running_loss / len(data_loader)
    acc, recall, precision, f1 = compute_metrics(true_labels, preds_labels)

    # Write epoch metrics
    with open(metrics_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch + 1, avg_loss, acc, recall, precision, f1])

    return avg_loss, acc, recall, precision, f1

def find_learning_rate(visualbert_input, batch_size=4, start_lr=1e-7, end_lr=1, num_iters=100,
                            criterion=None, device=None, seed=42, safe_frac=0.1, max_lr=1e-3):
    """
    Safe Learning Rate Finder for VisualBERTWithClassifier.
    Automatically picks a conservative LR suitable for transformers.
    """
    import math
    import random
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Deterministic
    # -------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------------------------------
    # Hardcoded model
    # -------------------------------
    num_classes = max(visualbert_input["labels"]) + 1
    model = VisualBERTWithClassifier(num_classes)
    model.to(device)
    model.train()

    # -------------------------------
    # Dataset and loader
    # -------------------------------
    dataset = VisualBERTDataset(visualbert_input)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=start_lr)

    # Exponential LR increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iters)

    lrs = []
    losses = []
    iter_count = 0

    for batch in loader:
        if iter_count >= num_iters:
            break

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device).long()
        attention_mask = batch["attention_mask"].to(device).long()
        visual_embeds = batch["visual_embeds"].to(device).float().detach()
        visual_attention_mask = batch["visual_attention_mask"].to(device).long()
        labels = batch["label"].to(device).long()

        logits = model(
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # record
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(loss.item())

        print(f"Iter {iter_count+1}/{num_iters}: LR = {current_lr:.2e}, Loss = {loss.item():.4f}")

        # update LR
        optimizer.param_groups[0]["lr"] *= lr_mult
        iter_count += 1

    # -------------------------------
    # Suggest safe LR
    # -------------------------------
    losses_np = np.array(losses)
    lrs_np = np.array(lrs)
    
    grads = np.gradient(losses_np)
    steepest_idx = np.argmin(grads)
    best_lr_raw = lrs_np[steepest_idx]

    # Apply safe fraction and max cap
    best_lr = min(best_lr_raw * safe_frac, max_lr)
    print(f"\nRaw steepest-slope LR: {best_lr_raw:.2e}")
    print(f"Suggested safe learning rate: {best_lr:.2e}")

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(lrs, losses, label="Loss")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("VisualBERT Learning Rate Finder (Safe)")
    plt.axvline(best_lr, color='r', linestyle='--', label=f'Safe LR: {best_lr:.2e}')
    plt.legend()
    plt.show()

    return lrs, losses, best_lr

import math

def train_visualbert(
    directory_path: str,
    visualbert_input,
    val_input=None,
    filename: str = "visualbert_model.pt",
    num_epochs=10,
    batch_size=4,
    lr=5e-5,
    patience=3,
    weight_decay=0,
    use_scheduler: bool = True,      # enables Warmup+Cosine
    log_probabilities: bool = False,
    seed: int = 42
):
    """
    Train VisualBERT + classifier with warmup + cosine LR schedule (per batch),
    full metrics logging, deterministic behavior, probability logging, and early stopping.
    """

    import os, csv, math, random
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # -------------------------------
    # Deterministic seeds
    # -------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------------------------------
    # Setup
    # -------------------------------
    num_classes = max(visualbert_input["labels"]) + 1
    print(f"Inferred number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisualBERTWithClassifier(num_classes).to(device)

    # -------------------------------
    # Data loaders
    # -------------------------------
    train_dataset = VisualBERTDataset(visualbert_input)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    if val_input:
        val_dataset = VisualBERTDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -------------------------------
    # Scheduler: Warmup + Cosine Decay
    # -------------------------------
    scheduler = None
    if use_scheduler:
        total_training_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_training_steps)  # 10% warmup

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -------------------------------
    # Utility: Move stacked tensors to device
    # -------------------------------
    def tensorize(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.stack(x).to(device)

    # -------------------------------
    # File setup
    # -------------------------------
    model_name = os.path.splitext(filename)[0]
    model_folder = os.path.join(directory_path, model_name)
    os.makedirs(model_folder, exist_ok=True)

    train_metrics_path = os.path.join(model_folder, "train_model_metrics.csv")
    val_metrics_path = os.path.join(model_folder, "val_model_metrics.csv")

    with open(train_metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "accuracy", "macro_recall", "macro_precision", "macro_f1"])

    with open(val_metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "accuracy", "macro_recall", "macro_precision", "macro_f1"])

    # Probability logging
    if log_probabilities:
        train_prob_path = os.path.join(model_folder, "train_per_sample_class_probabilities.csv")
        val_prob_path = os.path.join(model_folder, "val_per_sample_class_probabilities.csv") if val_input else None

        header = ["epoch", "sample_index", "true_label", "pred_label"] + [
            f"class_{i}_prob" for i in range(num_classes)
        ]

        with open(train_prob_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

        if val_input and val_prob_path:
            with open(val_prob_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    # -------------------------------
    # Early Stopping setup
    # -------------------------------
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    # -------------------------------
    # Training Loop
    # -------------------------------
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

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

            # Step scheduler PER BATCH
            if scheduler is not None:
                scheduler.step()
                global_step += 1

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # -------------------------------
        # Print current learning rate per epoch
        # -------------------------------
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch+1}] Current LR: {current_lr:.2e}")

        # -------------------------------
        # Metrics (train)
        # -------------------------------
        train_metrics = evaluate_and_log(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
            metrics_path=train_metrics_path,
            prob_path=train_prob_path if log_probabilities else None,
            log_probabilities=log_probabilities
        )

        print(
            f"[Epoch {epoch+1}] TRAIN → Loss: {train_metrics[0]:.4f} "
            f"| Acc: {train_metrics[1]:.4f} | Recall: {train_metrics[2]:.4f} "
            f"| Precision: {train_metrics[3]:.4f} | F1: {train_metrics[4]:.4f}"
        )

        # -------------------------------
        # Metrics (validation)
        # -------------------------------
        if val_input:
            val_metrics = evaluate_and_log(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                epoch=epoch,
                metrics_path=val_metrics_path,
                prob_path=val_prob_path if log_probabilities else None,
                log_probabilities=log_probabilities
            )

            print(
                f"[Epoch {epoch+1}] VAL → Loss: {val_metrics[0]:.4f} "
                f"| Acc: {val_metrics[1]:.4f} | Recall: {val_metrics[2]:.4f} "
                f"| Precision: {val_metrics[3]:.4f} | F1: {val_metrics[4]:.4f}"
            )

            # -------------------------------
            # Early Stopping check
            # -------------------------------
            current_val_loss = val_metrics[0]

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                counter = 0
                best_model_state = model.state_dict()
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter} out of {patience}")
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    # -------------------------------
    # Save model (best weights if early stopping)
    # -------------------------------
    save_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
