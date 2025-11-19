import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import VisualBertModel, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# -----------------------------
# Dataset
# -----------------------------
class VisualBERTPrototypeDataset(Dataset):
    def __init__(self, roi_files, labels, text_prompts_per_class, tokenizer, max_length=32):
        self.roi_files = roi_files
        self.labels = labels
        self.text_prompts_per_class = text_prompts_per_class
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.roi_files)

    def __getitem__(self, idx):
        roi_tensor = torch.load(self.roi_files[idx])
        class_idx = self.labels[idx]
        prompt = random.choice(self.text_prompts_per_class[class_idx])
        encoding = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "roi_features": roi_tensor,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(class_idx, dtype=torch.long)
        }

# -----------------------------
# Model
# -----------------------------
class VisualBERTClassifier(nn.Module):
    def __init__(self, num_classes, roi_feature_dim=1024, pretrained_model="uclanlp/visualbert-vqa-coco-pre"):
        super().__init__()
        self.visualbert = VisualBertModel.from_pretrained(pretrained_model, use_safetensors=True)
        self.visual_proj = nn.Linear(roi_feature_dim, self.visualbert.config.visual_embedding_dim)
        self.classifier = nn.Linear(self.visualbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, visual_embeds):
        visual_embeds = self.visual_proj(visual_embeds)
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_visualbert(model_path, roi_files, labels, text_prompts, num_classes, device="cuda", batch_size=32):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = VisualBERTPrototypeDataset(roi_files, labels, text_prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: {
        "roi_features": [item["roi_features"] for item in x],
        "input_ids": torch.stack([item["input_ids"] for item in x]),
        "attention_mask": torch.stack([item["attention_mask"] for item in x]),
        "labels": torch.stack([item["labels"] for item in x])
    })

    model = VisualBERTClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            visual_embeds = torch.stack([b for b in batch["roi_features"]]).to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           visual_embeds=visual_embeds)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Print metrics
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")

    return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec, "macro_f1": f1}

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_auc_visualbert(model_path, roi_files, labels, text_prompts, num_classes, device="cuda", batch_size=32, save_path="roc_auc_shaded_safe.png"):
    from transformers import BertTokenizer
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Dataset & Dataloader
    dataset = VisualBERTPrototypeDataset(roi_files, labels, text_prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: {
        "roi_features": [item["roi_features"] for item in x],
        "input_ids": torch.stack([item["input_ids"] for item in x]),
        "attention_mask": torch.stack([item["attention_mask"] for item in x]),
        "labels": torch.stack([item["labels"] for item in x])
    })

    # Load model
    model = VisualBERTClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Collect predictions
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            visual_embeds = torch.stack([b for b in batch["roi_features"]]).to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           visual_embeds=visual_embeds)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels_tensor.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # One-hot encode labels
    labels_onehot = np.zeros_like(all_probs)
    labels_onehot[np.arange(len(all_labels)), all_labels] = 1

    # Compute ROC curves for each class
    fpr_list = []
    tpr_list = []
    roc_auc = dict()
    for i in range(num_classes):
        try:
            fpr_i, tpr_i, _ = roc_curve(labels_onehot[:, i], all_probs[:, i])
            if np.isnan(fpr_i).any() or np.isnan(tpr_i).any():
                raise ValueError("NaN detected in ROC calculation")
            fpr_list.append(fpr_i)
            tpr_list.append(tpr_i)
            roc_auc[i] = auc(fpr_i, tpr_i)
        except:
            # Skip classes with no positive samples
            print(f"Skipping class {i} due to NaN ROC")
            continue

    if len(fpr_list) == 0:
        raise RuntimeError("No valid classes for ROC calculation.")

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr_list)):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= len(tpr_list)
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # Determine min/max TPR for shading (ignoring NaNs)
    tpr_matrix = np.array([np.interp(all_fpr, fpr_list[i], tpr_list[i]) for i in range(len(tpr_list))])
    tpr_min = np.nanmin(tpr_matrix, axis=0)
    tpr_max = np.nanmax(tpr_matrix, axis=0)

    # Plot ROC curve with shaded area
    plt.figure(figsize=(8, 6))
    plt.fill_between(all_fpr, tpr_min, tpr_max, color='skyblue', alpha=0.4, label='Class ROC range')
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle=':')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - VisualBERT (Shaded, Safe)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    
    print(f"Shaded ROC AUC plot saved to: {save_path}")
    return roc_auc

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_confusion_matrix_visualbert(model_path, roi_files, labels, text_prompts, num_classes, device="cuda", batch_size=32, save_path="confusion_matrix.png"):
    from transformers import BertTokenizer

    # Dataset & Dataloader
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = VisualBERTPrototypeDataset(roi_files, labels, text_prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: {
        "roi_features": [item["roi_features"] for item in x],
        "input_ids": torch.stack([item["input_ids"] for item in x]),
        "attention_mask": torch.stack([item["attention_mask"] for item in x]),
        "labels": torch.stack([item["labels"] for item in x])
    })

    # Load model
    model = VisualBERTClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            visual_embeds = torch.stack([b for b in batch["roi_features"]]).to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           visual_embeds=visual_embeds)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved to: {save_path}")
    return cm


