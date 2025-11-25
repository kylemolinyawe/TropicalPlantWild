from doctest import master
from math import e

import test
import utils
import preprocessing
from tqdm import tqdm
import model
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, VisualBertModel
import os
import vis
import evaluate


img_demo_dir = os.path.join('images_demo')
images_dir = os.path.join('images')
spc10_dir = os.path.join('spc10')
models_dir = os.path.join('models')
baseline_dir = os.path.join('baseline')
spc30_dir = os.path.join('spc30')
main_dir = os.path.join('main')

example_index = 374

# Experiment Params
num_classes = 104
dir = main_dir
num_rois = 5
patience = 3
num_epochs = 20
samples_per_class = 10 # spc
lr = 1e-5
weight_decay = 0 # wd
model_name = '20epochs_10spc_5rois_0wd_1e-5adamW_01wu.pt'
log_probabilities = False
use_scheduler = True



# 1 ROIs
# ---

# Creating the master index as a reference for every sample in the dataset
utils.generate_master_index(images_dir, dir)
train_set, val_set, test_set = utils.load_master_index(dir)
master_index = train_set + val_set + test_set

# Overwriting the master_index with a subset of itself (for experimentation)
print("")
subset = utils.subset_master_index(dir, master_index, samples_per_class=samples_per_class)
train_set, val_set, test_set = utils.load_master_index(dir)


# print("")
# print("Train Set")
# utils.inspect_master_index(train_set)

# print("")
# print("Test Set")
# utils.inspect_master_index(test_set)

print("\nTrain")
# preprocessing.extract_rois_fasterrcnn(dir, 'train', train_set, num_rois)
train_tensors_rois = utils.get_tensor_rois_file_paths(dir, 'train', train_set)
# utils.count_tensors_rois_shape(train_tensors_rois)

print("\nVal")
# preprocessing.extract_rois_fasterrcnn(dir, 'val', val_set, num_rois)
val_tensors_rois = utils.get_tensor_rois_file_paths(dir, 'val', val_set)
# utils.count_tensors_rois_shape(val_tensors_rois)

print("\nTest")
# preprocessing.extract_rois_fasterrcnn(dir, 'test', test_set, num_rois)
test_tensors_rois = utils.get_tensor_rois_file_paths(dir, 'test', test_set)
# utils.count_tensors_rois_shape(test_tensors_rois)


# Visualizing the ROIs per sample
# for item in train_set:
#     vis.plot_rois_on_image(dir, train_set, item['id'])


# 2 Feature Vectors
# ---

print("\nTrain")
# preprocessing.extract_feature_vectors(dir, 'train', train_tensors_rois, train_set)
train_tensors_rois_features = utils.get_tensor_rois_features_file_paths(dir, 'train', train_set)
# utils.count_tensors_rois_shape(train_tensors_rois_features)

print("\nVal")
# preprocessing.extract_feature_vectors(dir, 'val', val_tensors_rois, val_set)
val_tensors_rois_features = utils.get_tensor_rois_features_file_paths(dir, 'val', val_set)
# utils.count_tensors_rois_shape(val_tensors_rois_features)

print("\nTest")
# preprocessing.extract_feature_vectors(dir, 'test', test_tensors_rois, test_set)
test_tensors_rois_features = utils.get_tensor_rois_features_file_paths(dir, 'test', test_set)
# utils.count_tensors_rois_shape(test_tensors_rois_features)



# # 3 Visual Prototypes
# # ---

# preprocessing.construct_visual_prototypes(demo_dir, train_tensors_rois_features, train_set)
# utils.inspect_visual_prototypes(demo_dir)

# # 4 Model Training (VisualBERT)
# #---

# Text prompts
print("\nText Prompts")
text_prompts = utils.load_text_prompts()
print(text_prompts.keys())
model.tokenize_text_prompts(dir, text_prompts)
utils.inspect_tokenized_prompts(dir)

# Labels
print("\nLabels")
labels = utils.get_labels(dir)
print("\nLabel to Integer Mapping")
labels_int = model.map_labels_to_int(labels)


# Preparing VisualBERT input
print("\nTrain VisualBERT Input")
train_visualbert_input = model.prepare_visualbert_input(dir, train_set, train_tensors_rois_features, labels_int, num_rois)
print("\nVal VisualBERT Input")
val_visualbert_input = model.prepare_visualbert_input(dir, val_set, val_tensors_rois_features, labels_int, num_rois)


# # Finding a learning rate
# print("")
# model.find_learning_rate(train_visualbert_input)

# Training VisualBERT
print("")
model.train_visualbert(models_dir, 
                       train_visualbert_input, 
                       val_input=val_visualbert_input,
                       filename=model_name,
                       num_epochs=num_epochs,
                       lr=lr,
                       log_probabilities = log_probabilities,
                       use_scheduler=use_scheduler,
                       patience=patience,
                       weight_decay=weight_decay)


# 5 Evaluation
# ---

# Usage in your demo.py after training:
print("Evaluation")
vis.plot_training_metrics(models_dir, model_name)

test_visualbert_input = model.prepare_visualbert_input(dir, test_set, test_tensors_rois_features, labels_int, num_rois)

print("")

evaluate.per_sample_class_probabilities(models_dir,
                                        test_visualbert_input,
                                        labels_int,
                                        model_name, 
                                        test_set,
                                        num_classes=num_classes)

evaluate.class_metrics(models_dir, model_name)