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
demo_dir = os.path.join('demo')
models_dir = os.path.join('models')
baseline_dir = os.path.join('baseline')

example_index = 374

# Experiment Params
lr=1e-5
num_rois = 5
num_classes = 104
samples_per_class = 10 # spc
num_epochs = 10
model_name = '10epochs_10spc_5rois.pt'



# 1 ROIs
# ---

# Creating the master index as a reference for every sample in the dataset
utils.generate_master_index(images_dir, demo_dir)
train_set, val_set, test_set = utils.load_master_index(demo_dir)
master_index = train_set + val_set + test_set

# Overwriting the master_index with a subset of itself (for experimentation)
print("")
subset = utils.subset_master_index(demo_dir, master_index, samples_per_class=samples_per_class)
train_set, val_set, test_set = utils.load_master_index(demo_dir)


# print("")
# print("Train Set")
# utils.inspect_master_index(train_set)

# print("")
# print("Test Set")
# utils.inspect_master_index(test_set)

print("\nTrain")
# preprocessing.extract_rois_fasterrcnn(demo_dir, 'train', train_set, num_rois)
train_tensors_rois = utils.get_tensor_rois_file_paths(demo_dir, 'train')
utils.count_tensors_rois_shape(train_tensors_rois)

print("\nVal")
# preprocessing.extract_rois_fasterrcnn(demo_dir, 'val', val_set, num_rois)
val_tensors_rois = utils.get_tensor_rois_file_paths(demo_dir, 'val')
utils.count_tensors_rois_shape(val_tensors_rois)

print("\nTest")
# preprocessing.extract_rois_fasterrcnn(demo_dir, 'test', test_set, num_rois)
test_tensors_rois = utils.get_tensor_rois_file_paths(demo_dir, 'test')
utils.count_tensors_rois_shape(test_tensors_rois)


# Visualizing the ROIs per sample
# for item in train_set:
#     vis.plot_rois_on_image(demo_dir, train_set, item['id'])


# 2 Feature Vectors
# ---

print("\nTrain")
# preprocessing.extract_feature_vectors(demo_dir, 'train', train_tensors_rois, train_set)
train_tensors_rois_features = utils.get_tensor_rois_features_file_paths(demo_dir, 'train')
utils.count_tensors_rois_shape(train_tensors_rois_features)

print("\nVal")
# preprocessing.extract_feature_vectors(demo_dir, 'val', val_tensors_rois, val_set)
val_tensors_rois_features = utils.get_tensor_rois_features_file_paths(demo_dir, 'val')
utils.count_tensors_rois_shape(val_tensors_rois_features)

print("\nTest")
# preprocessing.extract_feature_vectors(demo_dir, 'test', test_tensors_rois, test_set)
test_tensors_rois_features = utils.get_tensor_rois_features_file_paths(demo_dir, 'test')
utils.count_tensors_rois_shape(test_tensors_rois_features)



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
model.tokenize_text_prompts(demo_dir, text_prompts)
utils.inspect_tokenized_prompts(demo_dir)

# Labels
print("\nLabels")
labels = utils.get_labels(demo_dir)
print("\nLabel to Integer Mapping")
labels_int = model.map_labels_to_int(labels)


# Preparing VisualBERT input
print("\nTrain VisualBERT Input")
train_visualbert_input = model.prepare_visualbert_input(demo_dir, train_set, train_tensors_rois_features, labels_int, num_rois)
print("\nVal VisualBERT Input")
val_visualbert_input = model.prepare_visualbert_input(demo_dir, val_set, val_tensors_rois_features, labels_int, num_rois)


# Training VisualBERT
print("")
model.train_visualbert(models_dir, 
                       train_visualbert_input, 
                       val_input=val_visualbert_input,
                       filename=model_name,
                       num_epochs=num_epochs,
                       lr=lr,
                       log_probabilities = True)


# 5 Evaluation
# ---

test_visualbert_input = model.prepare_visualbert_input(demo_dir, test_set, test_tensors_rois_features, labels_int, num_rois)

print("")

evaluate.per_sample_class_probabilities(models_dir,
                                        test_visualbert_input,
                                        labels_int,
                                        model_name, 
                                        test_set,
                                        num_classes=num_classes)

evaluate.class_metrics(models_dir, model_name)

