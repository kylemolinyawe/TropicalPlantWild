from math import e
import utils
import preprocessing
from tqdm import tqdm
import json
import torch
import test

images_resized_dir = r'C:\School Files\4TH YEAR 1ST SEM\Thesis II\Implementation\TropicalPlantWild\images_resized'
augment_and_normalize_dir = r'C:\School Files\4TH YEAR 1ST SEM\Thesis II\Implementation\TropicalPlantWild\tensors_augment-and-normalize'
rois_dir = r'C:\School Files\4TH YEAR 1ST SEM\Thesis II\Implementation\TropicalPlantWild\tensors_rois'
rois_features_dir = r'C:\School Files\4TH YEAR 1ST SEM\Thesis II\Implementation\TropicalPlantWild\tensors_rois_features'
epochs = 5
epoch = 1
example_index = 10364

# ---

# ROIs

# Preparing the train - validation - test split
images_resized_paths = utils.get_file_paths(images_resized_dir)
split_paths = utils.stratified_split_paths(images_resized_paths)
train_paths = split_paths[0]
print(train_paths[example_index])
print(len(train_paths))

# # 1 Extracting ROIs
# utils.verify_image_paths(train_paths)
# preprocessing.extract_rois(train_paths)

# ---

# Feature Vectors

# # Building the ROI file path list
# roi_paths = utils.get_file_paths(rois_dir)
# print(len(roi_paths))

# # Sorting the list wherein the index matches the file name
# roi_paths = utils.reorder_list_by_filename(roi_paths)
# utils.check_index_filename_alignment(roi_paths)

# # Visualzing an example ROI .pt file
# example_rois = torch.load(roi_paths[example_index])
# print(roi_paths[example_index])
# utils.pretty_print_tensor(example_rois)
# utils.show_rois(example_rois)

# # 2 Extracting feature vectors from ROIs
# preprocessing.extract_roi_features(roi_paths)

# # Building the ROI features file path list
# roi_features_paths = utils.get_file_paths(rois_features_dir)
# print(len(roi_features_paths))
# roi_features_paths = utils.reorder_list_by_filename(roi_features_paths)
# utils.check_index_filename_alignment(roi_features_paths)
# print(roi_features_paths[example_index])

# classes = utils.load_json_file('subfolders.json')
# probs = utils.generate_dummy_probs(classes)
# utils.visualize_top5(classes, probs)

# utils.print_nonzero_feature_tensor()

# utils.print_tokenization_table()

# test.generate_all_confusion_matrices()