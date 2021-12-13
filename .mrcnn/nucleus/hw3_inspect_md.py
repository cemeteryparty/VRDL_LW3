import os
import re
import sys
import math
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import nucleus

""" # Configurations """
DATASET_DIR = "dataset"
SUBSET = "train"
fnames = os.listdir(os.path.join(DATASET_DIR, SUBSET))

config = nucleus.NucleusInferenceConfig(n_images=len(fnames), n_val_images=0)
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"
LOGS_DIR = "/mnt/left/phlai_DATA/pykiras/logs"

""" # Load Validation Dataset """
dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, subset="train")
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info["name"]))

""" # Load Model """
#with tf.device(DEVICE):
#    model = modellib.MaskRCNN(mode="inference",model_dir=LOGS_DIR, config=config)
model = modellib.MaskRCNN(mode="inference",model_dir=LOGS_DIR, config=config)
MODEL_PATH = os.path.join(LOGS_DIR, "models")
weights_path = os.path.join(MODEL_PATH, "mrcnn_nuc_ep19.h5")

print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

""" # Run Detection """
image_id = np.random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(
        info["source"], info["id"], image_id, dataset.image_reference(image_id)
    )
)
print("Original image shape: ", modellib.parse_image_meta(np.expand_dims(image_meta, axis=0))["original_image_shape"][0])

""" Run object detection """
results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

# Display results
r = results[0]
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# """ Compute AP over range 0.5 to 0.95 and print it """
# utils.compute_ap_range(
#     gt_bbox, gt_class_id, gt_mask, 
#     r["rois"], r["class_ids"], r["scores"], r["masks"]
# )


# fig = plt.figure(figsize=(20, 20))
# ax = plt.subplot(111)
# visualize.display_differences(
#     image, gt_bbox, gt_class_id, gt_mask,
#     r["rois"], r["class_ids"], r["scores"], r["masks"],
#     dataset.class_names, ax=ax, show_box=False, show_mask=False,
#     iou_threshold=0.5, score_threshold=0.5
# )
# plt.show()

""" Display predictions and ground truth """
# fig = plt.figure(figsize=(20, 40))
# ax = plt.subplot(121)
# visualize.display_instances(
#     image, r["rois"], r["masks"], r["class_ids"], 
#     dataset.class_names, r["scores"], ax=ax,
#     show_bbox=False, show_mask=False, title="Predictions"
# )
# ax = plt.subplot(122)
# visualize.display_instances(
#     image, gt_bbox, gt_mask, gt_class_id, dataset.class_names, ax=ax,
#     show_bbox=False, show_mask=False, title="Ground Truth"
# )
# plt.savefig("results/{}.png".format(info["id"]))
