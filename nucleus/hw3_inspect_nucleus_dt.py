import os
import sys
import json
import time

import numpy as np
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
from imgaug import augmenters as iaa

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log

import nucleus

# class NoResizeConfig(nucleus.NucleusConfig):
#     IMAGE_RESIZE_MODE = "none"
class InfConfig(nucleus.NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

def image_stats(image_id):
    """Returns a dict of stats for one image."""
    image = dataset.load_image(image_id)
    mask, _ = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)
    # Sanity check
    assert mask.shape[:2] == image.shape[:2]
    # Return stats dict
    return {
        "id": image_id,
        "shape": list(image.shape),
        "bbox": [
                [b[2] - b[0], b[3] - b[1]] for b in bbox if b[2] - b[0] > 1 and b[3] - b[1] > 1
                # Exclude nuclei with 1 pixel width or height (often on edges)
            ],
        "color": np.mean(image, axis=(0, 1)),
    }

""" Configuation """
# Dataset directory
DATASET_DIR = os.path.join("dataset")
fnames = os.listdir(os.path.join(DATASET_DIR, "train"))
# from sklearn.model_selection import train_test_split
# TRA_IMAGE_IDS, VAL_IMAGE_IDS = train_test_split(fnames, test_size=1/6, random_state=1000)
VAL_IMAGE_IDS = ['TCGA-HE-7128-01Z-00-DX1', 'TCGA-HE-7130-01Z-00-DX1', 'TCGA-21-5784-01Z-00-DX1', 'TCGA-21-5786-01Z-00-DX1']
TRA_IMAGE_IDS = list(set(fnames) - set(VAL_IMAGE_IDS))
    
config = InfConfig(len(TRA_IMAGE_IDS), 0)
# config.display()
print(config.IMAGE_RESIZE_MODE)

""" Load Dataset """
dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, subset="train", selects=TRA_IMAGE_IDS)
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info["name"]))


""" Display Samples """
for image_id in dataset.image_ids:
	image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
	    dataset, config, image_id, use_mini_mask=False
	)
	log("molded_image", image)
	log("mask", mask)
	s = image_stats(image_id)
	nuc_shape = np.array([b for b in s["bbox"]])
	nuc_area = nuc_shape[:, 0] * nuc_shape[:, 1]

	fig = plt.figure(figsize=(60, 30))
	ax = plt.subplot(121)
	visualize.display_instances(
	    image, bbox, mask, class_ids, dataset.class_names, show_bbox=False, ax=ax
	)
	ax = plt.subplot(122)
	_ = plt.hist2d(nuc_shape[:, 1], nuc_shape[:, 0], bins=20, cmap="Blues")
	plt.title(
	    "{} - mu={:.2f}  range:{:.2f}~{:.2f}".format(
	        nuc_shape.shape[0], np.mean(nuc_area), np.min(nuc_area), np.max(nuc_area)
	    ),
	    fontsize=35
	)
	plt.savefig("results/{}.png".format(dataset.image_info[image_id]["id"]))
