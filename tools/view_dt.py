from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import sys

if len(sys.argv) < 2:
	print("Usage: python3 view_dt.py DATASET_PATH")
	exit(1)
base_path = sys.argv[1]

image_directory = f"{base_path}/"
annotation_file = f"{base_path}/annotation.json"

example_coco = COCO(annotation_file)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category["name"] for category in categories]
print("Custom COCO categories: \n{}\n".format(category_names))
category_names = set([category["supercategory"] for category in categories])
print("Custom COCO supercategories: \n{}".format(category_names))

category_ids = example_coco.getCatIds(catNms=["nucleus"])
image_ids = example_coco.getImgIds(catIds=category_ids)

fig = plt.figure(figsize=(20, 20))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    image_data = example_coco.loadImgs(np.random.choice(image_ids).item())[0]
    image = io.imread(image_directory + image_data["file_name"])
    plt.imshow(image)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data["id"], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    example_coco.showAnns(annotations)
    plt.axis("off")
plt.savefig(f"results/{base_path}.png")
