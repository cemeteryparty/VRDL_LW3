from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2
import sys
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Perform data augmentation"
    )
    parser.add_argument(
        "--input-path", required=True, metavar="/path/to/dataset/",
        help="Path to the original dataset"
    )
    parser.add_argument(
        "--output-path", required=True, metavar="/path/to/dataset/",
        help="Path to save augment results"
    )
    args = parser.parse_args()

    PROBLEM         = "instance_segmentation"
    INPUT_PATH      = args.input_path
    OUTPUT_MODE     = "coco"
    OUTPUT_PATH     = args.output_path
    ANNOTATION_MODE = "coco"
    GENERATION_MODE = "linear"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    augmentor = createAugmentor(
        PROBLEM, ANNOTATION_MODE, OUTPUT_MODE, GENERATION_MODE, INPUT_PATH, {"outputPath": OUTPUT_PATH}
    )

    """ generator """
    transformer = transformerGenerator(PROBLEM)
    # Rotations
    for angle in [0, 90, 180, 270]:
        rotate = createTechnique("rotate", {"angle": angle})
        augmentor.addTransformer(transformer(rotate))

    # Flips
    flip1 = createTechnique("flip", {"flip": 1})
    augmentor.addTransformer(transformer(flip1))
    flip0 = createTechnique("flip", {"flip": 0})
    augmentor.addTransformer(transformer(flip0))

    # Gaussian blurring
    Gaussian = createTechnique("gaussian_blur", {"kernel": 3})
    augmentor.addTransformer(transformer(Gaussian))

    # Average blurring
    average_blurring = createTechnique("average_blurring", {"kernel": 3})
    augmentor.addTransformer(transformer(average_blurring))

    # Crop
    crop_0 = createTechnique("crop", {"percentage": 0.25, "startFrom": "CENTER"})
    augmentor.addTransformer(transformer(crop_0))
    crop_1 = createTechnique("crop", {"percentage": 0.5, "startFrom": "CENTER"})
    augmentor.addTransformer(transformer(crop_1))
    crop_2 = createTechnique("crop", {"percentage": 0.25, "startFrom": "TOPLEFT"})
    augmentor.addTransformer(transformer(crop_2))
    crop_3 = createTechnique("crop", {"percentage": 0.5, "startFrom": "TOPLEFT"})
    augmentor.addTransformer(transformer(crop_3))
    crop_4 = createTechnique("crop", {"percentage": 0.25, "startFrom": "BOTTOMRIGHT"})
    augmentor.addTransformer(transformer(crop_4))
    crop_5 = createTechnique("crop", {"percentage": 0.5, "startFrom": "BOTTOMRIGHT"})
    augmentor.addTransformer(transformer(crop_5))

    augmentor.applyAugmentation()
