# import some common libraries
from pycocotools.mask import encode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import json
import cv2
import os
# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Submission file"
    )
    parser.add_argument(
        "--model-path", required=True, metavar="/path/to/MODEL.pth",
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--testset-path", required=True, metavar="/path/to/dataset/",
        help="Path to thetesting dataset"
    )
    parser.add_argument(
        "--testset-info", required=True, metavar="/path/to/TEST_INFO.json",
        help="Path to the testing information"
    )
    parser.add_argument(
        "--output-path", required=True, metavar="/path/to/dataset/",
        help="Path to save augment results"
    )
    parser.add_argument(
        "--config", required=True, metavar="/path/to/CONFIG.yaml",
        help="Path to the training configuration file"
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    TESTSET_PATH = args.testset_path
    TESTSET_INFO_PATH = args.testset_info
    SAVE_PATH = args.output_path

    CONFIG_PATH = args.config

    """ Configuration """
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.MAX_SIZE = 1000
    cfg.TEST.AUG.MIN_SIZES = (500, 550, 600, 650, 700, 750, 800)

    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    predictor = DefaultPredictor(cfg)

    """ Inference """
    dataset_path = TESTSET_PATH
    fd = open(TESTSET_INFO_PATH, "r")
    content = json.load(fd)
    mapper = {}
    for entry in content:
        mapper[entry["id"]] = entry
    fd.close()

    result_to_json = []
    for image_id in range(1, 7):
        imname = mapper[image_id]["file_name"]
        impath = os.path.join(dataset_path, imname)
        print("\rProcess on {}".format(imname), end="")
        im = cv2.imread(impath)
        outputs = predictor(im)["instances"].to("cpu")

        bboxes = outputs._fields["pred_boxes"].tensor.numpy()
        scores = outputs._fields["scores"].numpy()
        masks = outputs._fields["pred_masks"].numpy()

        # print(bboxes.shape, scores.shape, masks.shape)
        num_of_instance = scores.size
        for i in range(num_of_instance):
            print("\rProcess on {}, instance {}/{}".format(
                imname, i + 1, num_of_instance), end=""
            )
            x = float(bboxes[i][0])
            y = float(bboxes[i][1])
            w = abs(float(bboxes[i][2]) - x)
            h = abs(float(bboxes[i][3]) - y)
            seg = encode(np.asfortranarray(masks[i]))
            seg["counts"] = seg["counts"].decode("utf-8")
            ins_info = {}
            ins_info["image_id"] = image_id
            ins_info["category_id"] = 1
            ins_info["score"] = float(scores[i])
            ins_info["bbox"] = [x, y, w, h]
            ins_info["segmentation"] = seg

            result_to_json.append(ins_info)
        print()

    os.makedirs(SAVE_PATH, exist_ok=True)
    json_object = json.dumps(result_to_json, indent=4)
    with open(os.path.join(SAVE_PATH, "answer.json"), "w") as outfile:
        outfile.write(json_object)

    """ Generate Figure """
    fig = plt.figure(figsize=(30, 20))
    for image_id in range(1, 7):
        imname = mapper[image_id]["file_name"]
        plt.subplot(2, 3, image_id)
        impath = os.path.join(dataset_path, imname)
        im = cv2.imread(impath)
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            scale=1,
            # remove the colors of unsegmented pixels
            instance_mode=ColorMode.IMAGE_BW
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(v.get_image())
    plt.savefig(os.path.join(SAVE_PATH, "result.png"))
