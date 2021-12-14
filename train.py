""" Train a Mask-RCNN model for image segmentation """

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


""" https://github.com/facebookresearch/detectron2/issues/2114 """
from detectron2.engine.train_loop import HookBase
import logging


class BestCheckpointer(HookBase):

    def __init__(self, log_path):
        self.LOGPATH = log_path
        self.logstr = ""

    def before_train(self):
        self.best_metric = np.inf
        self.logger = logging.getLogger("detectron2.trainer")
        self.logger.info("Running best check pointer")

    def after_step(self):
        monitor = "total_loss"
        if monitor in self.trainer.storage._history:
            eval_metric, batches = \
                self.trainer.storage.history(monitor)._data[-1]

            if self.best_metric > eval_metric:
                self.logstr = "Iter {}: {} improved from {:.6f} to {:.6f}".format(
                    self.trainer.iter, monitor, self.best_metric, eval_metric
                )
                self.best_metric = eval_metric
                self.logger.info(self.logstr)
                self.trainer.checkpointer.save(f"best_{monitor}_model")

    def after_train(self):
        fd = open(os.path.join(self.LOGPATH, "BestCheckpointer.log"), "w")
        fd.write(self.logstr + "\n")
        fd.close()


class BestTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer(self.cfg.OUTPUT_DIR))
        return ret

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a Mask-RCNN model for image segmentation"
    )
    parser.add_argument(
        "--dataset-path", required=True, metavar="/path/to/dataset/",
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--anno-path", required=True, metavar="/path/to/ANNO_NAME.json",
        help="Path to the annotation file"
    )
    parser.add_argument(
        "--config", required=True, metavar="/path/to/CONFIG.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--output-path", required=True, metavar="/path/to/dataset/",
        help="Path to save model results"
    )
    """ Parse arguments """
    args = parser.parse_args()
    DATASET_PATH = args.dataset_path
    ANNO_PATH = args.anno_path
    CONFIG_PATH = args.config
    SAVE_PATH = args.output_path

    """ Register Dataset """
    setup_logger()
    register_coco_instances("trainset", {}, ANNO_PATH, DATASET_PATH)
    metadata = MetadataCatalog.get("trainset")
    dataset_dicts = DatasetCatalog.get("trainset")

    """ Configuration """
    # Default value:
    #    https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
    #    https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.OUTPUT_DIR = SAVE_PATH

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)
    # cfg.MODEL.WEIGHTS = os.path.join(Pre-trained PATH)

    cfg.DATASETS.TRAIN = ("trainset",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    # Warm up Lerning rate, iter 0~STEPS[0]: Linear increase lrate to BASE_LR
    #    iter reaches STEPS[0], STEPS[1]: decrease lrate
    cfg.SOLVER.STEPS = (3000, 4000) 
    cfg.SOLVER.MAX_ITER = 6000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    cfg.INPUT.MIN_SIZE_TRAIN = (500, 550, 600, 650, 700, 750, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1000

    """ Train Process """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Dump all configuation to a .yaml file
    fd = open(os.path.join(cfg.OUTPUT_DIR, "train_conf.yaml"), "w")
    fd.write(cfg.dump())
    fd.close()

    trainer = BestTrainer(cfg)  # need gpu support
    trainer.resume_or_load(resume=False)
    trainer.train()
