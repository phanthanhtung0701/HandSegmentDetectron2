# /opt/pycode/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

import detectron2
from detectron2.utils.logger import setup_logger

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, sem_seg_evaluation, COCOPanopticEvaluator, DatasetEvaluators

import datasets.mask_binary_dataset as my_data_DHY
import datasets.my_dataset as my_data_FHAB


from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "test"]:
    DatasetCatalog.register("my_hand_dataset_" + d, lambda d=d: my_data_DHY.get_hand_dicts('/mnt/disks/hs03/Data_Hand/data/' + d + '.json'))
    MetadataCatalog.get("my_hand_dataset_" + d).set(thing_classes=['right hand', 'left hand'])


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_hand_dataset_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
# cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # initialize from model zoo
# cfg.MODEL.WEIGHTS = "/opt/pycode/detectron2/output/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.CHECKPOINT_PERIOD = 10000
cfg.SOLVER.MAX_ITER = 20000
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # number class


sset = "train"
if sset == "train":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if sset == "test":
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    evaluator = COCOEvaluator("my_hand_dataset_test", False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_hand_dataset_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
