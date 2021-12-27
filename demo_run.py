# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from .demo.predictor import VisualizationDemo


config_file=""
input_path = ""
output = ""
confidence_threshold=0.75
opts = []

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TEST = ("my_hand_dataset_train",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Lab/data/Hand_Data/DHY/checkpoint/model_0004999.pth"
    # cfg.merge_from_list(opts)
    return cfg


mp.set_start_method("spawn", force=True)
setup_logger(name="fvcore")
logger = setup_logger()

cfg = setup_cfg()

demo = VisualizationDemo(cfg)

# files = np.load(input_path)
input_path = glob.glob(os.path.expanduser(input_path))
for path in tqdm.tqdm(input_path, disable=not output):
    out_filename = os.path.join(output, os.path.basename(path))
    if os.path.exists(out_filename):
        continue
    # use PIL, to be consistent with evaluation
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    logger.info(
        "{}: {} in {:.2f}s".format(
            path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    if output:
        if os.path.isdir(output):
            assert os.path.isdir(output), output
            out_filename = os.path.join(output, os.path.basename(path))
        else:
            assert len(input_path) == 1, "Please specify a directory with output"
            out_filename = output
        visualized_output.save(out_filename)