# Hand segmentation with detectron2
1. Download and setup detectron2
2. Copy file on datasets into datasets folder of detectron2 folder
3. Change default train config _C.INPUT.RANDOM_FLIP from "horizontal" to "none" at detectron2/detectron2/config/defaults.py
4. With data DHY, create json file from data to prepare for training, see in file create_json_from_mask_image
5. Training in file hand_segment_train

## Pretrained file
1. Hand segmentation: https://drive.google.com/file/d/17T_4Ik45I9nWGnE6kGlJ875z_uI-U4bE/view?usp=sharing
2. Full hand segmentation: https://drive.google.com/file/d/1Ki8dJuecuPcu-b5swMoLu2u-AlHpCWwp/view?usp=sharing