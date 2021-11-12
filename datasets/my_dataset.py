import os
import numpy as np
import json
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

# Data loader for FHAB dataset
# Change this path
root = '/mnt/disks/hs03/Data_Hand/FHAB'
# hand_annotation_root = os.path.join(root, 'Hand_segmetation_annotation')
file_root = os.path.join(root, 'Video_files')


def get_hand_dicts(hand_annotation_root):
    classes = ['hand']
    # name_dict = {"hand": 1}
    dataset_dicts = []
    idx = 0
    for subject in os.listdir(hand_annotation_root):
        for action_name in os.listdir(os.path.join(hand_annotation_root, subject)):
            for seq_idx in os.listdir(os.path.join(hand_annotation_root, subject, action_name)):
                dir_json = os.path.join(hand_annotation_root, subject, action_name, seq_idx)
                hand_annotation_path = os.path.join(hand_annotation_root, subject, action_name, seq_idx,
                                                    'via_project_json.json')
                if os.path.exists(hand_annotation_path):
                    json_file = hand_annotation_path
                    with open(json_file) as f:
                        img_anns = json.load(f)
                    img_anns = list(img_anns.values())
                    for annotation in img_anns:
                        if annotation["regions"]:
                            record = {}
                            file_name = annotation['filename']
                            img_path = os.path.join(file_root, subject, action_name, seq_idx, 'color', file_name)
                            # filename = os.path.join(directory, img_anns["imagePath"])

                            record["file_name"] = img_path
                            record["image_id"] = idx
                            record["height"] = 1920
                            record["width"] = 1080

                            annos = annotation["regions"]
                            objs = []
                            for anno in annos:
                                # px = [a[0] for a in anno['points']]
                                # py = [a[1] for a in anno['points']]
                                # poly = [(x, y) for x, y in zip(px, py)]
                                # poly = [p for x in poly for p in x]
                                px = anno['shape_attributes']['all_points_x']
                                py = anno['shape_attributes']['all_points_y']
                                poly = [(x, y) for x, y in zip(px, py)]
                                poly = [p for x in poly for p in x]

                                object_name = anno['region_attributes']['objects']
                                obj = {
                                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "segmentation": [poly],
                                    "category_id": classes.index(object_name),
                                    "iscrowd": 0
                                }
                                objs.append(obj)
                            record["annotations"] = objs
                            dataset_dicts.append(record)
                            idx += 1
    print(len(dataset_dicts))
    return dataset_dicts


### uncomment to test data loader
# from detectron2.data import DatasetCatalog, MetadataCatalog
# for d in ["train"]:
#     DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_hand_dicts('/mnt/disks/hs03/Data_Hand/FHAB/Hand_segmetation_annotation/' + d))
#     MetadataCatalog.get("my_dataset_" + d).set(thing_classes=['hand'])
# my_dataset_metadata = MetadataCatalog.get("my_dataset_train")
# print(type(my_dataset_metadata))
# print(my_dataset_metadata)