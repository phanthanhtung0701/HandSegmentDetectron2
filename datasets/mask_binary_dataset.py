import os
import json
import numpy as np
from detectron2.structures import BoxMode

# Data loader for DHY dataset
# For hand dataset, ground truth is mask binary image
def get_hand_dicts(path_json_file, train_path):
    classes = ['right hand', 'left hand']
    dataset_dicts = []
    json_file = os.path.join(path_json_file)
    with open(json_file) as f:
        img_anns = json.load(f)
    # img_anns = list(img_anns.values())
    for annotation in img_anns:
        record = {}
        record["file_name"] = os.path.join(train_path, annotation["file_name"])
        record["image_id"] = annotation["image_id"]
        record["height"] = annotation["height"]
        record["width"] = annotation["width"]

        annos = annotation["regions"]
        objs = []
        if annotation["regions"]:
            for anno in annos:
                object_name = anno['objects']
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [anno["segmentation"]],
                    "category_id": classes.index(object_name),
                    "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# For full hand dataset
def get_hand_dicts_from_via_json(path_json_file, data_path):
    dataset_dicts = []
    with open(path_json_file) as f:
        img_anns = json.load(f)
    img_anns = list(img_anns.values())
    idx = 0
    # classes = ['right hand', 'left hand']
    classes = ['2', '1']
    for annotation in img_anns:
        if annotation["regions"]:
            record = {}
            file_name = annotation['filename']
            img_path = os.path.join(data_path, file_name)

            record["file_name"] = img_path
            record["image_id"] = idx
            record["height"] = 1440
            record["width"] = 1920

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

                category_id = anno['region_attributes']['category_id']
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(category_id),
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            idx += 1
    return dataset_dicts
# a = get_hand_dicts(r'D:\timelap\function\DATN\train.json')


# from detectron2.data import DatasetCatalog, MetadataCatalog
# for d in ["train"]:
#     DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_hand_dicts('/mnt/disks/hs03/Data_Hand/FHAB/Hand_segmetation_annotation/' + d + '.json'))
#     MetadataCatalog.get("my_dataset_" + d).set(thing_classes=['right hand', 'left hand'])
# my_dataset_metadata = MetadataCatalog.get("my_dataset_train")
# print(type(my_dataset_metadata))
# print(my_dataset_metadata)