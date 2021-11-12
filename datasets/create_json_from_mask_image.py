import numpy as np
import cv2
import os
import json


# paths to dataset
train_root = "/mnt/disks/hs03/Data_Hand/data/Train_Set"
train_right_hand_label_path = "/mnt/disks/hs03/Data_Hand/data/right_hand_labels_full"
train_left_hand_label_path = "/mnt/disks/hs03/Data_Hand/data/left_hand_labels_full"
test_right_hand_label_path = "/mnt/disks/hs03/Data_Hand/data/right_hand_labels_test/right_hand_labels"
test_left_hand_label_path = "/mnt/disks/hs03/Data_Hand/data/left_hand_labels_test/left_hand_labels"
classes = ['right hand', 'left hand']
idx_train = 0
idx_test = 0
train_dataset_dicts = []
test_dataset_dicts = []
i = 0
files_test = []
for file_name in os.listdir(train_root):
    ex = file_name.split('.')[1]
    if not ex == '.png' and not ex == '.jpg':
        print(f"{file_name} is not image")
        continue
    frame_idx = file_name.split('.')[0]
    record = {}
    img_path = os.path.join(train_root, file_name)
    image = cv2.imread(img_path)

    right_hand_mask_path = os.path.join(train_right_hand_label_path, file_name)
    left_hand_mask_path = os.path.join(train_left_hand_label_path, file_name)

    # process for some image id has two binary mask
    test_right_hand_mask_path_1 = os.path.join(test_right_hand_label_path, frame_idx+'_1.png')
    test_right_hand_mask_path_2 = os.path.join(test_right_hand_label_path, frame_idx+'_2.png')
    test_left_hand_mask_path_1 = os.path.join(test_left_hand_label_path, frame_idx+'_1.png')
    test_left_hand_mask_path_2 = os.path.join(test_left_hand_label_path, frame_idx+'_2.png')

    height, width = image.shape[:2]

    record["file_name"] = file_name
    record["height"] = height
    record["width"] = width
    sset = "train"
    objs = []

    if os.path.exists(test_right_hand_mask_path_1):
        sset = "test"
        right_hand_mask = cv2.imread(test_right_hand_mask_path_1)
        gray = cv2.cvtColor(right_hand_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            obj = {
                "bbox": [x, y, x+w, y+h],
                "segmentation": c.flatten().tolist(),
                "objects": "right hand",
            }
            objs.append(obj)
    if os.path.exists(test_right_hand_mask_path_2):
        sset = "test"
        right_hand_mask = cv2.imread(test_right_hand_mask_path_2)
        gray = cv2.cvtColor(right_hand_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            obj = {
                "bbox": [x, y, x+w, y+h],
                "segmentation": c.flatten().tolist(),
                "objects": "right hand",
            }
            objs.append(obj)
    if os.path.exists(test_left_hand_mask_path_1):
        sset = "test"
        left_hand_mask = cv2.imread(test_left_hand_mask_path_1)
        gray = cv2.cvtColor(left_hand_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            obj = {
                "bbox": [x, y, x+w, y+h],
                "segmentation": c.flatten().tolist(),
                "objects": "left hand",
            }
            objs.append(obj)
    if os.path.exists(test_left_hand_mask_path_2):
        sset = "test"
        left_hand_mask = cv2.imread(test_left_hand_mask_path_2)
        gray = cv2.cvtColor(left_hand_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            obj = {
                "bbox": [x, y, x+w, y+h],
                "segmentation": c.flatten().tolist(),
                "objects": "left hand",
            }
            objs.append(obj)

    if sset != "test":
        if os.path.exists(right_hand_mask_path):
            right_hand_mask = cv2.imread(right_hand_mask_path)
            gray = cv2.cvtColor(right_hand_mask, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)

                obj = {
                    "bbox": [x, y, x+w, y+h],
                    "segmentation": c.flatten().tolist(),
                    "objects": "right hand",
                }
                objs.append(obj)

        if os.path.exists(left_hand_mask_path):
            left_hand_mask = cv2.imread(left_hand_mask_path)
            gray = cv2.cvtColor(left_hand_mask, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                obj = {
                    "bbox": [x, y, x+w, y+h],
                    "segmentation": c.flatten().tolist(),
                    "objects": "left hand",
                }
                objs.append(obj)

    if sset == "train":
        record["image_id"] = idx_train
        idx_train += 1
    else:
        record["image_id"] = idx_test
        idx_test += 1
    record["regions"] = objs
    if sset=="train":
        train_dataset_dicts.append(record)
    else:
        test_dataset_dicts.append(record)
        files_test.append(img_path)
    print(f"{i}: {img_path}")
    i+=1

print(f"train: {idx_train}")
print(f"test: {idx_test}")
with open('train.json', 'w') as outfile:
    json.dump(train_dataset_dicts, outfile)
with open('test.json', 'w') as outfile:
    json.dump(test_dataset_dicts, outfile)





