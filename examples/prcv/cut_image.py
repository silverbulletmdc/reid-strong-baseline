"""
使用检测结果对数据集进行裁剪
"""
import json
import cv2
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

source_dir = "/home/dechao/disk1/datasets/prcv/testA/testA"
car_detect_results_path = "/home/dechao/disk1/projects/darknet/car_results.json"
save_dir = "/home/dechao/disk1/datasets/prcv/crop_testA/"

# 将所有的图片名字加入到字典中
detect_dict = defaultdict(lambda: [])
for phase in ["query", "gallery"]:
    for img in os.listdir(os.path.join(source_dir, phase)):
        detect_dict[img] = []

# 读取检测结果
with open(car_detect_results_path, 'r') as f:
    detect_results = json.load(f)

# 过滤检测框
for detect_result in detect_results:
    img_path = detect_result['img_path']

    if detect_result["class"] == "car" and detect_result["prob"] > 0.8:
        detect_dict[img_path].append(detect_result)

for img_path, results in detect_dict.items():
    img_dir, img_name = os.path.split(img_path)
    img_save_path = os.path.join(save_dir, os.path.split(img_dir)[-1], img_name)
    if len(results) == 1:
        result = results[0]

    # 出现多个框使用概率最大的框
    elif len(results) > 1:
        print("{} has {} boxes".format(img_path, len(results)))
        probs = [result["prob"] for result in results]
        result = results[np.argmax(probs)]

    # 没有检测到框则直接使用原图
    elif len(results) == 0:
        print("{} has {} boxes".format(img_path, len(results)))
        result = None

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if result is not None:
        center_box = np.array(result["box"])
        corner_box = np.empty(4)
        corner_box[0] = center_box[0] - center_box[2] / 2
        corner_box[1] = center_box[1] - center_box[3] / 2
        corner_box[2] = center_box[0] + center_box[2] / 2
        corner_box[3] = center_box[1] + center_box[3] / 2
        corner_box = corner_box.astype(np.int)
        crop_img: np.ndarray = img[corner_box[1]:corner_box[3], corner_box[0]:corner_box[2]]
        if crop_img.size > 100 * 100:
            img = crop_img

    print(img_save_path)
    cv2.imwrite(img_save_path, img)
