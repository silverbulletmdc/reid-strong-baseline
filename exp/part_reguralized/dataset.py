from torch.utils.data import Dataset
import numpy as np
import math
from data.datasets.dataset_loader import ImageDataset
from utils.iotools import read_nori_image, read_image
import nori2 as nori
import json
from pprint import pprint


class PartReguralizedDataset(ImageDataset):

    def __init__(self, metas, transform=None, use_nori=True):
        """

        :param metas: list[
                    dict{
                        filename: str,
                        nori_id: str,
                        boxes: [(class, prob, xtl, ytl, xbr, ybr)]
                        cam_id: int,
                        id: int
        } ] The boxes is based on original size.
        :param transform:
        """
        if use_nori:
            base_meta = [(item['nori_id'], item['id'], item['cam_id']) for item in metas]
        else:
            base_meta = [(item['img_path'], item['id'], item['cam_id']) for item in metas]
        super(PartReguralizedDataset, self).__init__(base_meta, transform)
        self.metas = metas
        self.use_nori = use_nori

    def __getitem__(self, index):
        nori_id, pid, camid = self.dataset[index]
        if self.use_nori:
            img = read_nori_image(nori_id)
        else:
            img = read_image(nori_id)

        original_shape = np.array(img.size)  # width, height

        if self.transform is not None:
            img = self.transform(img)

        transform_shape = np.array(img.size)

        boxes = self.metas[index]['boxes']
        boxes = transform_boxes(boxes, original_shape, transform_shape)

        return {
            "img": img,  # PIL.Image
            "pid": pid,  # int
            "camid": camid,  # int
            "nori_id": nori_id,  # str
            "boxes": boxes  # np.array [3,4] xtl,ytl,xrb, yrb
        }


def transform_boxes(boxes, original_shape, transform_shape):
    """
    将原始boxes转化为降采样16倍后的ROI坐标

    :param boxes:
    :return: window_box, left_light_box, right_light_box, (N, 4). Stacked together for further extension.
    """

    window_boxes = [box for box in boxes if box[0] == 'window']
    light_boxes = [box for box in boxes if box[0] == 'light']

    # hard code平均框。该坐标是源代码中16*16特征图上的坐标，转化到目标坐标
    avg_window_box = np.array([0, 3, 15, 7]).reshape((2, 2)) / 16 * transform_shape / 16
    if len(window_boxes) == 0:
        window_box = avg_window_box
    else:
        raw_window_box = window_boxes[0]
        class_, prob, *window_box = raw_window_box
        window_box = np.array(window_box).reshape((2, 2)) / original_shape * transform_shape / 16

    window_box[1, :] = np.ceil(window_box[1, :])
    window_box[0, :] = np.round(window_box[0, :])

    if len(light_boxes) == 0:
        left_light_box = np.array([0, 11, 5, 16]).reshape((2, 2)) / 16 * transform_shape / 16
        right_light_box = np.array([11, 11, 16, 16]).reshape((2, 2)) / 16 * transform_shape / 16

    # 如果只有一个车灯则重复两个
    elif len(light_boxes) == 1:
        raw_light_box = light_boxes[0]
        class_, prob, *light_box = raw_light_box
        light_box = np.array(light_box).reshape((2, 2)) / original_shape * transform_shape / 16
        left_light_box = light_box
        right_light_box = np.copy(light_box)

    elif len(light_boxes) >= 2:
        raw_left_box = light_boxes[0]
        raw_right_box = light_boxes[1]
        class_, prob, *left_box = raw_left_box
        left_light_box = np.array(left_box).reshape((2, 2)) / original_shape * transform_shape / 16

        class_, prob, *right_box = raw_right_box
        right_light_box = np.array(right_box).reshape((2, 2)) / original_shape * transform_shape / 16

    left_light_box[1, :] = np.ceil(left_light_box[1, :])
    left_light_box[0, :] = np.round(left_light_box[0, :])
    right_light_box[1, :] = np.ceil(right_light_box[1, :])
    right_light_box[0, :] = np.round(right_light_box[0, :])

    window_box = window_box.reshape(4)
    left_light_box = left_light_box.reshape(4)
    right_light_box = right_light_box.reshape(4)

    if left_light_box[0] > right_light_box[0]:
        left_light_box, right_light_box = right_light_box, left_light_box

    return np.stack([window_box, left_light_box, right_light_box])


def make_dataset():
    with open('data/veri776_with_box.json', 'r') as f:
        metas = json.load(f)
    dataset = PartReguralizedDataset(metas["train"])
    return dataset
