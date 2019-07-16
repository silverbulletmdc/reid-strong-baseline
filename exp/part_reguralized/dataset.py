from torch.utils.data import Dataset
import numpy as np
import math
from data.datasets.dataset_loader import ImageDataset
from utils.iotools import read_nori_image, read_image
import nori2 as nori
import json


def veri776_adaptor(dataset, json_path):
    """
    将原工程使用的dataset(tuple(img_path, pid, cam_id))转化为新的格式，并增加nori信息
    :param dataset:
    :param json_path:
    :return:
    """
    with open(json_path, 'r') as f:
        box_json = json.load(f)

    s3_root = "s3://normal/veri776"
    nr = nori.open(s3_root)
    filename_noriid_dict = {}
    for nid, _, metas in nr.scan(scan_data=False):
        filename_noriid_dict[metas['filename']] = nid

    output_dict = []
    for box_meta in box_json:

        box_meta['nori_id'] = filename_noriid_dict[box_meta['img_name']]


class PartReguralizedDataset(ImageDataset):

    def __init__(self, metas, transform=None, use_nori=True):
        """

        :param metas: list[
                    dict{
                        img_path: str,
                        nori_id: str,
                        boxes: [(class, prob, xtl, ytl, xbr, ybr)]
                        cam_id: int,
                        id: int
        } ] The boxes is based on original size.
        :param transform:
        """
        if use_nori:
            base_meta = [(item['nori_id'], item['id'], item['camid']) for item in metas]
        else:
            base_meta = [(item['img_path'], item['id'], item['camid']) for item in metas]
        super(PartReguralizedDataset, self).__init__(base_meta, transform)
        self.metas = metas
        self.use_nori = use_nori

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        if self.use_nori:
            img = read_nori_image(img_path)
        else:
            img = read_image(img_path)

        original_shape = np.array(img.size)  # width, height

        if self.transform is not None:
            img = self.transform(img)

        transform_shape = np.array(img.size)

        boxes = self.metas[index]['boxes']
        boxes = transform_boxes(boxes, original_shape, transform_shape)

        return img, pid, camid, img_path, boxes


def transform_boxes(boxes, original_shape, transform_shape):
    """
    将原始boxes转化为降采样16倍后的ROI坐标

    :param boxes:
    :return: window_box, left_light_box, right_light_box, (N, 4). Stacked together for further extension.
    """

    window_boxes = [box[0] == 'window' for box in boxes]
    light_boxes = [box[0] == 'light' for box in boxes]

    # hard code平均框。该坐标是源代码中16*16特征图上的坐标，转化到目标坐标
    avg_window_box = np.array([0, 3, 15, 7]).reshape((2, 2)) / 16 * transform_shape / 16
    if len(window_boxes) == 0:
        window_box = avg_window_box
    else:
        raw_window_box = window_boxes[0]
        class_, prob, *window_box = raw_window_box
        window_box = np.array(window_box).reshape((2, 2)) / original_shape * transform_shape / 16

    window_box[1, :] = np.ceil(window_box[1, :])
    window_box[0, :] = np.round(window_box[1, :])

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

    elif len(light_boxes) == 2:
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

    return np.stack(window_box, left_light_box, right_light_box)
