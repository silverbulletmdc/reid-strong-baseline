import json
import os

import nori2 as nori
from data.datasets.veri776 import VeRi776Nori
import pprint
import time
import numpy as np


def veri776_adaptor(dataset, window_json_path, light_json_path, s3_root="s3://normal/veri776/image_train.nori"):
    """
    将原工程使用的dataset(list[tuple(img_path, pid, cam_id)])转化为dict格式，并增加nori信息。使用nori_id和img_name对齐

    :param dataset:
    :param json_path:
    :return:
    """

    # 构建nid和filename的查询表
    nr = nori.open(s3_root)
    filename_nid_dict = {}
    nid_filename_dict = {}

    for nid, _, meta in nr.scan(scan_data=False):
        filename = os.path.split(meta['filename'])[-1]  # from image/***.jpg to ***.jpg
        filename_nid_dict[filename] = nid
        nid_filename_dict[nid] = filename

    # 使用nid作为键值，将信息统一到一个dict中
    nid_meta_dict = {}
    for item in dataset:
        nid, pid, camid = item
        nid_meta_dict[nid] = {
            'id': pid,
            'cam_id': camid,
            'nori_id': nid,
            'boxes': [],
        }

    # 处理box
    with open(window_json_path, 'r') as f:
        window_box_dict = json.load(f)

    with open(light_json_path, 'r') as f:
        light_box_dict = json.load(f)

    for box_dict in (window_box_dict, light_box_dict):
        for box_meta in box_dict:
            filename = box_meta['img_name']
            if filename in filename_nid_dict:
                nori_id = filename_nid_dict[filename]

                # 把centerbox转化为coernerbox
                center_box = np.array(box_meta['box'])
                corner_box = center_to_corner(center_box)

                nid_meta_dict[nori_id]['boxes'].append(
                    (box_meta['class'], box_meta['prob'], *corner_box)
                )
                nid_meta_dict[nori_id]['filename'] = filename

    return list(nid_meta_dict.values())


def center_to_corner(center_box):
    corner_box = np.zeros(4)
    corner_box[0] = center_box[0] - center_box[2] / 2
    corner_box[1] = center_box[1] - center_box[3] / 2
    corner_box[2] = center_box[0] + center_box[2] / 2
    corner_box[3] = center_box[1] + center_box[3] / 2
    return corner_box


if __name__ == '__main__':
    window_json_path = '/home/mengdechao/projects/darknet/window_result.json'
    light_json_path = '/home/mengdechao/projects/darknet/light_result.json'
    dataset = VeRi776Nori()

    train_dataset = dataset.train
    query_dataset = dataset.query
    gallery_dataset = dataset.gallery

    train_s3_root = "s3://normal/veri776/image_train.nori"
    query_s3_root = "s3://normal/veri776/image_query.nori"
    gallery_s3_root = "s3://normal/veri776/image_test.nori"

    t1 = time.time()
    train_nid_meta_list = veri776_adaptor(train_dataset, window_json_path, light_json_path, train_s3_root)
    query_nid_meta_list = veri776_adaptor(query_dataset, window_json_path, light_json_path, query_s3_root)
    gallery_nid_meta_list = veri776_adaptor(gallery_dataset, window_json_path, light_json_path, gallery_s3_root)
    all_dict = {
        "train": train_nid_meta_list,
        "query": query_nid_meta_list,
        "gallery": gallery_nid_meta_list
    }
    cost_time = time.time() - t1
    print("Generated. Cost {} secs.".format(cost_time))  # about 2 sec
    pprint.pprint(train_nid_meta_list[:2])
    with open('veri776_with_box.json', 'w') as f:
        json.dump(all_dict, f)
