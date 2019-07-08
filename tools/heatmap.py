# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn
from torch import nn
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger


def get_heatmap(model, query, gallerys, upsample_shape):
    """
    get heatmap

    :param model:
    :param torch.Tensor query: [3, H, W]
    :param torch.Tensor gallerys: [B, 3, H, W]
    :return:
    """
    all_img = torch.cat([query.unsqueeze(0), gallerys])
    featuremaps = model.base(all_img)  # [B+1, C, H, W]
    B, C, H, W = featuremaps.shape
    B = B - 1

    global_featuremaps = model.gap(featuremaps)  # [B, C, 1, 1]
    query_featuermap = global_featuremaps[0]  # [C, 1, 1]
    gallery_featuremaps = global_featuremaps[1:]  # [B, C, 1, 1]
    distances = (gallery_featuremaps - query_featuermap) ** 2  # [B, C, 1, 1]
    query_heatmaps = torch.sum(featuremaps[0] * distances, dim=1)  # [B, H, W]
    gallery_heatmaps = torch.sum(featuremaps[1:] * distances, dim=1)  # [B, H, W]

    output_heatmaps = []
    for heatmaps in (query_heatmaps, gallery_heatmaps):
        for heatmap in heatmaps:
            heatmap = heatmap - torch.min(heatmap)
            heatmap = heatmap / torch.max(heatmap)
            heatmap = (255 * heatmap).type(torch.uint8)
            heatmap_np = heatmap.detach().cpu().numpy()
            output_heatmaps.append(cv2.resize(heatmap_np, upsample_shape))

    return output_heatmaps, torch.sum(distances, dim=1)


def denormalize(img, cfg):
    """
    从tensor恢复图片

    :param torch.Tensor img: [C, H, W]
    :param cfg:
    :return: np uint8 rgb图像
    """
    std = img.new_tensor(cfg.INPUT.PIXEL_STD)
    mean = img.new_tensor(cfg.INPUT.PIXEL_MEAN)
    img = img * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    img *= 255
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy().astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--id", required=True, type=int
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    query_loader: DataLoader
    gallery_loader: DataLoader
    query_loader, gallery_loader, num_classes = make_data_loader(cfg, get_demo_dataset=True)
    query_data = query_loader.dataset[args.id]

    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    if not os.path.exists('heatmaps/{}'.format(args.id)):
        os.makedirs('heatmaps/{}'.format(args.id))

    device = cfg.MODEL.DEVICE

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(cfg.MODEL.DEVICE)

    model.eval()
    with torch.no_grad():
        data, pid, camid, path = query_data
        query = data.to(device) if torch.cuda.device_count() >= 1 else data

    for batch in gallery_loader:
        data, pids, camids, paths = batch
        B = data.shape[0]
        gallerys = data.to(device) if torch.cuda.device_count() >= 1 else data

        heatmaps, distances = get_heatmap(model.module, query, gallerys, (224, 224))

        query_img = denormalize(query, cfg)
        for i in range(B):
            query_heatmap = cv2.applyColorMap(heatmaps[i], cv2.COLORMAP_JET)
            query_heatmap = (query_heatmap * 0.3 + query_img * 0.5).astype(np.uint8)

            gallery_heatmap = cv2.applyColorMap(heatmaps[B + i], cv2.COLORMAP_JET)

            gallery_img = denormalize(gallerys[i], cfg)
            gallery_heatmap = (gallery_heatmap * 0.3 + gallery_img * 0.5).astype(np.uint8)

            heatmap = np.concatenate((query_heatmap, gallery_heatmap), axis=1)

            plt.imshow(heatmap)
            plt.savefig(
                'heatmaps/{}/{}_{}_{}_{}.png'.format(args.id, distances[i].item(), i, camids[i], pids[i] == pid))


if __name__ == '__main__':
    main()
