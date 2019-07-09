"""
为所有训练集中的图片生成naive heatmap

naive heatmap指不加权直接对最后一层的特征图求和
"""
import sys

sys.path.insert(0, './')

import argparse
import os
from os import mkdir
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn
from torch import nn

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger

import torch
from torch.utils.data import DataLoader

from data.collate_batch import train_collate_fn, val_collate_fn
from data.datasets import init_dataset, ImageDataset
from data.samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid, \
    SimilarIdentitySampler  # New add by gu
from data.transforms import build_transforms


def get_naive_heatmap(model, x, upsample_shape):
    """
    get navie heatmap.即不使用任何权重，直接把heatmap结果相加.

    :param model:
    :param torch.Tensor x: [B, 3, H, W]
    :return:
    """
    featuremaps = model.base(x)  # [B, C, H, W]

    heatmaps = torch.sum(featuremaps, dim=1)  # [B, H, W]

    output_heatmaps = []
    for heatmap in heatmaps:
        heatmap = heatmap - torch.min(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        heatmap = (255 * heatmap).type(torch.uint8)
        heatmap_np = heatmap.detach().cpu().numpy()
        output_heatmaps.append(cv2.resize(heatmap_np, upsample_shape))

    return output_heatmaps


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


def make_naive_heatmap_dataloader(cfg):
    """
    之前的train和train_transformer绑定到一起了。这里需要在train数据集上使用val的transformer。仅用于对图片做后处理.

    :return:
    """
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    print(len(cfg.DATASETS.NAMES))
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, val_transforms)
    val_set = ImageDataset(dataset.gallery, val_transforms)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers, shuffle=False,
        collate_fn=val_collate_fn)
    val_loader = DataLoader(
        val_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers, shuffle=False,
        collate_fn=val_collate_fn)
    return train_loader, val_loader, num_classes


def get_config():
    """
    依次从配置文件，命令行读取默认配置并返回一个字典对象
    :return:
    """
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main():
    # 读取配置
    cfg = get_config()

    # 创建输出文件夹
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    # 注册logger
    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))

    # 设置cuda环境
    device = cfg.MODEL.DEVICE
    if device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        cudnn.benchmark = True

    # 热力图输出文件夹
    heatmap_path = os.path.join(cfg.OUTPUT_DIR, 'naive_heatmaps')
    rendered_path = os.path.join(cfg.OUTPUT_DIR, 'naive_heatmaps_rendered')
    for path in [heatmap_path, rendered_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 读取数据集
    train_loader: DataLoader
    val_loader: DataLoader
    train_loader, val_loader, num_classes = make_naive_heatmap_dataloader(cfg)

    # 读取模型
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(cfg.MODEL.DEVICE)

    logger.info("Generating heatmaps......")
    model.eval()
    with torch.no_grad():
        # 主循环
        # for loader in [train_loader, val_loader]:
        for loader in [train_loader, val_loader]:
            for batch in tqdm(loader):
                # 读取数据
                data, pids, camids, paths = batch
                data = data.to(device) if torch.cuda.device_count() >= 1 else data
                B = data.shape[0]

                # 获得heatmap
                if torch.cuda.device_count() > 1:
                    heatmaps = get_naive_heatmap(model.module, data, (224, 224))
                else:
                    heatmaps = get_naive_heatmap(model, data, (224, 224))

                # 后处理，保存heatmap
                for i in range(B):
                    # save heatmap
                    basename = os.path.splitext(os.path.split(paths[i])[-1])[0]
                    save_path = os.path.join(heatmap_path, '{}.npy'.format(basename))
                    np.save(save_path, heatmaps[i])

                    # 渲染可视化结果并保存
                    heatmap = cv2.applyColorMap(heatmaps[i], cv2.COLORMAP_JET)
                    img = denormalize(data[i], cfg)
                    rendered_heatmap = (heatmap * 0.3 + img * 0.5).astype(np.uint8)
                    plt.cla()
                    plt.axis('off')
                    plt.imshow(rendered_heatmap)
                    save_path = os.path.join(rendered_path, '{}.jpg'.format(basename))
                    plt.savefig(save_path)


if __name__ == '__main__':
    main()
