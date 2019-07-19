# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

from tqdm import tqdm
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import numpy as np

from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger
from data.datasets.prcv import PRCVDataset
from data.transforms import build_transforms
from data.datasets import ImageDataset
from data.collate_batch import val_collate_fn


def make_test_dataloader(cfg):
    dataset = PRCVDataset("/home/dechao/disk1/datasets/prcv/crop_testA")
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    query_set = ImageDataset(dataset.query, val_transforms)
    query_loader = DataLoader(query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                            collate_fn=val_collate_fn)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = DataLoader(gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                              collate_fn=val_collate_fn)

    return query_loader, gallery_loader


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    query_dataloader, gallery_dataloader =  make_test_dataloader(cfg)
    model = build_model(cfg, 1)
    model.load_param(cfg.TEST.WEIGHT)

    device = cfg.MODEL.DEVICE

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model.to(cfg.MODEL.DEVICE)

    inference(cfg, model, query_dataloader, gallery_dataloader)


def inference(cfg, model, query_dataloader, gallery_dataloader):

    # 加载数据
    loader_dict = {
        'query': query_dataloader,
        'gallery': gallery_dataloader
    }
    feats_dict = {}
    names_dict = {}

    # 提取特征
    for phase in loader_dict:
        feats = []
        all_names = []
        for batch in tqdm(loader_dict[phase]):
            model.eval()
            with torch.no_grad():
                data, pids, camids, paths = batch
                data = data.to(cfg.MODEL.DEVICE) if torch.cuda.device_count() >= 1 else data
                feat = model(data)
                feats.append(feat)
                names = [os.path.splitext(os.path.split(path)[-1])[0] for path in paths]
                all_names.extend(names)
                # if len(all_names) > 30:
                #     break # small for test

        feats_dict[phase] = torch.cat(feats, 0)
        names_dict[phase] = np.array(all_names)

    # 求距离矩阵
    qf = feats_dict['query']
    gf = feats_dict['gallery']
    if cfg.TEST.FEAT_NORM == 'yes':
        print("The test feature is normalized")
        qf = torch.nn.functional.normalize(qf, dim=1, p=2)
        gf = torch.nn.functional.normalize(gf, dim=1, p=2)


    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())

    # 计算top20
    dist, topk = distmat.topk(20, 1, False)
    topk.detach().cpu().numpy()
    result_dict = {}

    # 保存中间文件以后使用
    torch.save({
        "dist_mat": distmat,
        "qf": qf,
        "gf": gf,
        "query_name": names_dict["query"],
        "gallery_name": names_dict["gallery"]
    }, 'results.pth')

    # 生成结果文件
    for q_name, g_ind in zip(names_dict['query'], topk):
        result_dict[q_name] = [names_dict['gallery'][idx] for idx in g_ind]

    with open('/home/dechao/disk1/datasets/prcv/testA/testA/query.txt', 'r') as f:
        query_list = f.read().strip().split('\n')
    with open('result.txt', 'w') as f:
        for q_name in query_list:
            f.write('\t'.join(result_dict[q_name]))
            f.write('\n')







if __name__ == '__main__':
    main()
