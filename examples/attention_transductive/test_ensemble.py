# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

两个模型直接concat到一起，小模型使用原图
"""

import argparse
import os
import sys
sys.path.append('.')
from os import mkdir

import torch
from torch.backends import cudnn

from modeling import Baseline

from exp.attention_transductive.config import _C as cfg
from data import make_data_loader
from engine.inference import inference
from utils.logger import setup_logger
from torch import nn

def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    big_model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    big_model.load_param(cfg.TEST.BIG_WEIGHT)
    small_model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    small_model.load_param(cfg.TEST.SMALL_WEIGHT)
    model = EnsembleModel(big_model, small_model, cfg.TEST.LAMBDA)
    return model

class EnsembleModel(nn.Module):

    def __init__(self, big_model, small_model, lambda_=0.5):
        super(EnsembleModel, self).__init__()
        self.big_model = big_model
        self.small_model = small_model
        self.lambda_ = lambda_

    def forward(self, x):
        big_feat = self.big_model(x)
        small_feat = self.small_model(x)
        feat = torch.cat((big_feat, self.lambda_ * small_feat), 1)
        return feat



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

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)

    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()
