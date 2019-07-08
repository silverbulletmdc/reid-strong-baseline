# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import argparse
from config import cfg
import os
from torch.backends import cudnn

def load_cfg():
    parser = argparse.ArgumentParser(description="ReID Baseline small model")
    parser.add_argument(
        "--config_file", default="", help="path to config flie", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # read file
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    # read command line
    cfg.merge_from_list(args.opts)

    # calc attr
    cfg.USE_GPU = cfg.MODEL.DEVICE == 'cuda'

    cfg.freeze()
    return cfg


def setup_device(device, device_ids=None):
    """
    设置cuda环境

    :param device: "cuda" or "cpu"
    :param device_ids: "0,1,2,3"
    :return:
    """
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
        cudnn.benchmark = True

