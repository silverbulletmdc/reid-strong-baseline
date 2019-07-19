import sys

sys.path.append('.')
import os
import argparse
import torch
import numpy as np
from data.datasets.dataset_loader import read_nori_image, read_image
from torch.utils.data import DataLoader
from data.collate_batch import train_collate_fn, val_collate_fn
from data.datasets import init_dataset, ImageDataset
from data.samplers import RandomIdentitySampler, SimilarIdentitySampler
from data.transforms import build_transforms
from engine.trainer import do_train_with_center, do_train
from layers import make_loss, make_loss_with_center
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR, make_optimizer_with_center
from utils import setup_device
from utils.logger import setup_logger

from examples.attention_transductive.config import cfg

try:
    import nori2 as nori

    use_nori = True
except ImportError:
    use_nori = False


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


class AttentionTransductiveDataset(ImageDataset):
    def __init__(self, dataset, transform=None, heatmaps_path="/data/veri776_train_output/naive_heatmaps", erase_prob=1):
        super(AttentionTransductiveDataset, self).__init__(dataset, transform)
        self.heatmaps_path = heatmaps_path
        self.erase_prob = erase_prob

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        if use_nori:
            img = read_nori_image(img_path)
        else:
            img = read_image(img_path)

        if self.transform is not None:
            img: torch.Tensor = self.transform(img)

        if np.random.rand() < self.erase_prob:
            sigmoid_reverse_heatmap = self.get_sigmoid_reverse_heatmap(img_path)
            img = img * img.new_tensor(sigmoid_reverse_heatmap)

        return img, pid, camid, img_path

    def get_sigmoid_reverse_heatmap(self, img_path, temperature=100):
        basename = os.path.splitext(os.path.split(img_path)[-1])[0]
        heatmap_path = os.path.join(self.heatmaps_path, '{}.npy'.format(basename))
        heatmap = np.load(heatmap_path)
        reverse_heatmap = (255 - heatmap) / 255
        sigmoid_reverse_heatmap = 1 / (1 + np.exp(-(reverse_heatmap - 0.5) * temperature))
        return sigmoid_reverse_heatmap


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=False)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = AttentionTransductiveDataset(dataset.train, train_transforms, heatmaps_path=cfg.DATASETS.NAIVE_HEATMAP_DIR)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        sampler = None

    elif cfg.DATALOADER.SAMPLER == 'hard':
        sim_mat = torch.load('exp/sim_mat.pth').numpy()
        sampler = SimilarIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, sim_mat)
    else:
        sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)

    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    val_set = AttentionTransductiveDataset(dataset.query + dataset.gallery, val_transforms, heatmaps_path=cfg.DATASETS.NAIVE_HEATMAP_DIR)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)  # modified by gu

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer, map_location=cfg.MODEL.DEVICE))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        arguments = {}

        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,  # modify for using self trained model
            loss_func,
            num_query,
            start_epoch  # add for using self trained model
        )
    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        arguments = {}

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            path_to_center_loss = cfg.MODEL.PRETRAIN_PATH.replace('model', 'centerloss')
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            center_criterion.load_state_dict(torch.load(path_to_center_loss))

            optimizer.load_state_dict(torch.load(path_to_optimizer))
            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))

            # pytorch的bug，需要把optimizer中的参数手工移动到gpu中
            if cfg.USE_GPU:
                for opt in [optimizer, optimizer_center]:
                    for state in opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train_with_center(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,  # modify for using self trained model
            loss_func,
            num_query,
            start_epoch  # add for using self trained model
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(
            cfg.MODEL.IF_WITH_CENTER))


if __name__ == "__main__":
    # 读取配置文件
    cfg = load_cfg()
    setup_device(cfg.MODEL.DEVICE, cfg.MODEL.DEVICE_ID)

    # 创建输出文件夹
    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 初始化logger
    logger = setup_logger("reid_baseline", output_dir, 0)
    train(cfg)
