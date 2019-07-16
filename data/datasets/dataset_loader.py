# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
import os.path as osp
from utils.iotools import read_image, read_nori_image

try:
    import nori2 as nori

    nf = nori.Fetcher()
    import cv2

    use_nori = True
except ImportError as e:
    use_nori = False



class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        if use_nori:
            img = read_nori_image(img_path)
        else:
            img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path
