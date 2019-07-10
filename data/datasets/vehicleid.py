# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

try:
    import nori2 as nori

    use_nori = True
except ImportError:
    use_nori = False

from .bases import BaseImageDataset


class VehicleIDNori(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'VehicleID_V1.0'

    def __init__(self, root="s3://lyd-share/crh_reid_data/reid/PKU_VehicleID/VehicleID_V1.0", verbose=True, **kwargs):
        super(VehicleIDNori, self).__init__()
        self.dataset_dir = "s3://lyd-share/crh_reid_data/reid/PKU_VehicleID/VehicleID_V1.0/train_test_split"

        self.train_dir = osp.join(self.dataset_dir, 'train_data_view_keypoints.pkl')
        self.query_dir = osp.join(self.dataset_dir, 'image_query.nori')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test.nori')

        self.train, self.train_label2pid, self.train_pid2label = self._process_dir(self.train_dir, relabel=True)
        self.query, self.query_label2pid, self.query_pid2label = self._process_dir(self.query_dir, relabel=False)
        self.gallery, self.gallery_label2pid, self.gallery_pid2label = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> VeRi776 loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        nr = nori.open(dir_path, 'r')
        img_paths = []
        nids = []

        for nid, _, meta in nr.scan(scan_data=False):
            img_paths.append(meta['filename'])
            nids.append(nid)
        nr.close()

        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        label2pid = {label: pid for pid, label in pid2label.items()}

        dataset = []
        for nid, img_path in zip(nids, img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((nid, pid, camid))

        return dataset, label2pid, pid2label

