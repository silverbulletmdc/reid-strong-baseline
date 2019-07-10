import glob
import re
import pickle

import os.path as osp

try:
    import nori2 as nori

    use_nori = True
except ImportError:
    use_nori = False

from .bases import BaseImageDataset
from utils.oss_tool import get_oss_client, load_from_oss


class VehicleID(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'VehicleID'

    def __init__(self, root="s3://normal/veri776/", verbose=True, **kwargs):
        super(VehicleID, self).__init__()

        bucket = 'lyd-share'
        root_meta_path = "crh_reid_data/reid/PKU_VehicleID/VehicleID_V1.0/train_test_split/"
        train_meta_name = "train_data_view_keypoints.pkl"

        train_meta_path = osp.join(root_meta_path, train_meta_name)
        train_meta_dict = pickle.loads(load_from_oss(get_oss_client('hh-b'), bucket, train_meta_path))

        self.train, self.train_label2pid, self.train_pid2label = self.process_meta(train_meta_dict, relabel=True)

        if verbose:
            print("=> VehicleID loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def load_pickle(self, raw_id_metas_dict: dict, relabel=False):
        raw_ids = [raw_id for raw_id in raw_id_metas_dict.keys()]
        id_container = set()
        id2label = {id: label for label, id in enumerate(id_container)}
        label2id = {label: id for id, label in id2label.items()}

        dataset = []
        for id, raw_id in id2label.items():
            metas = raw_id_metas_dict[raw_id]
            for meta in metas:
                dataset.append(meta['nori_id'], id, -1)
        for nid, img_path in zip(nids, img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((nid, pid, camid))

        return dataset, label2id, id2label
