import nori2 as nori
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pprint import pprint
import torch
import numpy as np
import sys

sys.path.append('../')
from data.datasets.veri776 import VeRi776Nori

nf = nori.Fetcher()


def read_nori_image(nid):
    data = nf.get(nid)
    img = cv2.imdecode(np.fromstring(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def get_simmilarity_matrix(center_vecs, method='cos'):
    center_vecs -= torch.mean(center_vecs, dim=0)
    lengths = torch.sum(center_vecs ** 2, dim=1).view(-1, 1) ** 0.5
    normalized_center_vecs = center_vecs / lengths

    if method == 'cos':
        matrix = torch.mm(normalized_center_vecs, normalized_center_vecs.transpose(0, 1))
    elif method == 'euclidean':
        similarity_matrixs = torch.mm(center_vecs, center_vecs.transpose(0, 1))
        matrix = (lengths.view(-1, 1) ** 2 + lengths.view(1, -1) ** 2 - 2 * similarity_matrixs) ** 0.5

    return matrix
