import torch
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from data.datasets.veri776 import VeRi776Nori

centerloss_checkout_ckpt = torch.load('../veri776_train_output/resnet50_centerloss_120.pth', map_location='cpu')
center_vecs = centerloss_checkout_ckpt['centers']  # 576, 2048


def get_simmilarity_matrix(center_vecs, method='cos'):
    center_vecs -= torch.mean(center_vecs, dim=0)
    lengths = torch.sum(center_vecs ** 2, dim=1).view(-1, 1) ** 0.5
    normalized_center_vecs = center_vecs / lengths

    if method == 'cos':
        matrixs = torch.mm(normalized_center_vecs, normalized_center_vecs.transpose(0, 1))
    elif method == 'euclidean':
        similarity_matrixs = torch.mm(center_vecs, center_vecs.transpose(0, 1))
        matrix = (lengths.view(-1, 1) ** 2 + lengths.view(1, -1) ** 2 - 2 * similarity_matrixs) ** 0.5

    return matrix


cos_similarity_matrix = get_simmilarity_matrix(center_vecs)

query_index = 10

similarity, rank = torch.sort(cos_similarity_matrix[query_index][1:], descending=True)
# plt.plot(np.arange(len(distance)), sorted(distance))
plt.plot(np.arange(len(rank) - 1), sorted(cos_similarity_matrix[query_index][1:]))

plt.show()

dataset = VeRi776Nori()
pids = [dataset.train_label2pid[label.item()] for label in rank]
print(similarity)

print(dataset.train_label2pid[query_index])
print(pids)
