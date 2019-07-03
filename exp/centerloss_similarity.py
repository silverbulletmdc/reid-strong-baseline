import torch
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

centerloss_checkout_ckpt = torch.load('../veri776_train_output/resnet50_centerloss_120.pth', map_location='cpu')
pprint(centerloss_checkout_ckpt)
center_vecs = centerloss_checkout_ckpt['centers']  # 576, 2048
lengths = torch.sum(center_vecs ** 2, dim=1).view(-1, 1) ** 0.5
normalized_center_vecs = center_vecs / lengths
cos_similarity_matrixs = torch.mm(normalized_center_vecs, normalized_center_vecs.transpose(0, 1))
similarity_matrixs = torch.mm(center_vecs, center_vecs.transpose(0, 1))
distance_matrix = (lengths.view(-1, 1) ** 2 + lengths.view(1, -1) ** 2 - 2 * similarity_matrixs) ** 0.5

print(distance_matrix)
print(similarity_matrixs.shape)
print(distance_matrix[0])
print(cos_similarity_matrixs[0])

distance = distance_matrix[0].detach().numpy()

plt.plot(np.arange(len(distance)), sorted(distance))
plt.show()
