import torch
from pprint import pprint

centerloss_checkout_ckpt = torch.load('../veri776_train_output/resnet50_centerloss_120.pth', map_location='cpu')
center_vecs = centerloss_checkout_ckpt['centers'] # 576, 2048
center_vecs -= torch.mean(center_vecs, dim=0)
lengths =  torch.sum(center_vecs**2, dim=1).view(-1, 1)**0.5
normalized_center_vecs = center_vecs / lengths
cos_similarity_matrixs = torch.mm(normalized_center_vecs, center_vecs.transpose(0, 1))
similarity_matrixs = torch.mm(center_vecs, center_vecs.transpose(0, 1))
distance_matrix = (lengths.view(-1, 1) **2 + lengths.view(1, -1) **2 - 2 * similarity_matrixs) ** 0.5

m, n = center_vecs.shape[0], center_vecs.shape[0]
distmat = torch.pow(center_vecs, 2).sum(dim=1, keepdim=True).expand(m, n) + \
		  torch.pow(center_vecs, 2).sum(dim=1, keepdim=True).expand(n, m).t()
distmat.addmm_(1, -2, center_vecs, center_vecs.t())


m, n = center_vecs.size(0), center_vecs.size(0)
x_norm = torch.pow(center_vecs, 2).sum(1, keepdim=True).sqrt().expand(m, n)
y_norm = torch.pow(center_vecs, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
xy_intersection = torch.mm(center_vecs, center_vecs.t())
dist = xy_intersection/(x_norm * y_norm)
dist = (1. - dist) / 2

print(cos_similarity_matrixs[0])

