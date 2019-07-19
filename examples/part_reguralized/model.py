from modeling.backbones.resnet import ResNet
from modeling.baseline import Baseline, weights_init_kaiming, weights_init_classifier
from torch import nn
import torch
from torch.testing import assert_allclose


class PartReguralizedModel(Baseline):

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, num_boxes):
        super(PartReguralizedModel, self).__init__(num_classes, last_stride, model_path, neck, neck_feat,
                                                   model_name, pretrain_choice)
        self.num_boxes = num_boxes

        if self.neck == 'no':
            self.box_layers = [nn.Linear(self.in_planes, self.num_classes) for i in range(num_boxes)]

        elif self.neck == 'bnneck':
            self.box_bottlenecks = [nn.BatchNorm1d(self.in_planes) for i in range(num_boxes)]
            for box_bottleneck in self.box_bottlenecks:
                box_bottleneck.bias.requires_grad_(False)  # no shift
                box_bottleneck.apply(weights_init_kaiming)

            self.box_classifiers = [nn.Linear(self.in_planes, self.num_classes, bias=False) for i in range(num_boxes)]

            for box_classifier in self.box_classifiers:
                self.box_classifier.apply(weights_init_classifier)

    def forward(self, x, batch_boxes=None):
        """

        :param torch.Tensor x:
        :param boxes: (b, 3, 4)
        :return:
        """
        B, C, W, H = x.shape
        feature_map = self.base(x)
        global_feat = self.gap(feature_map)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        bn_global_feat = self.bnneck(global_feat)

        if self.training:
            global_cls_score = self.classifier(bn_global_feat)

            # for循环很脏。。。以后解决吧

            # ROI projection
            box_feats = [[None for i in range(B)] for j in range(self.num_boxes)]
            for i, boxes in enumerate(batch_boxes):
                for j, box in enumerate(boxes):
                    box_featuremap = feature_map[i][:, box[1]:box[3], box[0]:box[2]]
                    box_feats[j][i] = box_featuremap
            box_feats = x.new_tensor(box_feats)  # [3, B, C, W, H]

            # box feats
            box_feats = [self.gap(box_feats[i]).view(B, C) for i in range(len(self.num_boxes))]  # 3 [B C]
            bn_box_feats = [self.bnneck(box_feat) for box_feat in box_feats]

            box_cls_scores = [box_classifier(box_feat) for box_classifier, box_feat in
                              zip(self.box_classifiers, bn_box_feats)]

            return global_cls_score, box_cls_scores, bn_global_feat, bn_box_feats  # global feature for triplet loss

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return bn_global_feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def bnneck(self, global_feat):
        if self.neck == 'no':
            feat = global_feat

        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        else:
            raise NotImplementedError

        return feat



