"""
Towards Fewer Annotations: Active Learning via Region Impurity and
    Prediction Uncertainty for Domain Adaptive Semantic Segmentation (CVPR 2022 Oral)

view code: https://github.com/BIT-DA/RIPU
"""

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


class RIPUMining(torch.nn.Module):

    def __init__(self, ymir_cfg: edict, class_number: int):
        self.ymir_cfg = ymir_cfg
        self.region_radius = int(ymir_cfg.param.ripu_region_radius)
        # note parameter: with_blank_area
        self.class_number = class_number
        self.image_topk = int(ymir_cfg.param.topk_superpixel_score)
        # ratio = float(ymir_cfg.param.ratio)

        kernel_size = 2 * self.region_radius + 1
        self.region_pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=self.region_radius)

        self.depthwise_conv = torch.nn.Conv2d(in_channels=self.class_number,
                                              out_channels=self.class_number,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=self.region_radius,
                                              bias=False,
                                              padding_mode='zeros',
                                              groups=self.class_number)

        weight = torch.ones((self.class_number, 1, kernel_size, kernel_size), dtype=torch.float32)
        weight = torch.nn.Parameter(weight)
        self.depthwise_conv.weight = weight
        self.depthwise_conv.requires_grad_(False)

    def get_region_uncertainty(self, logit: torch.Tensor) -> torch.Tensor:
        C = torch.tensor(logit.shape[1])
        entropy = -logit * torch.log(logit + 1e-6)  # BCHW
        uncertainty = torch.sum(entropy, dim=1, keepdim=True) / torch.log(C)  # B1HW

        region_uncertainty = self.region_pool(uncertainty)  # B1HW

        return region_uncertainty

    def get_region_impurity(self, logit: torch.Tensor) -> torch.Tensor:
        C = torch.tensor(logit.shape[1])
        predict = torch.argmax(logit, dim=1)  # BHW
        one_hot = F.one_hot(predict, num_classes=self.class_number).permute((0, 3, 1, 2))  # BHW --> BHWC --> BCHW
        summary = self.depthwise_conv(one_hot)  # BCHW
        count = torch.sum(summary, dim=1, keepdim=True)  # B1CH
        dist = summary / count  # BCHW
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / torch.log(C)  # B1HW

        return region_impurity

    def get_region_score(self, logit: torch.Tensor) -> torch.Tensor:
        """
        logit: [B,C,H,W] prediction result with softmax/sigmoid
        """
        score = self.get_region_uncertainty(logit) * self.get_region_impurity(logit)  # B1HW

        return score

    def get_image_score(self, logit: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logit.shape
        score = self.get_region_score(logit).view(size=(B, C * H * W))  # B1HW
        topk = torch.topk(score, k=self.image_topk, dim=2, largest=True)  # BK
        image_score = torch.sum(topk.values, dim=1)  # B
        return image_score
