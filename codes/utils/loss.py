import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PTAL_Loss(nn.Module):
    def __init__(self, mc_loss_weight=0.1):
        super().__init__()
        self.factor_dict = {
            'prop_point_loss': 1,
            'prop_saliency_loss': 20,
            'mc_variance_loss': mc_loss_weight,
        }
        self.loss_dic = {
            'prop_point_loss': self._prop_point_loss,
            'prop_saliency_loss': self._prop_saliency_loss,
            'mc_variance_loss': self._mc_variance_loss,
        }        

    def _prop_point_loss(self, inputs):
        # maxsup_alpha: 0.05 - 0.01
        alpha = self.maxsup_alpha
        prop_cas = inputs[0].permute(0, 2, 1)  # [B, T, C]
        prop_att = inputs[1].permute(0, 2, 1)  # 已经在模型中sigmoid过了
        point_label = inputs[2]
        # target_T = prop_cas.shape[1]
        #
        # if point_label.shape[1] != target_T:
        #     point_label = point_label.permute(0, 2, 1)

        # 1. 计算 Logits
        logits = prop_cas * prop_att

        # 2. 构建 Hard Label (不进行 Label Smoothing)
        point_label = torch.cat((point_label, torch.zeros_like(point_label[:, :, [0]])), dim=-1)
        point_label[:, torch.where(torch.sum(point_label[0, :, :], dim=-1) == 0)[0], -1] = 1

        # 3. 标准交叉熵 (使用 Hard Label)
        # 论文 Eq (6): H(y, q)
        log_probs = F.log_softmax(logits, dim=-1)
        loss_fuse = - (point_label * log_probs).sum(dim=-1).mean()

        # 4. MaxSup 正则化项
        # 论文 Eq (8): alpha * (z_max - z_mean)
        if alpha > 0:
            # z_max: 当前模型预测最强的那个类别的 logit
            z_max = torch.max(logits, dim=-1)[0]
            # z_mean: 所有类别的 logit 均值
            z_mean = torch.mean(logits, dim=-1)
            # 公式 (8): alpha * (z_max - mean(z)):正则项：无论预测对错，都压制最大值，防止过拟合噪声
            loss_maxsup = alpha * (z_max - z_mean).mean()
            return loss_fuse + loss_maxsup

        return loss_fuse


    def _prop_saliency_loss(self, inputs):
        center_score = torch.sigmoid(inputs[0]).squeeze()
        center_label = inputs[1].squeeze()
        loss = torch.mean(torch.square(center_score - center_label))
        return loss
    
    def _mc_variance_loss(self, inputs):
        """
        注意力一致性损失 (Attention Consistency Loss)
        通过惩罚高质量提议的注意力方差，促进注意力预测的稳定性
        
        inputs: [attn_variance, prop_center]
        attn_variance: [B, 1, M] - 注意力权重 x_atn 的 MC 采样方差
        prop_center: [B, 1, M] - 中心得分（提议质量评估）
        """
        attn_var = inputs[0]  # [B, 1, M]
        prop_center = inputs[1]    # [B, 1, M]
        
        # 使用 sigmoid(center) 作为权重：高质量提议应有低方差
        w_center = torch.sigmoid(prop_center).detach() # [B, 1, M] +.detach()
        w_var = attn_var * w_center  # 加权方差：鼓励高置信度 proposal 的注意力保持一致,[B, 1, M]
        loss = torch.mean(w_var)
        return loss

    def forward(self, s_l_dict):
        losses = {}
        for type, score_label in s_l_dict.items():
            if type in self.loss_dic:
                losses[type] = self.loss_dic[type](score_label)
        return losses

    def compute_total_loss(self, loss_dict):
        total_loss = 0
        for type, loss in loss_dict.items():
            total_loss += self.factor_dict[type] * loss
        return total_loss
