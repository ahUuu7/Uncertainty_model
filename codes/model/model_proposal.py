import numpy as np
import torch
import torch.nn as nn
import torchvision
import math


class FCA_module(nn.Module):
    def __init__(self, channel, dct_h, reduction=16):
        super(FCA_module, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h  # 代表频域分析的固定长度 (如 16 或 64)

        # 1. 生成 DCT 滤波器权重
        mapper_x, _ = self.get_dct_filter(dct_h, dct_h, channel)  # 适配 1D: 我们只需要时间维度的 DCT 变换，不需要 2D 的 mapper_y
        # mapper_x shape: [channel, T]
        self.register_buffer('dct_mapper', mapper_x)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(dct_h)  # 先通过自适应池化统一长度

        # 2. 通道注意力网络
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape
        # [B, C, T] -> [B, C, dct_h]
        x_pooled = self.adaptive_pool(x)

        # Multi-Spectral DCT Pooling
        # x: [B, C, dct_h(T)] * dct_mapper: [C, dct_h(T)] -> [B, C, T] -> sum -> [B, C]：提取每个通道特定的频率分量，而不是简单的平均值
        y = torch.sum(x_pooled * self.dct_mapper, dim=2)  

        y = self.fc(y).view(b, c, 1)
        return y

    def get_dct_filter(self, tile_size_x, tile_size_y, c):
        """
        生成 1D DCT 基函数权重
        频率分量索引 (0 是平均值/GAP, 1, 2... 是高频)
        “均匀轮询分配 (Round-Robin)” 策略（而不是top-k）：
        将通道 C 分成多个组，每组关注不同的频率(Channel 0-63 使用 freq=0 (GAP))
        """
        mapper_x = torch.zeros(c, tile_size_x)
        basis = torch.arange(tile_size_x).float()[None, :]  # [1, T]

        for i in range(c):
            # 确定当前通道 i 应该使用哪个频率分量 u
            u = i % tile_size_x
            # DCT-II 公式: cos(pi * u * (2k + 1) / 2N)
            dct_kernel = torch.cos((math.pi / tile_size_x) * u * (basis + 0.5))
            # 归一化系数
            if u == 0:
                dct_kernel = dct_kernel * math.sqrt(1 / tile_size_x)
            else:
                dct_kernel = dct_kernel * math.sqrt(2 / tile_size_x)

            mapper_x[i, :] = dct_kernel

        return mapper_x, None

# class ECA_module(nn.Module):
#     def __init__(self, channels, gamma=2, b=1):
#         super().__init__()
#         # 动态计算卷积核大小 k，通道越多，交互范围越大
#         t = int(abs((math.log(channels, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         # 使用 1D 卷积代替全连接层，不进行降维
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: [B, C, T]
#         y1 = self.avg_pool(x)  # [B, 1024, 1]
#         y2 = self.conv(y1.transpose(1, 2)).transpose(1, 2)  # ——>[B, 1024, 1]
#         y = self.sigmoid(y2)
#
#         return y  # 仅返回权重


class Attn(torch.nn.Module):
    def __init__(self, feat_dim, temperature, roi_size=16):
        super().__init__()
        embed_dim = 1024
        self.tau = temperature
        # 位级注意力
        self.reduce = nn.Conv1d(feat_dim, feat_dim // 2, 1)
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(feat_dim//2, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))

        # 通道注意力卷积层
        self.channel_Attn = FCA_module(feat_dim, dct_h=roi_size)  # 传入roi_size=16
        
        # self.channel_Attn = ECA_module(feat_dim)
        # self.channel_avg = nn.AdaptiveAvgPool1d(1)  # 通道维度平均池化
        # self.channel_conv = nn.Sequential(nn.Conv1d(feat_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))

        # 注意力生成网络
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 1, 1), nn.Dropout(0.5))

    def forward(self, vfeat, ffeat):
        # 通道注意力计算
        channel_attn = self.channel_Attn(vfeat)
        channel_attn_norm = channel_attn/torch.norm(channel_attn, p=2, dim=1, keepdim=True)
        # channelfeat = self.channel_avg(vfeat)
        # channel_attn = self.channel_conv(channelfeat)  # [B, 1024, 1]——[1, 1024, 1]
        # channel_attn_norm = channel_attn / torch.norm(channel_attn, p=2, dim=1, keepdim=True)

        # 位级注意力计算
        f_feat = self.reduce(ffeat)
        bit_wise_attn = self.bit_wise_attn(f_feat)  # [B, 1024, T=27]
        bit_wise_attn_norm = bit_wise_attn/torch.norm(bit_wise_attn, p=2, dim=1, keepdim=True)

        # 注意力交互（通道与位级注意力的相似度）
        temp_attn = torch.einsum('bdn,bdt->bnt', [channel_attn_norm, bit_wise_attn_norm])  # [10, 1, 320]

        filter_feat = torch.sigmoid(bit_wise_attn*temp_attn)*vfeat  # 过滤特征（利用注意力抑制背景）

        x_atn = self.attention(filter_feat)
        x_atn = torch.sigmoid(x_atn/self.tau)  # 温度系数控制注意力锐度
        return x_atn, filter_feat


class Backbone_TSPNet(torch.nn.Module):

    def __init__(self, feat_dim, n_class, dropout_ratio, roi_size, temperature, **args):
        super().__init__()
        embed_dim = feat_dim // 2
        self.roi_size = roi_size

        # 提取提议级特征来学习三种语义分数
        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        # 动作类别得分
        self.prop_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, n_class + 1, 1),
        )
    
        # 中心得分
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        # 注意力权重融合
        self.vAttn = Attn(1024, args['opt'].sig_T_attn)
        self.fAttn = Attn(1024, args['opt'].sig_T_attn)

        self.fusion = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7)
            )
        
        self.fusion2 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7)
            )

    def forward(self, feat, **args):
        """
        Inputs:
            feat: tensor of size [B, M, roi_size, D] 通过分段池化提取局部特征，强调中间区域的核心地位
        Outputs:
            prop_cas:  tensor of size [B, C, M]
            prop_attn: tensor of size [B, 1, M]
            prop_center:  tensor of size [B, 1, M]
            action_flow: tensor of size [B, 1, M]
            action_rgb: tensor of size [B, 1, M]
        """

        # 经过局部特征提取与差分融合的加工特征
        feat1 = feat[:, :, : self.roi_size // 6, :].max(2)[0]
        feat2 = feat[:, :, self.roi_size // 6: self.roi_size // 6 * 5, :].max(2)[0]
        feat3 = feat[:, :, self.roi_size // 6 * 5:, :].max(2)[0]
        feat = torch.cat((feat2 - feat1, feat2, feat2 - feat3), dim=2)  # [B,M,3D]对分割后的沿着roi_size维度拼接
        feat_fuse = self.prop_fusion(feat)  # 压缩回D维[B=1, M, D]
        feat_fuse = feat_fuse.transpose(-1, -2)  # [B=1, D, M]  输入

        # 生成视频特征和融合特征的注意力权重
        v_atn, vfeat = self.vAttn(feat_fuse[:,:1024,:],feat_fuse[:,1024:,:])  # 前一半[B, 1, M]
        f_atn, ffeat = self.fAttn(feat_fuse[:,1024:,:],feat_fuse[:,:1024,:])  # 后一半

        # 融合两种注意力权重 [B, 1, M]
        x_atn = args['opt'].convex_alpha*f_atn+(1-args['opt'].convex_alpha)*v_atn
        # 融合特征并通过卷积层处理
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat_out0 = self.fusion(nfeat)
        nfeat_out = self.fusion2(nfeat_out0) 

        prop_cas = self.prop_classifier(feat_fuse)  # [1, C, M] 动作序列
        bg = 1 - x_atn   # [B, 1, M]
        prop_bg = prop_cas * bg  # [1, C, M]*[B, 1, M]=[B,C,M1*M2]
        prop_attn = x_atn  # [B,1,M] 动作性得分
        prop_center = self.prop_completeness(feat_fuse)  # [1, 1, M]
        
        return prop_cas, prop_bg, prop_attn, prop_center, feat_fuse  

    def _forward_attn_only(self, feat_fuse, opt):
        """
        只计算注意力分支的 x_atn（含dropout），供训练阶段采样使用
        """
        v_atn, _ = self.vAttn(feat_fuse[:, :1024, :], feat_fuse[:, 1024:, :])
        f_atn, _ = self.fAttn(feat_fuse[:, 1024:, :], feat_fuse[:, :1024, :])

        x_atn = opt.convex_alpha * f_atn + (1 - opt.convex_alpha) * v_atn
        return x_atn
    

class TSPNet_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_class = args.num_classes
        dropout_ratio = 0.5
        self.feat_dim = args.feature_size  # 2048
        self.max_proposal = args.max_proposal
        self.roi_size = args.roi_size
        self.hp_alpha = args.hp_alpha
        self.noise_ratio = args.noise_ratio
        self.args = args
        # Model(config.len_feature, config.num_classes, config.num_segments)
        self.prop_backbone = Backbone_TSPNet(self.feat_dim, n_class, dropout_ratio, 
                                             self.roi_size, self.hp_alpha, opt=args)
        self.refined_label = []

    # 周围对比特征提取（SCFE）——获取提议级特征
    def extract_roi_features(self, feature, proposal, is_training):
        """
        Surrounding contrastive feature extraction
        Extract region of interest (RoI) features from raw i3d features based on given proposals
        These codes are from <proposal-based multiple instance learning for weakly-supervised temporal action localization>
        Inputs:
            feature: [T, D] tensors
            proposal: [M, 2] tensors
            is_training: bool

        Outputs:
            prop_features:tensor of size [B, M, roi_size, D]
            prop_mask: tensor of size [B, M]
        """
        proposal = proposal[0]
        num_prop = proposal.shape[0]
        # Limit the max number of proposals during training
        if is_training:
            num_prop = min(num_prop, self.max_proposal)
        prop_features = torch.zeros((1, num_prop, self.roi_size, self.feat_dim)).to(feature.device)  # batchsize=1
        prop_mask = torch.zeros((1, num_prop)).to(feature.device)

        if proposal.shape[0] > num_prop:
            sampled_idx = torch.randperm(proposal.shape[0])[:num_prop]
            proposal = proposal[sampled_idx]

        # Extend the proposal by 25% of its length at both sides
        start, end = proposal[:, 0], proposal[:, 1]
        len_prop = end - start
        start_ext = start - 0.25 * len_prop
        end_ext = end + 0.25 * len_prop

        # Fill in blank at edge of the feature, offset 0.5, for more accurate RoI_Align results
        fill_len = torch.ceil(0.25 * len_prop.max()).long() + 1  # +1 because of offset 0.5
        fill_blank = torch.zeros(fill_len, self.feat_dim).to(feature.device)
        feature = torch.cat([fill_blank, feature[0], fill_blank], dim=0)
        start_ext = start_ext + fill_len - 0.5
        end_ext = end_ext + fill_len - 0.5
        proposal_ext = torch.stack((start_ext, end_ext), dim=1)

        # Extract RoI features using RoI Align operation
        y1, y2 = proposal_ext[:, 0], proposal_ext[:, 1]
        x1, x2 = torch.zeros_like(y1), torch.ones_like(y2)
        boxes = torch.stack((x1, y1, x2, y2), dim=1)  # [M, 4]
        feature = feature.transpose(0, 1).unsqueeze(0).unsqueeze(3)  # [1, D, T, 1]
        feat_roi = torchvision.ops.roi_align(feature, [boxes], [self.roi_size, 1])  # [M, D, roi_size, 1]
        feat_roi = feat_roi.squeeze(3).transpose(1, 2)  # [M, roi_size, D]
        prop_features[0, :proposal.shape[0], :, :] = feat_roi  # [1, M, roi_size, D]
        prop_mask[0, :proposal.shape[0]] = 1  # [1, M]
        
        # 训练时添加高斯噪声，模拟边界模糊
        if is_training:
            feat_std = prop_features.std(dim=[0,1], keepdim=True)
            noise = torch.randn_like(prop_features) * (feat_std * self.noise_ratio)
            # 控制上限：将噪声限制在一倍标准差内，防止噪声过大
            noise = torch.clamp(noise, -feat_std, feat_std)
            prop_features = prop_features + noise
        
        return prop_features, prop_mask

    def forward(self, features, proposals, is_training=True):
        """
        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool
        Outputs:
        """
        prop_features, prop_mask = self.extract_roi_features(features, proposals, is_training)
        prop_cas, prop_bg, prop_attn, prop_center, feat_fuse = self.prop_backbone(
            prop_features, opt=self.args)
        return prop_cas, prop_bg, prop_attn, prop_center, feat_fuse

    
    def forward_attn_only(self, feat_fuse):
        """
        只计算注意力分支，供 trainer 访问
        """
        return self.prop_backbone._forward_attn_only(feat_fuse, self.args)
