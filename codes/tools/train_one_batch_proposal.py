import numpy as np
import torch
import torch.nn as nn
import torch.optim
import os
from utils.utils import RunningVar, get_mc_samples, train_metrics


def train_one_proposal_batch(config,
                             model: nn.Module,
                             device: torch.device,
                             dataloader: iter,
                             criterion: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             batch_size: int,
                             current_itr: int = 0,
                             total_itr: int = 2000,
                             mc_config: dict = None):
    model.train()
    total_loss = []
    log_losses = {}

    # 默认MC配置,并根据当前训练进度确定MC采样次数
    if mc_config is None:
        mc_config = {
            'stage1_ratio': 0.2, 'stage2_ratio': 0.6,
            'stage1_k': 1, 'stage2_k': 2, 'stage3_k': 3
        }
    mc_samples = get_mc_samples(current_itr, total_itr, mc_config)
    
    for b in range(1, batch_size + 1):
        (feature, v_label, vid_name, proposal, proposals_point, proposals_center_label, proposals_multi_flag) = next(dataloader)

        feature = feature.to(device)
        proposal = proposal.to(device)
        proposals_point = proposals_point.to(device)
        proposals_center_label = proposals_center_label.to(device)

        prop_cas, prop_bg, prop_attn, prop_center_score, feat_fuse = model(
            feature, proposal, is_training=True)

        # MC Dropout Sampling (二次采样发生时) - 仅关注注意力分支
        attn_variance = None
        if mc_samples > 1:
    
            attn_tracker = RunningVar()  # 初始化 RunningVar 统计器，仅跟踪 x_atn
          
            for _ in range(mc_samples):
                x_atn = model.forward_attn_only(feat_fuse)
                attn_tracker.update(x_atn)

            # 计算注意力方差用于一致性损失
            # x_atn shape: [B, 1, M]
            attn_variance = attn_tracker.get_variance()  
            
            # prop_cas, prop_bg, prop_attn, prop_center_score全部使用首次前向结果，不做均值化 

        # compute loss
        s_l_dict = {
            'prop_point_loss': [prop_cas, prop_attn, proposals_point],
            'prop_saliency_loss': [prop_center_score, proposals_center_label],
        }        
        if mc_samples > 1 and attn_variance is not None:
            s_l_dict['mc_variance_loss'] = [attn_variance, prop_center_score]  # [B, 1, M] 
        

        # Modifying the parameters of the loss function based on the training progress
        progress = current_itr / total_itr
        
        if progress < 0.35:
            criterion.maxsup_alpha = config.sup_max - (config.sup_max - config.sup_min) * progress * 1.5  # start + (end - start) * progress 
        elif 0.6 > progress >= 0.35:
            criterion.maxsup_alpha = 0.01
        elif progress >= 0.6:
            criterion.maxsup_alpha = 0.005
         
        
        if 'mc_variance_loss' in criterion.factor_dict:
            weight = config.mc_loss_weight
            if progress >= 0.5:
                weight = 0.01
            elif progress > 0.35:
                # start = 0.09  
                # end = 0.01
                weight = 0.03 
                # start - (start - end) * (progress - 0.2)
                
            criterion.factor_dict['mc_variance_loss'] = weight
        # (可选) 打印调试信息，确认变化 (建议每隔一定 iteration 打印一次)
        # if current_itr % 200 == 0:
        #     print(f"Iter {current_itr}: MaxSup_Alpha={criterion.maxsup_alpha:.4f}, MC_Weight={criterion.factor_dict['mc_variance_loss']:.4f}")
        
        loss_dict = criterion(s_l_dict)

        for k in loss_dict.keys():
            if k not in log_losses:
                log_losses[k] = []
            else:
                log_losses[k].append(loss_dict[k].detach().cpu().item())
        total_loss.append(criterion.compute_total_loss(loss_dict))

    total_loss = sum(total_loss) / batch_size
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    
    # 学习率与损失函数曲线图 
    cur_lr = optimizer.param_groups[0]['lr']
    cur_loss1 = total_loss.item()
    cur_alpha = getattr(criterion, 'maxsup_alpha', 0)
    mc_weight = criterion.factor_dict.get('mc_variance_loss', 0) if hasattr(criterion, 'factor_dict') else 0

    log_path = os.path.join(config.output_folder, 'lr_loss.csv')
    train_metrics(
        csv_path=log_path,
        iteration=current_itr,
        lr=cur_lr,
        total_loss=cur_loss1,
        extra_info={
            'MaxSup_Alpha': cur_alpha,
            'MC_Weight': mc_weight
        }
    )

    return log_losses
