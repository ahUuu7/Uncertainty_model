import json
import time
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import shutil
from torch.utils.data import DataLoader
from utils import res2json, log_metrics, soft_nms, filter_segments, boundary_adaption, csv_metrics
from utils.eval_detectionpmil import ANETdetection


def enable_mc_dropout(model):
    """启用 MC Dropout：保持 model.eval() 但手动开启 Dropout 层"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_entropy(probs, eps=1e-6):
    """计算分类分布的熵"""
    probs = probs + eps  # 避免 log(0)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    return entropy


@torch.no_grad()
def test_proposal(config,
                  model: nn.Module,
                  device: torch.device,
                  dataloader: DataLoader,
                  itr: int):
    model.eval()
    dataset = config.data_path.split('/')[-1]

    final_res = {}
    final_res['results'] = {}
    inf_time = 0
    predictions = []

    for idx, [feature, _, vd_name, proposals, _, _, _] in enumerate(
            dataloader):

        feature = feature.to(device)
        proposals = proposals.to(device)
        vid_name = vd_name[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time() #epoch开始

        if config.enable_mc_dropout:
            # ========== 训练/推理一致性方案 ==========
            # 第一步：单次前向计算特征和固定分支（CAS、中心度）
            model.eval()
            prop_cas, _, _, prop_center, feat_fuse = model(feature, proposals, is_training=False)
            # `prop_cas`: 动作分类分数 [B, C+1, M]（包含背景类）
            # 第二步：仅对注意力分支进行MC采样（与训练策略一致）
            attn_samples = []
            # x_atn_ref = None  # 用于记录第一次 MC 结果
            for _ in range(config.mc_dropout_samples):
                enable_mc_dropout(model)
                x_atn = model.forward_attn_only(feat_fuse)
                attn_samples.append(x_atn)

            # 第三步：统计注意力不确定性
            attn_stack = torch.stack(attn_samples)  # [T, B, 1, M]
            attn_mean = attn_stack.mean(0)  # [B, 1, M]
            attn_var = attn_stack.var(0)    # [B, 1, M] - 注意力方差
            attn_var = attn_var.clamp(min=1e-6, max=1.0)

            # 第四步：计算前景/背景置信度
            bg_attn = 1 - attn_mean   # [B, 1, M]
            prop_bg = prop_cas * bg_attn  # [B, 1, M]*[1, C, M]=[B, C, M]
            # prop_att = prop_cas * attn_mean  # [B, C, M]
            prop_att = attn_mean  # [B, 1, M]修改

            # 第五步：训练/推理对齐的不确定性因子
            # 训练时：L = Mean(Var × Center) 
            # 推理时：u_factor = exp(-λ × Var × Center)

            # Center加权的不确定性
            w_center = torch.sigmoid(prop_center)  # [B, 1, M] - 与训练一致的sigmoid
            weighted_var = attn_var * w_center     # [B, 1, M] - 加权方差（过滤背景噪声）

            # 映射到不确定性因子
            # 高center + 高var → weighted_var大 → u_factor小 → 强抑制（困难样本）
            # 低center + 高var → weighted_var小 → u_factor大 → 弱抑制（背景噪声）
            u_factor = torch.exp(-config.uncertainty_weight * weighted_var).detach()  # [B, 1, M] 提议级不确定性
            u_factor = u_factor.mean(dim=(0, 1))  # [M]
            
            # 计算熵（基于背景CAS分布）
            prop_bg_probs = F.softmax(prop_bg.permute(0, 2, 1), dim=-1).squeeze()  # [M, C]
            prop_bg_entropy = compute_entropy(prop_bg_probs).detach()  # [M]
            
            # 第六步作为最终输出
            prop_attn = prop_att.permute(0, 2, 1)[0]  # [M, 1] 方便按 proposal/时间步处理
            prop_cas = F.softmax(prop_cas.permute(0, 2, 1), dim=-1).squeeze()  # [M, C+1]，softmax转化为”概率分布“
            
            # 保存原始方差和加权方差用于分析
            prop_bg_var = attn_var.expand_as(prop_bg)  # [B, C, M] - 原始方差
            prop_weighted_var = weighted_var.expand_as(prop_bg)  # [B, C, M] - 加权方差
        else:
            # 标准推理（不使用 MC Dropout）
            print("MC dropout samples:", config.mc_dropout_samples)  # 循环是否被执行
            prop_cas, prop_bg, prop_att, prop_center, _ = model(feature, proposals, is_training=False)
            prop_cas = F.softmax(prop_cas.permute(0, 2, 1), dim=-1).squeeze()  # [M, C+1]
            # prop_att=[B,1,M]
            u_factor = torch.ones(prop_att.size(2), device=prop_att.device)  # 标准推理时，不确定性因子为1
            prop_bg_var = torch.zeros_like(prop_att)  # zeros_like自动继承传入tensor的dtype和device——[B,1,M]
            prop_bg_entropy = torch.zeros(prop_att.shape[-1], device=prop_att.device)  # [M]

            prop_attn = prop_att.permute(0, 2, 1)[0]  # [B,1,M]——[B,M,1]——[M,1]

        # 计算推理时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        batch_time = end_time - start_time
        inf_time += batch_time

        prop_center = torch.sigmoid(prop_center.permute(0, 2, 1))[0]  # sigmoid 规范化到 [0,1]——[M,1]

        pred_vid_score = (prop_cas * prop_attn).sum(0) / (prop_attn.sum(0) + 1e-6)
        pred_vid_score = pred_vid_score[:-1]   # 去除背景类别，时间维度加权求平均获得每个类别的平均置信度
        pred_vid_score = pred_vid_score.cpu().numpy()
        
        # 在与对齐置信度相同的位置，转换不确定性为 numpy
        prop_score = (prop_cas * prop_attn * prop_center).cpu().numpy()  # 获得对齐置信度：每个提议在每一类别下的最终分数
        u_factor_np = u_factor.cpu().numpy()  # [M]

        bg_var_np = prop_bg_var.permute(0, 2, 1).squeeze().cpu().numpy()  # [M, C] 原始方差用于输出
        bg_entropy_np = prop_bg_entropy.cpu().numpy()  # [M]

        pred = np.where(pred_vid_score >= 0.1)[0]

        # 选择动作类别的top-k策略
        if len(pred) > 2:
            candidate_scores = pred_vid_score[pred]
            top3_indices = np.argsort(candidate_scores)[::-1][:3]
            pred = pred[top3_indices]

        # Top-1 策略：如果没有一个类别得分高于0.1的，则取得分最高的作为预测结果
        if len(pred) == 0:
            pred = np.array([np.argmax(pred_vid_score)])

        proposal_dict = {}
        proposals = proposals[0].cpu().numpy()
        # 最终生成提议总数为 proposals.shape[0]*len(pred)，另一方面边界回归不执行筛选，因此不会减少提议数量
        for c in pred:
            c_temp = []
            c_proposals = boundary_adaption(proposals, prop_score[:, c],
                                            config.refine_threshold)  # 加权边界更新u_factor_np,

            # 记录但不反馈
            for i in range(proposals.shape[0]):
                c_score = prop_score[i, c]
                b_score = bg_var_np[i, c]  # 不确定性分数（方差）
                u_score = u_factor_np[i]
                b_entropy = bg_entropy_np[i]  # 熵
                c_temp.append([c_proposals[i, 0], c_proposals[i, 1], c, c_score, u_score, b_score, b_entropy])
            proposal_dict[c] = c_temp  # 结果记录，保存到.JSON中，用于可视化高/低不确定性区域

        final_proposals = []
        for class_id in proposal_dict.keys():  #汇总proposals，NMS去重
            temp_proposal = soft_nms(proposal_dict[class_id])  
            final_proposals += temp_proposal

        final_proposals = np.array(final_proposals)
        if dataset == 'THUMOS14':
            final_proposals = filter_segments(final_proposals, vid_name)


        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for i in range(np.shape(final_proposals)[0]):
            video_lst.append(vid_name)
            t_start_lst.append(final_proposals[i, 0])
            t_end_lst.append(final_proposals[i, 1])
            label_lst.append(final_proposals[i, 2])
            score_lst.append(final_proposals[i, 3])
        prediction = pd.DataFrame({"video-id": video_lst,
                                   "t-start": t_start_lst,
                                   "t-end": t_end_lst,
                                   "label": label_lst,
                                   "score": score_lst, })
        predictions.append(prediction)

        final_res['results'][vid_name] = res2json(final_proposals, dataset)

    iou = np.linspace(0.1, 0.7, 7)
    # 计算mAP
    dmap_detect = ANETdetection(f'./data/{dataset}/Annotations', tiou_thresholds=iou,
                                subset="test", verbose=True)
    dmap_detect.prediction = pd.concat(predictions).reset_index(drop=True)
    mAP, dmap_class = dmap_detect.evaluate()

    # 调用记录函数保存CSV
    csv_metrics(dmap_detect, config.csv_path, itr)

    best_flag = log_metrics(map_iou=mAP, time=inf_time, step=itr, pred=len(dmap_detect.prediction), config=config)

    # 暂时不保存last_proposal
    # model_file = os.path.join(config.output_folder, "last_model.pkl")
    # torch.save(model.state_dict(), model_file)

    # json_path = os.path.join(config.output_folder, 'last_proposals.json')
    # with open(json_path, 'w') as f:
    #     json.dump(final_res, f)
    #     f.close()

    best_proposal_file = os.path.join(config.output_folder, 'best_proposals.json')
    if not os.path.exists(best_proposal_file) or best_flag:
        with open(best_proposal_file, 'w') as f:
            json.dump(final_res, f)
            f.close()

    best_model_file = os.path.join(config.output_folder, 'best_model.pkl')
    if not os.path.exists(best_model_file) or best_flag:
        torch.save(model.state_dict(), best_model_file)

        # shutil.copy(json_path, best_proposal_file)
        # shutil.copy(model_file, best_model_file)

    # best_proposal_file = os.path.join(config.output_folder, 'best_proposals.json')
    # if not os.path.exists(best_proposal_file) or best_flag:
    #     os.system(f'cp {json_path} {best_proposal_file}')
    #
    # best_model_file = os.path.join(config.output_folder, 'best_model.pkl')
    # if not os.path.exists(best_model_file) or best_flag:
    #     os.system(f'cp {model_file} {best_model_file}')
