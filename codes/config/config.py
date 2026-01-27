import numpy as np
import argparse
import shutil
import os


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--des', type=str, default='')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='')
    # log
    parser.add_argument('--log_path', type=str, default='./codes/log/THUMOS14')

    # device
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda-id')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: -1)')
    # dataset
    parser.add_argument('--data_path', type=str, default='.data/THUMOS14')
    parser.add_argument('--subset', type=str, default='train')
    parser.add_argument('--modality', type=str, default='both')
    parser.add_argument('--feature_fps', type=int, default=25)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--soft_value', type=float, default=0.4)

    # model
    parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')
    parser.add_argument('--roi_size', type=int, default=12,
                        help='roi size for proposal features extraction (default: 12)')
    parser.add_argument('--max_proposal', type=int, default=3000,
                        help='maximum number of proposal during training (default: 1000)')
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--hp_alpha', type=float, default=0.1)
    parser.add_argument('--sig_T_attn', type=float, default=1.0,
                        help='Temperature for attention normalization')
    parser.add_argument('--convex_alpha', type=float, default=0.5,
                        help='Convex combination weight for attention fusion')

    # optimizer
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--decay_process', type=float, default=0.4) # 设置学习率衰减0.1的时间
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # train
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_itr', type=int, default=2000)  # 模型训练的迭代次数
    parser.add_argument('--update_fre', type=int, default=200)
    parser.add_argument('--up_threshold', type=float, default=0.8)

    # test
    parser.add_argument('--refine_threshold', type=float, default=0.4)
    parser.add_argument('--gt_path', type=str, default='./data/THUMOS14/gt.json')
    parser.add_argument('--scale', type=float, default=24)

    # 测试阶段中MC Dropout 和噪声相关参数
    parser.add_argument('--mc_dropout_samples', type=int, default=10, 
                        help='Number of MC Dropout samples for uncertainty estimation')
    parser.add_argument('--enable_mc_dropout', action='store_true', 
                        help='Enable MC Dropout during inference')
    parser.add_argument('--noise_ratio', type=float, default=0.01, 
                        help='Gaussian noise ratio for training (relative to feature std)')  # model_proposal.py/TSP-Net
    parser.add_argument('--uncertainty_weight', type=float, default=1.0, 
                        help='Weight coefficient for uncertainty in boundary adaption')  # exp.py
    
    # MC三阶段训练策略
    parser.add_argument('--stage1_ratio', type=float, default=0.2,
                        help='Progress ratio for stage 1 (no MC dropout)')
    parser.add_argument('--stage2_ratio', type=float, default=0.6,
                        help='Progress ratio for stage 2 (light MC dropout)')
    parser.add_argument('--stage1_k', type=int, default=1,
                        help='MC samples in stage 1 (20%)')
    parser.add_argument('--stage2_k', type=int, default=2,
                        help='MC samples in stage 2 (20%-60%)')
    parser.add_argument('--stage3_k', type=int, default=3,
                        help='MC samples in stage 3 (60%-100%)')
    parser.add_argument('--mc_loss_weight', type=float, default=0.18,
                        help='Weight for MC variance regularization loss')

    # 参数调整
    parser.add_argument('--sup_min', type=float, default=0.01,
                        help='MaxSup start at this value')
    parser.add_argument('--sup_max', type=float, default=0.05,
                        help='MaxSup maximum value after linear enhancement')
    
    # proposal_based
    return parser.parse_args()


if __name__ == '__main__':
    args = generate_args()
    print(vars(args))
