import datetime
import time
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import random
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.backends as cudnn
from tqdm import tqdm
from dataset import PTAL_Dataset
from config import generate_thumos_args
from model import TSPNet_Model
from utils import PTAL_Loss, update_label, execution_time
from tools import train_one_proposal_batch, test_proposal

# conda activate /home/featurize/work/myenv
# cd /home/featurize/work/CVPR2024-TSPNet
# python codes/main.py --enable_mc_dropout
# D:\Software\anaconda3\envs\tsp\python.exe codes/main.py --enable_mc_dropout


class Exp(object):
    def __init__(self, exp_type='THUMOS14'):
        self.config = self._get_config(exp_type)
        # self._validate_config()  # 验证配置参数
        if self.config.seed != -1:
            self._setup_seed()
        self.device = self._get_device()


    def train(self):
        t1 = time.time()
        train_dataset, train_loader = self._get_data(subset='train')  # 获取数据集
        test_dataset, test_loader = self._get_data(subset='test')

        model = self._get_model().to(self.device)
        criterion = self._get_criterion()
        optimizer = self._get_optimizer(model)

        # MC三阶段配置
        mc_config = {
            'stage1_ratio': self.config.stage1_ratio,
            'stage2_ratio': self.config.stage2_ratio,
            'stage1_k': self.config.stage1_k,
            'stage2_k': self.config.stage2_k,
            'stage3_k': self.config.stage3_k
        }

        train_time = 0

        loader = iter(train_loader)
        for itr in tqdm(range(1, self.config.num_itr + 1), total=self.config.num_itr):
            train1 = time.time()
            # 设定在 xxx 步 (xx%进度) 时衰减
            if itr == int(self.config.num_itr * self.config.decay_process):
                new_lr = self.config.lr * 0.1  # 将 5e-5 降为 5e-6
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\n [System] Learning Rate decayed to {new_lr} at step {itr}")

            if (itr - 1) % (len(train_loader) // self.config.batch_size) == 0:
                loader = iter(train_loader)
            train_one_proposal_batch(self.config, model, self.device, loader, criterion, optimizer, 
                                     self.config.batch_size, current_itr=itr,
                                     total_itr=self.config.num_itr, mc_config=mc_config)
            train2 = time.time()
            step_time = train2 - train1
            train_time += step_time

            # 动态标签更新；生成高置信度伪标签
            if itr % self.config.update_fre == 0:
                update_label(dataset=train_dataset, dataloader=train_loader, model=model, device=self.device,
                             up_threshold=self.config.up_threshold)

            if itr % 100 == 0:
                test_proposal(self.config, model, self.device, test_loader, itr)

        t2 = time.time()
        total_time = str(datetime.timedelta(seconds=int(t2 - t1)))
        train_time = str(datetime.timedelta(seconds=int(train_time)))
        print(f"Training Time:  {train_time } (训练耗时)")
        print(f"Total Time:  {total_time} (总耗时)")
        execution_time(self.config, total_time, total_time)

    def test(self):
        test_dataset, test_loader = self._get_data(subset='test')
        model = self._get_model().to(self.device)
        model.load_state_dict(torch.load(self.config.model_path))
        test_proposal(self.config, model, self.device, test_loader, 100)

    def _get_config(self, exp_type):
        config_dict = {
            'THUMOS14': generate_thumos_args
        }
        config = config_dict[exp_type]()

        exp_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        # log_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
        config.output_folder = os.path.join(config.log_path, exp_time)
        if not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)

        config.csv_path = os.path.join(config.output_folder, f'4060_{config.decay_process}_{config.lr}.csv')
        json.dump(vars(config), open(os.path.join(config.output_folder, 'config.json'), 'w'), indent=1)
        
        # save codes 在训练或实验开始时备份代码，确保结果可复现
        # os.system(f'cp -r ./codes {config.output_folder}/codes')
        return config

    def _get_device(self):
        is_cuda = torch.cuda.is_available()
        device_id = self.config.device

        if not is_cuda or device_id == 'cpu':
        # if not is_cuda:
            device = torch.device('cpu')
            print('device: CPU')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = device_id
            # os.environ['CUDA_VISIBLE_DEVICES'] = device_id
            device = torch.device(f'cuda:{device_id}')
            print(f'device: CUDA {device_id}')
        return device

    def _get_data(self, subset):
        dataset = PTAL_Dataset(
            data_path=self.config.data_path,
            subset=subset,
            modality=self.config.modality,
            num_classes=self.config.num_classes,
            feature_fps=self.config.feature_fps,
            soft_value=self.config.soft_value
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True if subset == 'train' else False,
            num_workers=self.config.num_worker
        )
        return dataset, data_loader

    def _get_model(self):
        # model_proposal.py
        model = TSPNet_Model(
            args=self.config
        )
        return model

    def _get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(),
                               lr=self.config.lr,
                               weight_decay=self.config.weight_decay)
        return optimizer

    def _get_criterion(self):
        criterion = PTAL_Loss(mc_loss_weight=self.config.mc_loss_weight)
        return criterion

    def _setup_seed(self):
        """
        Set random seeds for reproducibility.
        """
        random.seed(self.config.seed)
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        cudnn.cudnn.benchmark = False
        cudnn.cudnn.deterministic = True
