import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing


class ASCOODTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        try:
            self.net.set_params(config.trainer.trainer_args.sigma)
        except AttributeError:
            self.net.module.set_params(config.trainer.trainer_args.sigma)
        self.train_loader = train_loader
        self.config = config
        backbone_params = list(net.parameters())[: -2]
        fc_params = list(net.parameters())[-2:]
        fc_lr = config.optimizer.fc_lr_factor * config.optimizer.lr
        params_list = [{'params': backbone_params},
                       {'params': fc_params, 'lr': fc_lr}]

        self.optimizer = torch.optim.SGD(
            params_list,
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        self.w = config.trainer.trainer_args.w
        self.num_classes = config.dataset.num_classes
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.p_inv = config.trainer.trainer_args.p_inv
        self.ood_type = config.trainer.trainer_args.ood_type
        self.alpha_min = config.trainer.trainer_args.alpha_min
        self.alpha_max = config.trainer.trainer_args.alpha_max

    def get_saliency_map(self, data, labels):
        # ****************************************************************************
        # *** You may obtain saliency map in training mode (instead of eval mode)! ***
        # ****************************************************************************
        self.net.eval()
        data.requires_grad = True
        logit = self.net(data)
        #prob = torch.softmax(logit, dim=1)
        #score = prob[torch.arange(len(data)), labels]
        score = logit[torch.arange(len(data)), labels]
        self.net.zero_grad()
        score.backward(torch.ones_like(labels))
        grad = data.grad.data
        self.net.train()
        return grad

    def shuffle_ood(self, data, mapp):
        batch_size, channels, height, width = data.shape
        n_pixels = int(height * width * self.p_inv)
        mapp = mapp.exp().sum(dim=1)
        mapp = mapp.view(batch_size, -1)
        # mapp = torch.rand((batch_size, height, width)).view(batch_size, -1).cuda()
        _, top_indices = torch.topk(mapp, n_pixels, dim=1)
        shuffle_indices = torch.argsort(torch.rand_like(top_indices.float()), dim=1)
        data_flat = data.view(batch_size, channels, -1)
        selected_pixels = torch.gather(data_flat, 2, top_indices.unsqueeze(1).expand(-1, channels, -1))
        shuffled_pixels = torch.gather(selected_pixels, 2, shuffle_indices.unsqueeze(1).expand(-1, channels, -1))
        data_ood = data_flat.clone()
        data_ood.scatter_(2, top_indices.unsqueeze(1).expand(-1, channels, -1), shuffled_pixels)
        data_ood = data_ood.view_as(data)
        return data_ood.detach()

    def gradient_ood(self, data, mapp):
        batch_size, channels, height, width = data.shape
        n_pixels = int(channels * height * width * self.p_inv)
        abs_mapp = abs(mapp).view(batch_size, channels * height * width)
        _, topk_indices = torch.topk(abs_mapp, n_pixels, dim=1)
        mask = torch.zeros_like(abs_mapp, dtype=torch.uint8)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.view(batch_size, channels, height, width)
        data_ood = data + mapp * self.alpha * mask
        return data_ood

    def get_ood_sample(self, data, target):
        if self.ood_type == 'gaussian':
            return self.alpha * torch.randn_like(data)
        else:
            mapp = self.get_saliency_map(data, target)
            if self.ood_type == 'shuffle':
                return self.shuffle_ood(data, mapp)
            elif self.ood_type == 'gradient':
                return self.gradient_ood(data, mapp)
            else:
                raise NotImplementedError

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        progress_percent = epoch_idx / self.config.optimizer.num_epochs
        self.alpha = (self.alpha_max - self.alpha_min) * (1 - progress_percent) + self.alpha_min

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()
            data_ood = self.get_ood_sample(data.clone(), target)
            data = torch.cat([data, data_ood], dim=0)
            batch_size = len(target)
            logit = self.net(data)
            id_logit, ood_logit = logit[:batch_size], logit[batch_size:]
            id_loss = F.cross_entropy(id_logit, target)
            target_ood = torch.Tensor(len(ood_logit), self.num_classes).fill_(1/self.num_classes).cuda()
            ood_loss = self.kl_loss(F.log_softmax(ood_logit, dim=1), target_ood)
            loss = id_loss + self.w * ood_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
