# -*- coding: utf-8 -*-
# @Time  : 2019/3/22 16:16
# @Author : yx
# @Desc : ==============================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, n_f):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(n_f, n_f, 3, 1, 1),  # 输入和输出的feature 大小不变
            nn.BatchNorm2d(n_f),
            nn.ReLU(),
            nn.Conv2d(n_f, n_f, 3, 1, 1),
            nn.BatchNorm2d(n_f),
        )

    def forward(self, x):
        x = x + self.residual(x)
        x = F.relu(x)
        return x


class Network(nn.Module):
    def __init__(self, board_size, n_res=3, n_f=128):
        super(Network, self).__init__()
        nf_p = 32  # 原论文为2  extra last layer filters trick
        nf_v = 16  # 原论文为1
        # 网络结构
        common_module_lst = nn.ModuleList([
            nn.Conv2d(4, n_f, 3, 1, 1),
            nn.BatchNorm2d(n_f),
            nn.ReLU()
        ])
        common_module_lst.extend([ResidualBlock(n_f) for _ in range(n_res)])
        self.body = nn.Sequential(*common_module_lst)

        self.head_p = nn.Sequential(
            nn.Conv2d(n_f, nf_p, 1, 1),  # 输入和输出的feature 大小不变
            nn.BatchNorm2d(nf_p),
            nn.ReLU(),
            Flatten(),
            nn.Linear(nf_p * board_size * board_size, board_size * board_size),
            nn.LogSoftmax(dim=-1)
        )

        self.head_v = nn.Sequential(
            nn.Conv2d(n_f, nf_v, 1, 1),  # # 输入和输出的feature 大小不变
            nn.BatchNorm2d(nf_v),
            nn.ReLU(),
            Flatten(),
            nn.Linear(nf_v * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.body(x)
        p = self.head_p(x)
        v = self.head_v(x)
        return p, v


class PolicyValueNet:
    def __init__(self, board_size, n_res=RES_BLOCK_NUM, n_f=FILTER_NUM, init_lr=LR, weight_decay=L2_WEIGHT_DECAY,
                 device_str='cuda:0'):
        self.device = torch.device(device_str)
        self.policy_value_net = Network(board_size, n_res, n_f).to(self.device)
        self.trainer = torch.optim.Adam(self.policy_value_net.parameters(),
                                        lr=init_lr, betas=[0.7, 0.99],
                                        weight_decay=weight_decay)
        self.l2_loss = nn.MSELoss()

    def get_policy_value(self, state):
        x = torch.tensor(state).float().unsqueeze(0).to(self.device)
        self.policy_value_net.eval()
        log_act_probs, z = self.policy_value_net(x)
        self.policy_value_net.train()
        pv = log_act_probs.exp()
        return pv.detach().cpu().numpy(), z.detach().cpu().numpy()

    def train_step(self, states, probs, winners, lr):
        ss = torch.tensor(states).float().to(self.device)
        ps = torch.tensor(probs).float().to(self.device)
        ws = torch.tensor(winners).unsqueeze(-1).float().to(self.device)

        # 设置学习率
        for param_group in self.trainer.param_groups:
            param_group['lr'] = lr

        # loss
        log_act_probs, z = self.policy_value_net(ss)
        loss = self.l2_loss(z, ws) - (ps * log_act_probs).sum(1).mean()

        # update
        self.trainer.zero_grad()
        loss.backward()
        self.trainer.step()

        log_act_probs_new, z_new = self.policy_value_net(ss)
        kl = (log_act_probs.exp() * (log_act_probs - log_act_probs_new)).sum(1).mean()
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs_new) * log_act_probs_new, 1))

        return loss.item(), kl.item(), entropy.item()

    def save_model(self, model_path):
        torch.save(self.policy_value_net.state_dict(), model_path)
        return model_path

    def restore_model(self, model_path):
        self.policy_value_net.load_state_dict(torch.load(model_path))
