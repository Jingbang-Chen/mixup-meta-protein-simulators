import numpy as np
import torch
import torch.nn as nn
import os

from model_EGNN_PHM_more_retrieve import EGNN
from model_EGNN_encoder import EGNN_encode
from model_EGNN_base import EGNN_base, Prompt
from automix import Automix

# https://github.com/sungyubkim/GBML/blob/master/main.py

class Network(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False, step=1):
        self.PHM_EGNN = EGNN(in_node_nf=1, hidden_nf=hidden_nf, out_node_nf=1, in_edge_nf=1)
        self.base_EGNN = EGNN_base(in_node_nf=1, hidden_nf=hidden_nf, out_node_nf=1, in_edge_nf=1)
        self.encoder = EGNN_encode(in_node_nf=1, hidden_nf=hidden_nf, out_node_nf=1, in_edge_nf=1)
        self.prompt = Prompt(hidden_nf=hidden_nf)
        self.mix_struct_net = Automix(hidden_nf=hidden_nf, device=device)
        self.mix_temp_net = Automix(hidden_nf=hidden_nf, device=device)
        self.hidden_nf = hidden_nf
        self.device = device
        self.to(device)
    def get_grad(self):
        # 将需要maml的部分require_grad = true
        return None
    def forward(self):
        return None
    def get_loss(self):
        # 需要copy下get_loss_meta函数
        return None
    def get_loss_mix(self):
        # 需要copy下get_loss_meta_mix函数
        return None

class GBML:
    '''
    Gradient-Based Meta-Learning
    '''
    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch_size
        return None

    def _init_net(self):
        self.network = Network(in_node_nf=1, hidden_nf=self.args.hidden_nf, out_node_nf=1, in_edge_nf=1)
        self.network.train()
        self.network.get_grad() # 把需要更新的网络requeires_grad设为true,剩下的false
        return None

    def _init_opt(self):
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.args.outer_lr, nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':
            self.outer_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.args.outer_lr)
        else:
            raise ValueError('Not supported outer optimizer.')
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)
        return None

    def unpack_batch(self, batch):
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.cuda()
        train_targets = train_targets.cuda()

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.cuda()
        test_targets = test_targets.cuda()
        return train_inputs, train_targets, test_inputs, test_targets

    def inner_loop(self):
        raise NotImplementedError

    def outer_loop(self):
        raise NotImplementedError

    def lr_sched(self):
        self.lr_scheduler.step()
        return None

    def load(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.encoder.load_state_dict(torch.load(path))

    def save(self,filename):
        path = os.path.join(self.args.result_path, self.args.alg, filename)
        torch.save(self.network.state_dict(), path)