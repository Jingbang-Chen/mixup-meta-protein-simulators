import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import json
from data import MDDataset
from model import model_EGNN_base, model_EGNN_encoder
from tensorboardX import SummaryWriter

def train(conf):
    # create training and validation datasets and data loaders
    # load network model

    train_temp = [280, 290, 300, 310, 320]

    encoder = model_EGNN_encoder.EGNN_encode(in_node_nf=1, hidden_nf=conf.hidden_dim, out_node_nf=1, in_edge_nf=1)
    data_to_restore = torch.load(os.path.join(conf.log_dir, conf.log_net_dir, "ckpts", f"{conf.log_net_epoch}-encoder.pth"))
    encoder.load_state_dict(data_to_restore, strict=False)
    encoder.to(conf.device)

    network = model_EGNN_base.EGNN_base(in_node_nf=1, hidden_nf=conf.hidden_dim, out_node_nf=1, in_edge_nf=1)
    data_to_restore = torch.load(os.path.join(conf.log_dir, conf.log_net_dir, "ckpts", f"{conf.log_net_epoch}-network.pth"))
    network.load_state_dict(data_to_restore, strict=False)
    network.to(conf.device)

    # load dataset
    train_dataset = MDDataset()
    train_dataset.load_data(conf.train_data_dir, name='traj', temp=train_temp, data_max=10)

    # evalnp.linalg.norm
    for ii in range(conf.distribut_time):
        prompts = get_distribution(train_dataset, train_temp, encoder, network, conf)
        train_prompt = {}
        for ww in range(len(train_temp)):
            train_prompt[ww] = prompts[ww].tolist()
        with open(os.path.join(conf.save_dir, 'result_%d_%d.json' % (conf.trial_id, ii)), 'w') as fout:
            json.dump(train_prompt, fout)

def get_distribution(dataset, temp_list, encoder, network, conf):

    temp_dict = {}
    prompt_net = []
    prompt_opt = []
    prompt_lr = []
    step_prompt = []
    for kk in range(len(temp_list)):
        temp_dict[temp_list[kk]] = kk

        prompt_net.append(model_EGNN_base.Prompt(hidden_nf=conf.hidden_dim))
        prompt_net[kk].to(conf.device)
        prompt_opt.append(torch.optim.Adam(filter(lambda p: p.requires_grad, prompt_net[kk].parameters()), lr=conf.lr,
                                           weight_decay=conf.weight_decay))
        prompt_lr.append(torch.optim.lr_scheduler.StepLR(prompt_opt[kk], step_size=conf.lr_decay_every,
                                                         gamma=conf.lr_decay_by))
        step_prompt.append(0)

    tot_data = len(dataset.data_buffer)
    train_count = min(3000, tot_data)
    arr = np.arange(tot_data)
    np.random.shuffle(arr)
    step = 0
    losses = 0
    start_time = time.perf_counter()
    for epoch in range(conf.finetune_epoch):

        for idd in range(train_count):
            batch = dataset.data_buffer[arr[idd]]

            pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
            id = temp_dict[temp]

            total_loss, _ = get_loss(batch, network, encoder, conf, prompt_net[id])
            total_loss.backward()

            step += 1

            step_prompt[id] += 1
            if step_prompt[id] % 16 == 0:
                prompt_opt[id].step()
                prompt_opt[id].zero_grad()
                prompt_lr[id].step()

            losses += total_loss.item()
            if(step % 500 == 0):
                losses /= 500
                print("loss:",losses)
                losses = 0
                end_time = time.perf_counter()
                print("time consume:", end_time - start_time)
                start_time = end_time

    prompts = []
    for ii in range(len(temp_list)):
        prompt = prompt_net[ii](torch.ones(1).to(conf.device)).detach().cpu().numpy()
        prompts.append(prompt)

    return prompts


def get_loss(batch, network_base, encoder, conf, prompt_net, grad=True):
    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    pos = torch.FloatTensor(np.array(pos)).view(-1, 3).to(conf.device)
    pos_diff = torch.FloatTensor(np.array(pos_diff)).view(-1, 3).to(conf.device)
    prompt = prompt_net(torch.ones(1).to(conf.device))
    # print("atom_num", atom_num)
    cnt1 = len(id1u)
    cnt2 = len(id2u)
    # print(atom_num)
    h = torch.from_numpy(np.array(atom_num)).to(conf.device)
    # print(h)
    id1u = torch.LongTensor(id1u).to(conf.device)
    id1v = torch.LongTensor(id1v).to(conf.device)
    id2u = torch.LongTensor(id2u).to(conf.device)
    id2v = torch.LongTensor(id2v).to(conf.device)
    idu = torch.cat([id1u, id1v, id2u, id2v])
    idv = torch.cat([id1v, id1u, id2v, id2u])
    m = torch.cat([torch.ones(cnt1 * 2, 1), torch.zeros(cnt2 * 2, 1)]).to(conf.device)
    edges = [idu, idv]
    # print(pos[0], pos[1])
    h = encoder(h, pos, edges, m, pos_diff)
    acc = network_base(h, pos, edges, m, prompt, pos_diff)

    if grad:
        new_vel = pos_diff + acc
        pred_nxt_pos = pos + new_vel

        nxt_pos = torch.FloatTensor(np.array(nxt_pos)).view(-1, 3).to(conf.device)
        cri = torch.nn.MSELoss(reduction='sum')
        total_loss = cri(pred_nxt_pos, nxt_pos)

        return total_loss, None
    else :
        real_vel = nxt_pos - pos.detach().cpu().numpy()
        real_acc = real_vel - pos_diff.detach().cpu().numpy()
        return h, real_acc

if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='0809log', help='log name')
    parser.add_argument('--trial_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--train_data_dir', type=str, default='./save_traj', help='train_data_dir')
    parser.add_argument('--save_dir', type=str, default='./prompt_vector_save_0929')
    # network settings
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--log_net_dir', type=str, default='0909_pretrain_log', help='exp logs directory')
    parser.add_argument('--log_net_epoch', type=int, default=40)
    # training parameters
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=1000)
    parser.add_argument('--finetune_epoch', type=int, default=5)
    parser.add_argument('--distribut_time', type=int, default=200)
    conf = parser.parse_args()
    conf.ignore_joint_info = True
    ### prepare before training

    if not os.path.exists(conf.save_dir):
        os.mkdir(conf.save_dir)


    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # file log
    # flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    # conf.flog = flog

    # backup command running
    # utils.printout(flog, ' '.join(sys.argv) + '\n')
    # utils.printout(flog, f'Random Seed: {conf.seed}')

    # set training device
    device = torch.device(conf.device)
    # utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    # parse params

    train(conf)

    ### before quit
    # close file log
    # flog.close()

