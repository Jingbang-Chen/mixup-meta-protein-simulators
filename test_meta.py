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
from data import MDDataset
from model import model_EGNN_base, model_EGNN_encoder, automix
# from tensorboardX import SummaryWriter

def train(conf):
    # create training and validation datasets and data loaders
    # load network model

    # create models
    train_temp = [280, 300, 320]
    if not conf.meta:
        train_temp = [280, 300, 320]
    test_temp = [285, 290, 295, 305, 310, 315]
    out_temp = [350]

    encoder = model_EGNN_encoder.EGNN_encode(in_node_nf=1, hidden_nf=conf.hidden_dim, out_node_nf=1, in_edge_nf=1)
    encoder.to(conf.device)
    data_to_restore = torch.load(
        os.path.join(conf.log_dir, conf.log_net_dir, "ckpts", f"{conf.log_net_epoch}-encoder.pth"))
    encoder.load_state_dict(data_to_restore, strict=False)

    network = model_EGNN_base.EGNN_base(in_node_nf=1, hidden_nf=conf.hidden_dim, out_node_nf=1, in_edge_nf=1)
    network.to(conf.device)
    data_to_restore = torch.load(
        os.path.join(conf.log_dir, conf.log_net_dir, "ckpts", f"{conf.log_net_epoch}-network.pth"))
    network.load_state_dict(data_to_restore, strict=False)
    if conf.meta:
        mix_dir = conf.log_net_meta_dir
        mix_epoch = conf.log_net_meta_epoch
    else:
        mix_dir = conf.log_net_dir
        mix_epoch = conf.log_net_epoch
    

    mix_temp_net = []
    mix_struct_net = []
    temp_dict = []
    temp_list = []
    data_pool = {}
    # load dataset
    train_dataset = MDDataset()
    val_dataset = MDDataset()
    middle_dataset = MDDataset()
    out_dataset = MDDataset()

    train_dataset.load_data(conf.train_data_dir, name='traj', temp=train_temp, data_max=10)
    val_dataset.load_data(conf.train_data_dir, name='traj', temp=train_temp, data_max=13, data_min=10)
    middle_dataset.load_data(conf.train_data_dir, name='traj', temp=test_temp, data_max=3)
    out_dataset.load_data(conf.train_data_dir, name='traj', temp=out_temp, data_max=3)

    prompt_net = []
    prompt_list = []
  
    ############################################################################################
    score_eval_list = []
    score_middle_list = []
    score_out_list = []
    for i in range(5):
        score_eval = eval_score(val_dataset, train_temp, encoder, network, conf, prompt_list, data_pool, mix_temp_net, mix_struct_net, temp_dict, temp_list, i)
        score_eval_list.append(score_eval)
        score_middle = eval_score(middle_dataset, test_temp, encoder, network, conf, prompt_list, data_pool, mix_temp_net, mix_struct_net, temp_dict, temp_list, i)
        score_middle_list.append(score_middle)
        score_out = eval_score(out_dataset, out_temp, encoder, network, conf, prompt_list, data_pool, mix_temp_net, mix_struct_net, temp_dict, temp_list, i)
        score_out_list.append(score_out)

    print("score_eval:", score_eval_list)
    print("score_middle:", score_middle_list)
    print("score_out:", score_out_list)

def eval_score(dataset, temp_list, encoder, network, conf, prompt_list, data_pool, mix_temp_net, mix_struct_net, train_temp_dict, train_temp_list, id):

    temp_dict = {}
    step_prompt = []
    for i in range(len(temp_list)):
        temp_dict[temp_list[i]] = i
        step_prompt.append(0)

    eval_score = 0
    tot_data = len(dataset.data_buffer)
    train_count = tot_data - int(tot_data * conf.test_data_rate)
    arr = np.arange(len(dataset.data_buffer))
    np.random.shuffle(arr)
    # mix_count = train_count / 2

    prompt_net = []
    prompt_opt = []
    prompt_lr = []
    for i in range(len(temp_list)):
        prompt_net.append(model_EGNN_base.Prompt(hidden_nf=conf.hidden_dim))
        prompt_opt.append(
            torch.optim.Adam(filter(lambda p: p.requires_grad, prompt_net[i].parameters()), lr=conf.finetune_lr,
                             weight_decay=conf.weight_decay))
        prompt_lr.append(torch.optim.lr_scheduler.StepLR(prompt_opt[i], step_size=conf.f_lr_decay_every,
                                                         gamma=conf.f_lr_decay_by))
        prompt_net[i].to(conf.device)
        if conf.meta:
            data_to_restore = torch.load(
                os.path.join(conf.log_dir, conf.log_net_meta_dir, "ckpts", f"{conf.log_net_meta_epoch}-prompt_init.pth"))
            prompt_net[i].load_state_dict(data_to_restore, strict=False)

    ################################## inner loop (finetuning) ##############################

    for ww in range(conf.finetune_epoch):
        losses = 0
        start_time = time.perf_counter()
        for idd in range(train_count):
            batch = dataset.data_buffer[arr[idd]]

            # forward pass (including logging)
            pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
            id = temp_dict[temp]
            prompt = prompt_net[id](torch.ones(1).to(conf.device))
            total_loss, _ = get_loss(batch, network, encoder, conf, prompt)
            total_loss.backward()

            step_prompt[id] += 1
            if step_prompt[id] % 16 == 0:
                prompt_opt[id].step()
                prompt_opt[id].zero_grad()
                prompt_lr[id].step()

            losses += total_loss.item()

        losses = losses / train_count
        end_time = time.perf_counter()
        print(f"finetune_epoch {ww}, time {end_time - start_time}, loss {losses}")
    ########################################################################################

    start_time = time.perf_counter()
    step = 0
    for idd in range(train_count, tot_data):
        batch = dataset.data_buffer[arr[idd]]
        pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
        id = temp_dict[temp]
        with torch.no_grad():
            prompt = prompt_net[id](torch.ones(1).to(conf.device))
            total_loss, _ = get_loss(batch, network, encoder, conf, prompt)
        eval_score += total_loss.item()
        step += 1
        if step % 100 == 0:
            end_time = time.perf_counter()
            print(f"step: {step}, time: {end_time - start_time}")
            start_time = end_time
    for i in temp_list:
        id = temp_dict[i]
        prompt = prompt_net[id](torch.ones(1).to(conf.device))
        prompt_np = prompt.data.cpu().numpy()
        np.save(os.path.join(conf.exp_dir, f"prompt_temp{i}_{id}.npy"), prompt_np, id)

    return eval_score / (tot_data - train_count)

def get_loss(batch, network_base, encoder, conf, prompt, grad=True):
    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    pos = torch.FloatTensor(np.array(pos)).view(-1, 3).to(conf.device)
    pos_diff = torch.FloatTensor(np.array(pos_diff)).view(-1, 3).to(conf.device)

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


    if grad:
        acc = network_base(h, pos, edges, m, prompt, pos_diff)
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

def get_loss_mix(batch, mix_net_temp, mix_net_struct, network, temp_dict, temp_list, encoder, prompt, data_pool, conf, prompt_pool):

    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    with torch.no_grad():
        h, real_acc = get_loss(batch, network, encoder, conf, prompt, grad=False)
        h = h.detach().cpu().numpy()

    lamb = random.random()

    new_temp_id = random.randint(0, len(temp_list) - 1)

    new_temp_id = (new_temp_id + 1) % len(temp_list)

    new_h_list = []
    for kk in range(h.shape[0]):
        new_h, new_real = find_similar(h[kk], temp_list[new_temp_id], data_pool, atom_num[kk], conf=conf)
        real_acc[kk] = real_acc[kk] * lamb + new_real * (1 - lamb)
        new_h_list.append(mix_net_struct(torch.from_numpy(h[kk]).to(conf.device), new_h, lamb))
    with torch.no_grad():
        prompt_new = prompt_pool[new_temp_id]

    prompt_mix = mix_net_temp(prompt, prompt_new, lamb)
    h = torch.cat(new_h_list).reshape(-1, conf.hidden_dim)
    real_acc = torch.FloatTensor(real_acc).to(conf.device)

    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    pos = torch.FloatTensor(np.array(pos)).view(-1, 3).to(conf.device)
    pos_diff = torch.FloatTensor(np.array(pos_diff)).view(-1, 3).to(conf.device)
    cnt1 = len(id1u)
    cnt2 = len(id2u)
    id1u = torch.LongTensor(id1u).to(conf.device)
    id1v = torch.LongTensor(id1v).to(conf.device)
    id2u = torch.LongTensor(id2u).to(conf.device)
    id2v = torch.LongTensor(id2v).to(conf.device)
    idu = torch.cat([id1u, id1v, id2u, id2v])
    idv = torch.cat([id1v, id1u, id2v, id2u])
    m = torch.cat([torch.ones(cnt1 * 2, 1), torch.zeros(cnt2 * 2, 1)]).to(conf.device)
    edges = [idu, idv]
    acc = network(h, pos, edges, m, prompt_mix, pos_diff)
    cri = torch.nn.MSELoss(reduction='sum')
    total_loss = cri(real_acc, acc)
    return total_loss

def find_similar(h, new_temp, data_pool, atom_id, maxinum = False, conf=None):
    # start_time = time.perf_counter()
    # pool_list = np.concatenate(data_pool[new_temp][atom_id][0])
    # end1 = time.perf_counter()
    data_list = data_pool[new_temp][atom_id][0]
    # end2 = time.perf_counter()
    h = torch.from_numpy(h).to(conf.device)
    dist = data_list - h
    with torch.no_grad():
        dist = torch.sum(dist**2, 1).unsqueeze(1)
        result = torch.min(dist, dim=0, keepdim=False)
    # end3 = time.perf_counter()
    id = result.indices.item()

    # if maxinum:
    #     id = np.argmin(dist)
    # else :
    #     now = 2
    #     min_dist = np.amin(dist)
    #     if min_dist < 0.001:
    #         min_dist = 0.01
    #     # print("min_dist", min_dist)
    #     id = np.where(dist < min_dist * now)[0]
    #     # while len(id) < 21:
    #     #     now = now + 0.1
    #     #     id = np.where(dist < min_dist * now)[0]
    #     id = id[random.randint(0, id.shape[0]-1)]

    # id = random.randint(0, len(data_pool[new_temp][atom_id][0]) - 1)
    # print(end1-start_time, end2 - end1, end3 - end2)
    sim_h = data_pool[new_temp][atom_id][0][id]
    real_acc = data_pool[new_temp][atom_id][1][id]

    return sim_h, real_acc


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='0809log', help='log name')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--train_data_dir', type=str, default='./save_traj', help='train_data_dir')
    # network settings
    parser.add_argument('--hidden_dim', type=int, default=32)
    # training parameters
    parser.add_argument('--tot_epoch', type=int, default=51)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--finetune_lr', type=float, default=0.01)
    parser.add_argument('--f_lr_decay_by', type=float, default=0.9)
    parser.add_argument('--f_lr_decay_every', type=float, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=10000)
    parser.add_argument('--test_data_rate', type=float, default=0.3)
    parser.add_argument('--finetune_epoch', type=int, default=3)
    parser.add_argument('--meta_epoch', type=int, default=3)
    parser.add_argument('--log_net_dir', type=str, default='0909_pretrain_log', help='exp logs directory')
    parser.add_argument('--log_net_epoch', type=int, default=40)
    parser.add_argument('--log_net_meta_dir', type=str, default='0909_pretrain_log', help='exp logs directory')
    parser.add_argument('--log_net_meta_epoch', type=int, default=40)
    parser.add_argument('--meta', action='store_true', default=False)
    parser.add_argument('--mix_epoch', type=int, default=1)
    parser.add_argument('--mix_every', type=int, default=16)
    parser.add_argument('--pool', type=int, default=8)

    conf = parser.parse_args()
    conf.ignore_joint_info = True
    ### prepare before training
    # make exp_name
    conf.exp_name = f'{conf.run_name}'

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    conf.tb_dir = os.path.join(conf.exp_dir, 'tb')
    if os.path.exists(conf.exp_dir):
        # response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
        # if response != 'y':
        #     exit(1)
        shutil.rmtree(conf.exp_dir)

    os.mkdir(conf.exp_dir)
    os.mkdir(conf.tb_dir)
    os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

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

