import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import logging
import torch
import torch.utils.data
from lion_pytorch import Lion
from data import MDDataset
from model import model_EGNN_base, model_EGNN_encoder, automix
from tensorboardX import SummaryWriter

def train(conf):
    # create training and validation datasets and data loaders
    # load network model

    if conf.more:
        train_temp = [280, 290, 300, 310, 320]
    else:
        train_temp = [280, 300, 320]
    # test_temp = [285, 295, 305]
    # test_out_temp = [350]
    logger = setup_logging('job', conf.exp_dir, console=True)
    logger.info(conf)
    # create models
    encoder = model_EGNN_encoder.EGNN_encode(in_node_nf=1, hidden_nf=conf.hidden_dim, out_node_nf=1, in_edge_nf=1)
    encoder.to(conf.device)
    encoder_opt = Lion(filter(lambda p: p.requires_grad, encoder.parameters()), lr=conf.lr,
                                   weight_decay=conf.weight_decay)


    network = model_EGNN_base.EGNN_base(in_node_nf=1, hidden_nf=conf.hidden_dim, out_node_nf=1, in_edge_nf=1)

    network_opt = Lion(filter(lambda p: p.requires_grad, network.parameters()), lr=conf.lr,
                                   weight_decay=conf.weight_decay)
    # learning rate scheduler

    mix_temp_net = automix.Automix(hidden_nf=conf.hidden_dim, device=conf.device)
    mix_temp_net.to(conf.device)
    mix_temp_opt = Lion(filter(lambda p: p.requires_grad, mix_temp_net.parameters()), lr=conf.lr,
                                   weight_decay=conf.weight_decay)

    mix_struct_net = automix.Automix(hidden_nf=conf.hidden_dim, device=conf.device)
    mix_struct_net.to(conf.device)
    mix_struct_opt = Lion(filter(lambda p: p.requires_grad, mix_struct_net.parameters()), lr=conf.lr,
                                    weight_decay=conf.weight_decay)


    # send parameters to device
    network.to(conf.device)

    prompt_net = []
    prompt_opt = []
    prompt_lr = []
    temp_dict = {}
    temp_list = train_temp
    for i in range(len(temp_list)):
        temp_dict[temp_list[i]] = i

    for i in range(len(temp_list)):
        prompt_net.append(model_EGNN_base.Prompt(hidden_nf=conf.hidden_dim))
        prompt_opt.append(Lion(filter(lambda p: p.requires_grad, prompt_net[i].parameters()), lr=conf.lr,
                                   weight_decay=conf.weight_decay))
        prompt_net[i].to(conf.device)


    # load dataset
    train_dataset = MDDataset()
    # val_dataset = MDDataset()
    # val_middle_dataset = MDDataset()
    # val_out_dataset = MDDataset()

    # val_dataset = MDDataset([conf.primact_type], data_features)

    train_dataset.load_data(conf.train_data_dir, name='traj', temp=train_temp, data_max=10)
    # val_dataset.load_data(conf.train_data_dir, name='traj', temp=train_temp, data_max=10, data_min=9)
    # val_middle_dataset.load_data(conf.train_data_dir, name='traj', temp=test_temp, data_max=3)
    # val_out_dataset.load_data(conf.train_data_dir, name='traj', temp=test_out_temp, data_max=3)

    # start training
    train_writer = SummaryWriter(os.path.join(conf.tb_dir, 'train'))
    # val_writer = SummaryWriter(os.path.join(conf.tb_dir, 'val'))

    # last_train_console_log_step, last_val_console_log_step = None, None

    start_epoch = 0

    network_opt.zero_grad()
    # train for every epoch
    step = 0
    losses = 0.0
    step_prompt = [0, 0, 0, 0, 0]
    tot_mix_step = 0
    lowest_loss = 10000000

    for epoch in range(conf.tot_epoch):

        train_batch_ind = 0
        tot_data = len(train_dataset.data_buffer)
        train_count = tot_data
        arr = np.arange(len(train_dataset.data_buffer))
        np.random.shuffle(arr)
        # mix_count = train_count / 2
        for p in network.parameters():
            p.requires_grad = False
        ############################# filling data pool ###########################################
        if conf.mix_every < 256:    
            print("filling data pool")
            start_time = time.perf_counter()
            data_pool = {}
            for ii in temp_list:
                data_pool[ii] = {}
            temp_l1 = []
            num_l1 = []
            for idd in range(train_count):
                if idd % conf.mix_every > conf.pool:
                    continue
                batch = train_dataset.data_buffer[arr[idd]]
                pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
                with torch.no_grad():
                    id = temp_dict[temp]
                    h, real_acc = get_loss(batch, network, encoder, conf, prompt_net[id], grad=False)
                    #h is the feature after encoder
                for kk in range(h.shape[0]):
                    num_id = atom_num[kk]
                    if data_pool[temp].get(num_id) == None:
                        data_pool[temp][num_id] = [[], []]
                        data_pool[temp][num_id][0].append(h[kk])
                        data_pool[temp][num_id][1].append(real_acc[kk])
                        # print(data_pool[temp][num_id][0])
                        temp_l1.append(temp)
                        num_l1.append(num_id)
                    else :
                        # print(data_pool[temp][num_id][0].shape)
                        data_pool[temp][num_id][0].append(h[kk])
                        data_pool[temp][num_id][1].append(real_acc[kk])
            for kk in range(len(temp_l1)):
                temp = temp_l1[kk]
                num = num_l1[kk]
                data_pool[temp][num][0] = torch.cat(data_pool[temp][num][0]).reshape(-1, conf.hidden_dim)

            end_time = time.perf_counter()
            print(f"done filling data pool, time: {end_time - start_time}")
            ############################################################################################
            for p in mix_struct_net.parameters():
                p.requires_grad = True
            if epoch < conf.start_mix_epoch and epoch > 0:
                ########################### adapting automix struct###############################################
                print("adapt automix struct")
                mix_step = 0
                mix_struct_opt.zero_grad()
                t_loss = 0
                start_time = time.perf_counter()
                for kk in range(conf.mix_epoch):
                    for idd in range(train_count):
                        if idd % conf.mix_every <= conf.pool:
                            continue
                        batch = train_dataset.data_buffer[arr[idd]]

                        loss = get_loss_mix_struct_only(batch, mix_struct_net, network, temp_dict, encoder, prompt_net,
                                            data_pool, conf)
                        loss.backward()
                        t_loss += loss.item()
                        mix_step += 1
                        tot_mix_step += 1
                        if mix_step % 16 == 0:
                            mix_struct_opt.step()
                            mix_struct_opt.zero_grad()
                            end_time = time.perf_counter()
                            # print("loss:", t_loss / 16, "time:", end_time - start_time)
                            t_loss = 0
                            start_time = end_time
                            train_writer.add_scalar('mix_loss', t_loss, tot_mix_step)
            #     ###########################################################################################

            if epoch >= conf.start_mix_epoch:
                ########################### adapting automix temp ########################################
                print("adapt automix all")
                for p in mix_temp_net.parameters():
                    p.requires_grad = True
                t_loss = 0
                mix_step = 0
                mix_temp_opt.zero_grad()
                start_time = time.perf_counter()
                for kk in range(conf.mix_epoch):
                    for idd in range(train_count):
                        if random.random() > 0.3:
                            continue
                        batch = train_dataset.data_buffer[arr[idd]]

                        loss = get_loss_mix(batch, mix_temp_net, mix_struct_net, network, temp_dict, temp_list, encoder, prompt_net, data_pool, conf)
                        loss.backward()
                        t_loss += loss.item()
                        mix_step += 1
                        tot_mix_step += 1
                        if mix_step % 16 == 0:
                            mix_temp_opt.step()
                            mix_temp_opt.zero_grad()
                            mix_struct_opt.step()
                            mix_struct_opt.zero_grad()
                            end_time = time.perf_counter()
                            # print("loss:", t_loss / 16, "time:", end_time - start_time)
                            start_time = end_time
                            train_writer.add_scalar('mix_loss', t_loss, tot_mix_step)
                            t_loss = 0

                print("done adapt automix all")

                for p in mix_struct_net.parameters():
                    p.requires_grad = True
            ###########################################################################################
        for p in network.parameters():
            p.requires_grad = True
        for p in mix_struct_net.parameters():
            p.requires_grad = False
        for p in mix_temp_net.parameters():
            p.requires_grad = False

        network_opt.zero_grad()
        start_time = time.perf_counter()
        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], profile_memory = True) as prof:

        for idd in range(train_count):
            batch = train_dataset.data_buffer[arr[idd]]

            # set models to training mode
            network.train()
            # for i in range(conf.batch_size):

            # forward pass (including logging)
            pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
            id = temp_dict[temp]

            total_loss, _ = get_loss(batch, network, encoder, conf, prompt_net[id])
            total_loss.backward()

            step += 1
            if epoch > conf.start_mix_loss and conf.mix_every <256:
                if epoch % conf.mix_anneal == 0 and conf.mix_every > 16:
                    conf.mix_every = conf.mix_every / 2
                if idd % conf.mix_every == (conf.mix_every - 1):

                    if epoch < conf.start_mix_epoch:
                        mix_loss = get_loss_mix_struct_only(batch, mix_struct_net, network, temp_dict, encoder,
                                                            prompt_net, data_pool, conf)
                    else :
                        mix_loss = get_loss_mix(batch, mix_temp_net, mix_struct_net, network, temp_dict, temp_list, encoder,
                                                prompt_net, data_pool, conf)
                    mix_loss.backward()
            
            # optimize one step

            if step % 16 == 0:
                network_opt.step()
                network_opt.zero_grad()
                encoder_opt.step()
                encoder_opt.zero_grad()
            step_prompt[id] += 1
            if step_prompt[id] % 16 == 0:
                prompt_opt[id].step()
                prompt_opt[id].zero_grad()


            losses += total_loss.item()
            train_batch_ind += 1
        losses /= train_count
        logger.info('Epoch %d, train loss %.5f', epoch, losses)
        if losses < lowest_loss:
            lowest_loss = losses
            with torch.no_grad():
                print('Saving checkpoint ...... ')
                torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '50-network.pth' ))
                torch.save(encoder.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '50-encoder.pth' ))
                torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '50-network_opt.pth' ))
                torch.save(encoder_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '50-encoder_opt.pth' ))

                torch.save(mix_struct_net.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '50-mixs.pth' ))
                torch.save(mix_temp_net.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '50-mixt.pth' ))
                torch.save(mix_struct_opt.state_dict(),
                            os.path.join(conf.exp_dir, 'ckpts', '50-mixs_opt.pth' ))
                torch.save(mix_temp_opt.state_dict(),
                            os.path.join(conf.exp_dir, 'ckpts', '50-mixt_opt.pth' ))
                for kk in range(len(temp_list)):
                    torch.save(prompt_net[kk].state_dict(), os.path.join(conf.exp_dir, 'ckpts', f'50-prompt_{kk}.pth'))
                    torch.save(prompt_opt[kk].state_dict(),
                                os.path.join(conf.exp_dir, 'ckpts', f'50-promptopt_{kk}.pth'))
            logger.info('A better checkpoint is saved !')
    # torch.save(train_dataset, os.path.join(conf.exp_dir, 'ckpts', '50-train_dataset.pth' % epoch))


        train_writer.add_scalar('loss', losses, step)
        losses = 0
        end_time = time.perf_counter()
        print("time consume:", end_time - start_time)
        start_time = end_time
            


def get_encode(batch, encoder, conf):
    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    pos = torch.FloatTensor(np.array(pos)).view(-1, 3).to(conf.device)
    pos_diff = torch.FloatTensor(np.array(pos_diff)).view(-1, 3).to(conf.device)
    cnt1 = len(id1u)
    cnt2 = len(id2u)
    h = torch.from_numpy(np.array(atom_num)).to(conf.device)
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
    return h

def get_loss_mix(batch, mix_net_temp, mix_net_struct, network, temp_dict, temp_list, encoder, prompt_net, data_pool, conf):

    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    with torch.no_grad():
        id = temp_dict[temp]
        h, real_acc = get_loss(batch, network, encoder, conf, prompt_net[id], grad=False)
        h = h.detach().cpu().numpy()

    lamb = random.random()

    now_temp_id = temp_dict[temp]
    new_temp_id = random.randint(0, len(temp_list) - 1)
    if new_temp_id == now_temp_id:
        new_temp_id = (new_temp_id + 1) % len(temp_list)

    new_h_list = []
    for kk in range(h.shape[0]):
        new_h, new_real = find_similar(h[kk], temp_list[new_temp_id], data_pool, atom_num[kk], conf=conf)
        real_acc[kk] = real_acc[kk] * lamb + new_real * (1 - lamb)
        new_h_list.append(mix_net_struct(torch.from_numpy(h[kk]).to(conf.device), new_h, lamb))
    with torch.no_grad():
        prompt = prompt_net[now_temp_id](torch.ones(1).to(conf.device))
        prompt_new = prompt_net[new_temp_id](torch.ones(1).to(conf.device))

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

def get_loss(batch, network_base, encoder, conf, prompt_net, grad=True):
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
    feat = encoder(h, pos, edges, m, pos_diff)


    if grad:
        prompt = prompt_net(torch.ones(1).to(conf.device))
        acc = network_base(feat, pos, edges, m, prompt, pos_diff)
        new_vel = pos_diff + acc
        pred_nxt_pos = pos + new_vel

        nxt_pos = torch.FloatTensor(np.array(nxt_pos)).view(-1, 3).to(conf.device)
        cri = torch.nn.MSELoss(reduction='sum')
        total_loss = cri(pred_nxt_pos, nxt_pos)

        return total_loss, None
    else :
        real_vel = nxt_pos - pos.detach().cpu().numpy()
        real_acc = real_vel - pos_diff.detach().cpu().numpy()
        return feat, real_acc

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

def get_loss_mix_struct_only(batch, mix_net_struct, network, temp_dict, encoder, prompt_net, data_pool, conf):

    pos, pos_diff, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, temp = batch
    with torch.no_grad():
        id = temp_dict[temp]
        h, real_acc = get_loss(batch, network, encoder, conf, prompt_net[id], grad=False)
        h = h.detach().cpu().numpy()

    lamb = random.random()
    new_h_list = []
    for kk in range(h.shape[0]):
        new_h, new_real = find_similar(h[kk], temp, data_pool, atom_num[kk], conf=conf)
        real_acc[kk] = real_acc[kk] * lamb + new_real * (1 - lamb)
        new_h_list.append(mix_net_struct(torch.from_numpy(h[kk]).to(conf.device), new_h, lamb))
    with torch.no_grad():
        prompt = prompt_net[temp_dict[temp]](torch.ones(1).to(conf.device))

    h = torch.cat(new_h_list).reshape(-1, conf.hidden_dim)
    # print(h.shape)
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
    acc = network(h, pos, edges, m, prompt, pos_diff)
    cri = torch.nn.MSELoss(reduction='sum')
    total_loss = cri(real_acc, acc)
    return total_loss

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output_mix.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
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
    parser.add_argument('--tot_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=10000)
    parser.add_argument('--start_mix_epoch', type=int, default=40,help='start prompt mix')
    parser.add_argument('--start_mix_loss', type=int, default=50,help='start using automix in the training of base network')    
    parser.add_argument('--mix_epoch', type=int, default=1)
    parser.add_argument('--mix_every', type=int, default=256)
    parser.add_argument('--mix_anneal', type=int, default=10)
    parser.add_argument('--pool', type=int, default=8)
    parser.add_argument('--more', action='store_true', default=False)
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

