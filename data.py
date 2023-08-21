"""
    SAPIENVisionDataset
        Joint data loader for six primacts
        for panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
"""

import os
import torch
import torch.utils.data as data
import numpy as np
import math
from openmm import *
from openmm.app import *
# from openmmtools.integrators import VVVRIntegrator


def count_dis(pos1, pos2):
    return math.sqrt(
        (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]) + (
                pos1[2] - pos2[2]) * (pos1[2] - pos2[2]))

class MDDataset(data.Dataset):

    def __init__(self):
        self.data_buffer = []  # (gripper_direction_world, gripper_action_dis, gt_motion)

    def load_data(self, dirr, name, step = 1, temp=[280, 290, 300, 310], data_max = 10, data_min = 0):

        for tempp in temp:
            print("loading",tempp)
            for data_id in range(data_min, data_max):
                dir = dirr+f'_{tempp}_{data_id}'
                print("loading", dir)
                for oo in range(1, 990):
                    if oo % 100 == 0:
                        print(oo)
                    filename = os.path.join(dir, name+f'_{oo}.pdb')
                    pdb = PDBFile(filename)
                    topology = pdb.topology
                    positions = pdb.positions
                    pos = []
                    for position in positions:
                        pos.append([position[0]._value, position[1]._value, position[2]._value])
                    pos = np.array(pos)
                    atom_num = []
                    for i in topology.atoms():
                        atom_num.append(i.element.atomic_number)

                    # print(atom_num)
                    id1u = []
                    id1v = []
                    dis1 = []
                    dict = {}
                    for bond in topology.bonds():
                        id1u.append(bond[0].index)
                        id1v.append(bond[1].index)
                        if(dict.get(bond[0].index) == None):
                            dict[bond[0].index] = []
                        if (dict.get(bond[1].index) == None):
                            dict[bond[1].index] = []
                        dict[bond[0].index].append(bond[1].index)
                        dict[bond[1].index].append(bond[0].index)
                        # dis1.append(pos[bond[1].index]-pos[bond[0].index])

                    id2u = []
                    id2v = []
                    dis2 = []

                    for i in range(topology.getNumAtoms()):
                        for j in range(topology.getNumAtoms()):
                            if i>=j:
                                continue
                            if count_dis(pos[i],pos[j]) < 0.3:
                                if dict.get(i):
                                    flag = 0
                                    for k in dict[i]:
                                        if k == j:
                                            flag = 1
                                            break
                                    if flag == 1:
                                        continue
                                id2u.append(i)
                                id2v.append(j)
                                # dis2.append(pos[j]-pos[i])

                    filename = os.path.join(dir, name + f'_{oo+step}.pdb')
                    pdb = PDBFile(filename)
                    nxt_positions = pdb.positions
                    nxt_pos = []
                    for position in nxt_positions:
                        nxt_pos.append([position[0]._value, position[1]._value, position[2]._value])
                    nxt_pos = np.array(nxt_pos)

                    filename = os.path.join(dir, name + f'_{oo - 1}.pdb')
                    pdb = PDBFile(filename)
                    prev_positions = pdb.positions
                    diff_pos = []
                    cnt = 0
                    for position in prev_positions:
                        diff_pos.append([pos[cnt][0]-position[0]._value, pos[cnt][1]-position[1]._value, pos[cnt][2]-position[2]._value])
                        cnt = cnt + 1

                    self.data_buffer.append((pos, diff_pos, atom_num, id1u, id1v, dis1, id2u, id2v, dis2, nxt_pos, tempp))

    def __str__(self):
        return "MDDataLoader"

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, index):
        data_feats = self.data_buffer[index]
        return data_feats
