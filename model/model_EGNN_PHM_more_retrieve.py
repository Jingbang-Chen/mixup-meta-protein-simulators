from torch import nn
import torch
import numpy as np
from phc.hypercomplex.layers import PHMLinear
import torch.nn.functional as F
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, phm_dim = 4, change_coord = True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.phm_dim = phm_dim
        self.ave = 1
        self.change_coord = change_coord
        self.act_fn = act_fn
        edge_coords_nf = 1

        self.phm_node_layer = PHMLinear(hidden_nf*4, hidden_nf, bias=True, phm_dim=phm_dim)
        self.phm_edge_layer = PHMLinear(hidden_nf, hidden_nf, bias=True, phm_dim=phm_dim)
        self.weight_node = nn.Sequential(
            nn.Linear(hidden_nf, phm_dim),
            act_fn,
            nn.Linear(phm_dim, phm_dim),
        )
        self.weight_edge = nn.Sequential(
            nn.Linear(hidden_nf, phm_dim),
            act_fn,
            nn.Linear(phm_dim, phm_dim),
        )
        if self.change_coord:
            self.phm_coord_edge = PHMLinear(hidden_nf*2, hidden_nf, bias=True, phm_dim=phm_dim)
            self.phm_coord_point = PHMLinear(hidden_nf*3, hidden_nf, bias=True, phm_dim=phm_dim)
            self.weight_coord_edge = nn.Sequential(
                nn.Linear(hidden_nf, phm_dim),
                act_fn,
                nn.Linear(phm_dim, phm_dim),
            )
            self.weight_coord_point = nn.Sequential(
                nn.Linear(hidden_nf, phm_dim),
                act_fn,
                nn.Linear(phm_dim, phm_dim),
            )

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d + hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )



        if self.change_coord:
            coord_mlp_edge = []
            coord_mlp_edge.append(act_fn)
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            coord_mlp_edge.append(layer)
            if self.tanh:
                coord_mlp_edge.append(nn.Tanh())
            self.coord_mlp_edge = nn.Sequential(*coord_mlp_edge)

            coord_mlp_point = []
            coord_mlp_point.append(act_fn)
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            coord_mlp_point.append(layer)
            if self.tanh:
                coord_mlp_point.append(nn.Tanh())
            self.coord_mlp_point = nn.Sequential(*coord_mlp_point)
        else :
            self.coord_mlp_edge = None
            self.coord_mlp_point = None

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr, prompt):
        edge_num = source.shape[0]
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr, prompt.repeat(edge_num,1)], dim=1)
        # npo = out.detach().cpu().numpy()
        # if (np.any(np.isnan(npo))):
        #     print("bad before edge mlp")
        # print("edge_att_before_mlp", out)

        out = self.edge_mlp(out)
        out = self.phm_edge(out, prompt)
        out = self.act_fn(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        # print("edge_att_after_mlp", out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, prompt, node_prompt):
        row, col = edge_index
        p_n = x.shape[0]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = agg * self.ave

        agg = torch.cat([x, agg, prompt.repeat(p_n,1), node_prompt], dim=1)
        # print("node_att_before", agg)
        out = self.phm_node(agg, prompt)
        out = self.node_mlp(out)
        # print("node_att_after", out)
        if self.residual:
            out = x + out
        return out, agg

    def phm_node(self, x, prompt):
        weight = F.softmax(self.weight_node(prompt), 0)
        x = self.phm_node_layer(x, weight = weight)
        return x

    def phm_edge(self, x, prompt):
        weight = F.softmax(self.weight_edge(prompt), 0)
        x = self.phm_edge_layer(x, weight = weight)
        return x

    def get_acc_edge(self, x, prompt):
        weight = F.softmax(self.weight_coord_edge(prompt), 0)
        x = self.phm_coord_edge(x, weight = weight)
        x = self.coord_mlp_edge(x)
        return x

    def get_acc_point(self, x, prompt):
        weight = F.softmax(self.weight_coord_point(prompt), 0)
        x = self.phm_coord_point(x, weight = weight)
        x = self.coord_mlp_point(x)
        return x

    def get_acc(self, h, edge_index, coord_diff, edge_feat, prompt, node_prompt):
        row, col = edge_index
        e_n = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat, prompt.repeat(e_n,1)], dim=1)
        coord_diff = F.normalize(coord_diff, p=2, dim=1)
        trans = coord_diff * self.get_acc_edge(edge_feat, prompt)
        agg = unsorted_segment_sum(trans, row, num_segments=h.size(0))
        agg = agg * self.ave
        h = torch.cat([h, prompt.repeat(h.size(0),1), node_prompt], dim=1)
        agg = agg * self.get_acc_point(h, prompt)

        return agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, prompt = None, node_prompt=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, prompt)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, prompt, node_prompt)
        acc = None
        if self.change_coord:
            acc = self.get_acc(h, edge_index, coord_diff, edge_feat, prompt, node_prompt)

        return h, coord, acc


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False, step=1):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        # self.embedding_in = torch.nn.Embedding(93, hidden_nf)
        # # init by uniform distribution
        # torch.nn.init.uniform_(self.embedding_in.weight, a=-np.sqrt(3), b=np.sqrt(3))
        self.embedding_in = nn.Linear(self.hidden_nf*3+3, self.hidden_nf)
        self.embedding_in1 = nn.Linear(self.hidden_nf, 3)
        self.embedding_in2 = nn.Linear(self.hidden_nf*2, 3)
        self.embedding_in3 = nn.Linear(6, self.hidden_nf)
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.act_fn = act_fn
        for i in range(0, n_layers - 1):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, change_coord=False))
        self.add_module("gcl_%d" % (n_layers - 1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                            act_fn=act_fn, residual=residual, attention=attention,
                                            normalize=normalize, tanh=tanh))
        self.step = step

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, prompt, pos_diff, point_h, point_acc, point_h_2, point_acc_2, point_prompt_2):
        point_num = h.shape[0]
        # print(h.shape, prompt.repeat(point_num, 1).shape, pos_diff.shape)
        point_h = point_h - h
        point_h_2 = point_h_2 - h
        point_prompt_2 = point_prompt_2 - prompt.repeat(point_num, 1)
        point_prompt_2 = point_prompt_2.detach()
        delta_h = self.embedding_in1(point_h)
        point_h_2 = torch.cat([point_h_2, point_prompt_2], dim=1)
        delta_h_2 = self.embedding_in2(point_h_2)
        point_acc = point_acc + delta_h
        point_acc_2 = point_acc_2 + delta_h_2
        point_prompt = torch.cat([point_acc, point_acc_2], dim=1)
        point_prompt = self.embedding_in3(point_prompt)
        h = torch.cat([h, prompt.repeat(point_num, 1), point_prompt, pos_diff], dim=1)
        h = self.embedding_in(h)
        h = self.act_fn(h)
        # print("init h",h)
        for i in range(0, self.n_layers - 1):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, prompt = prompt)
            if(np.isnan(x[0][0].item())):
                print("nan in layer ",i)
        h, x, acc = self._modules["gcl_%d" % (self.n_layers - 1)](h, edges, x, edge_attr=edge_attr, prompt = prompt)

        return acc

    def first_layer(self, h, x, edges, edge_attr, prompt, pos_diff, point_h, point_acc, point_h_2, point_acc_2, point_prompt_2):
        point_num = h.shape[0]
        # print(h.shape, prompt.repeat(point_num, 1).shape, pos_diff.shape)
        point_h = point_h - h
        point_h_2 = point_h_2 - h
        point_prompt_2 = point_prompt_2 - prompt.repeat(point_num, 1)
        delta_h = self.embedding_in1(point_h)
        point_h_2 = torch.cat([point_h_2, point_prompt_2], dim=1)
        delta_h_2 = self.embedding_in2(point_h_2)
        point_acc = point_acc + delta_h
        point_acc_2 = point_acc_2 + delta_h_2
        point_prompt = torch.cat([point_acc, point_acc_2], dim=1)
        point_prompt = self.embedding_in3(point_prompt)
        h = torch.cat([h, prompt.repeat(point_num, 1), point_prompt, pos_diff], dim=1)
        h = self.embedding_in(h)
        h = self.act_fn(h)
        # print("init h",h)
        h, x, _ = self._modules["gcl_%d" % 0](h, edges, x, edge_attr=edge_attr, prompt=prompt, node_prompt=point_prompt)
        return h, point_prompt

    def layer_x(self, h, x, edges, edge_attr, prompt, layer_id, point_prompt):
        h, x, acc = self._modules["gcl_%d" % layer_id](h, edges, x, edge_attr=edge_attr, prompt=prompt, node_prompt=point_prompt)
        return h, x, acc

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
