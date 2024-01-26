from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class E_GCL_encode(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, phm_dim = 4):
        super(E_GCL_encode, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.ave = 1
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=True),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=True),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        # npo = out.detach().cpu().numpy()
        # if (np.any(np.isnan(npo))):
        #     print("bad before edge mlp")
        # print("edge_att_before_mlp", out)
        out = self.edge_mlp(out)
        npo = out.detach().cpu().numpy()
        if (np.any(np.isnan(npo))):
            print("bad in edge mlp")
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        # print("edge_att_after_mlp", out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = agg * self.ave
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        # print("node_att_before", agg)
        out = self.node_mlp(agg)
        # print("node_att_after", out)
        if self.residual:
            out = x + out
        return out, agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h


class EGNN_encode(nn.Module):
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

        super(EGNN_encode, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = torch.nn.Embedding(93, hidden_nf)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embedding_in.weight, a=-np.sqrt(3), b=np.sqrt(3))
        self.embedding_in2 = nn.Linear(self.hidden_nf + 3, self.hidden_nf)
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.act_fn = act_fn

        self.GCL_layer = E_GCL_encode(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh)
        self.step = step


    def forward(self, h, x, edges, edge_attr, pos_diff):
        h = self.embedding_in(h - 1)
        point_num = h.shape[0]
        # print(h.shape, prompt.repeat(point_num, 1).shape, pos_diff.shape)
        h = torch.cat([h, pos_diff], dim=1)
        h = self.embedding_in2(h)
        h = self.act_fn(h)
        h = self.GCL_layer(h, edges, x, edge_attr=edge_attr)

        return h


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