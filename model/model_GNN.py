import os
import torch
from .model_embedding import AtomEmbedding, EdgeEmbedding, AtomUpdateBlock

class GNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_emb = AtomEmbedding(hidden_dim)
        self.edge_emb1 = EdgeEmbedding(hidden_dim, hidden_dim, hidden_dim, 3)
        self.edge_emb2 = EdgeEmbedding(hidden_dim, hidden_dim, hidden_dim, 3)
        self.update_edge1 = EdgeEmbedding(hidden_dim, hidden_dim, hidden_dim, 3)
        self.update_edge2 = EdgeEmbedding(hidden_dim, hidden_dim, hidden_dim, 3)
        # self.update_edge2 = EdgeEmbedding(hidden_dim, hidden_dim, hidden_dim, 3)
        # self.update_edge3 = EdgeEmbedding(hidden_dim, hidden_dim, hidden_dim, 3)
        self.update_atom = AtomUpdateBlock(hidden_dim, hidden_dim)
        self.update_atom_again = AtomUpdateBlock(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 3)

    def forward(self, atom_num, dis1, dis2, id1u, id1v, id2u, id2v):
        h = self.atom_emb(atom_num)
        m1 = self.edge_emb1(h, id1u, id1v, dis1)
        m2 = self.edge_emb2(h, id2u, id2v, dis2)
        h = self.update_atom(h, m1, m2, id1u, id1v, id2u, id2v)
        m1 = self.update_edge1(h, id1u, id1v, dis1)
        m2 = self.update_edge2(h, id2u, id2v, dis2)
        h = self.update_atom_again(h, m1, m2, id1u, id1v, id2u, id2v)
        delta = self.out(h)
        return delta