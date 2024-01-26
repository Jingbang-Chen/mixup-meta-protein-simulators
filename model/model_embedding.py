import numpy as np
import torch
from torch_scatter import scatter

class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, name=None):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        self.embeddings = torch.nn.Embedding(93, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):

    def __init__(self, atom_features, edge_features, out_features, input_dim):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.emb_dis = torch.nn.Linear(input_dim, edge_features, bias=True)
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, id_u, id_v, dis):

        h_u = h[id_u]
        h_v = h[id_v]

        m = self.emb_dis(dis)
        m = torch.cat([h_u, h_v, m], dim=-1)
        m = self.linear(m)
        return m

class AtomUpdateBlock(torch.nn.Module):

    def __init__(self, emb_size_atom, emb_size_edge):
        super().__init__()

        self.emb_size_edge = emb_size_edge

        self.mlp = torch.nn.Linear(emb_size_edge * 2 + emb_size_atom, emb_size_atom, bias=False)

    def forward(self, h, m1, m2, id1, id2, id3, id4):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]

        x1 = scatter(m1, id1, dim=0, dim_size=nAtoms, reduce="add")
        x2 = scatter(m2, id2, dim=0, dim_size=nAtoms, reduce="add")
        x3 = scatter(m1, id3, dim=0, dim_size=nAtoms, reduce="add")
        x4 = scatter(m2, id4, dim=0, dim_size=nAtoms, reduce="add")
        x1 = torch.sub(x1, x3)
        x2 = torch.sub(x2, x4)
        x = torch.cat([x1, x2, h], dim=-1)
        x = self.mlp(x)

        return x