from __future__ import print_function

from collections import defaultdict

import torch
from einops import rearrange, repeat

ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
ALPHABET_DICT = defaultdict(lambda: 0)
REVERSE_ALPHABET_DICT = {}
for i, letter in enumerate(ALPHABET):
    ALPHABET_DICT[i] = letter
    REVERSE_ALPHABET_DICT[letter] = i
ALPHABET_DICT[0] = "X"
REVERSE_ALPHABET_DICT["X"] = 0


def pna_aggregate(features, mask):
    """PNA aggregation of features shaped `(B, L, N, K)` from neighbors to nodes"""

    d = mask.unsqueeze(-1).sum(-2)
    # denom = ...
    # S_pos = torch.log(d + 1) / denom
    # S_neg = 1 / S_pos
    # print(f'{features.shape=}, {d.shape=}')
    # f_mean = features.sum(2) / d.unsqueeze(-1)
    # print(d)
    # d_mask = d.squeeze(-1) == 0
    n_mask = mask == 0
    d_mask = d.squeeze(-1) == 0
    if len(features.shape) == 5:
        d = d.unsqueeze(-1)
        d_mask = d_mask.squeeze(-1)
    f_mean = features.sum(2) / (d + 1e-6)
    feat = features.clone()
    feat[n_mask] = -float("inf")
    f_max = feat.max(2)[0]
    feat[n_mask] = float("inf")
    f_min = features.min(2)[0]
    f = torch.cat([f_mean, f_max, f_min], 2)
    f[d_mask] = 0
    return f


def metrics(
    S,
    log_probs,
    mask,
    X,
    X_pred,
    ignore_unknown,
    predict_all_atoms=False,
    predict_oxygens=False,
):
    if log_probs is None:
        true_false = torch.tensor(0)
        pp = torch.tensor(0)
    else:
        max_prob, S_argmaxed = torch.max(torch.softmax(log_probs, -1), -1)  # [B, L]
        pp = torch.exp(-(torch.log(max_prob) * mask).sum(-1) / mask.sum(-1)).sum()
        if ignore_unknown:
            S_argmaxed += 1
        true_false = (S == S_argmaxed).float()

    if not isinstance(X_pred, list):
        X_pred = [X_pred]
    for i, x in enumerate(X_pred):
        if x is None:
            rmsd = torch.tensor(0)
        else:
            rmsd = []
            dims = [[2]]
            if predict_all_atoms:
                if predict_oxygens:
                    dims.append([0, 1, 3])
                else:
                    dims.append([0, 1])
            for dim in dims:
                mask_ = mask.unsqueeze(-1).unsqueeze(-1)
                num_atoms = len(dim)
                sqd = (X[:, :, dim, :] - x[:, :, dim, :]) ** 2
                mean_sqd = (sqd * mask_).sum(-1).sum(-1).sum(-1)
                mean_sqd = mean_sqd / (mask.sum(-1) * num_atoms)
                rmsd.append(torch.sqrt(mean_sqd).sum().detach())
            if len(rmsd) == 1:
                rmsd = rmsd[0]
        if len(X_pred) > 1:
            print(f"{i}: {rmsd / x.shape[0]}")
    return true_false.detach(), rmsd, pp.detach()


def get_seq_loss(S, logits, mask, no_smoothing, ignore_unknown, weight=0.1):
    """Negative log probabilities"""
    if logits is None:
        return torch.tensor(0)
    S_onehot = torch.nn.functional.one_hot(S, 21).float()
    if not no_smoothing:
        S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    if ignore_unknown:
        mask_S = mask * (S != 0)
        S_onehot = S_onehot[:, :, 1:]
    else:
        mask_S = mask

    loss = torch.nn.CrossEntropyLoss(reduction="none")(
        logits.transpose(-1, -2), S_onehot.transpose(-1, -2)
    )
    loss = loss[(mask_S).bool()].sum() / mask_S.sum()
    return loss


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    if isinstance(h_neighbors, tuple):
        h_nn = (torch.cat([h_neighbors[0], h_nodes], -1), h_neighbors[1])
    else:
        h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def context_to_rbf(context, min_rbf, max_rbf, nb_rbf):
    device = context.device
    mu = torch.linspace(min_rbf, max_rbf, nb_rbf, device=device)
    mu = mu.view([1, -1])
    sigma = (max_rbf - min_rbf) / nb_rbf
    context_expand = torch.unsqueeze(context, -1)
    RBF = torch.exp(-(((context_expand - mu) / sigma) ** 2))
    return RBF


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def get_std_opt(parameters, d_model, step, lr=None):
    if lr is None:
        return NoamOpt(
            d_model,
            2,
            4000,
            torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9),
            step,
        )
    else:
        return torch.optim.Adam(parameters, lr=lr)


def get_vectors(X, mask, E_idx, edge=False):
    feature_types = ["backbone_orientation", "c_beta", "backbone_atoms"]
    other_vecs = X[:, :, 4:, :]
    b = X[:, :, 2, :] - X[:, :, 0, :]
    c = X[:, :, 1, :] - X[:, :, 2, :]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 2, :]
    Ca = X[:, :, 2, :]
    vectors = []
    if "backbone_atoms" in feature_types:
        vectors.append(X[:, :, [0, 1, 3]] - X[:, :, [2]])
    if "backbone_orientation" in feature_types:
        orientation = torch.zeros((X.shape[0], X.shape[1], 2, 3)).to(X.device)
        orientation[:, 1:, 0] = X[:, 1:, 2] - X[:, :-1, 2]
        orientation[:, :-1, 1] = X[:, :-1, 2] - X[:, 1:, 2]
        orientation = orientation / (
            torch.norm(orientation, dim=-1, keepdim=True) + 1e-7
        )
        orientation = orientation * mask[..., None, None]
        orientation[:, 1:, 0] *= mask[:, :-1, None]
        orientation[:, :-1, 1] *= mask[:, 1:, None]
        vectors.append(orientation)
    if "c_beta" in feature_types:
        diff = (Cb - Ca).unsqueeze(2)
        vectors.append(diff / (torch.norm(diff, dim=-1, keepdim=True) + 1e-7))
    if len(vectors) > 0:
        vectors = torch.cat(vectors, dim=2)
    else:
        vectors = None
    vectors = torch.cat([vectors, other_vecs], dim=-2)
    if edge:
        X_gather = gather_nodes(rearrange(X, "b l a d -> b l (a d)"), E_idx)
        X_gather = rearrange(X_gather, "b l n (a d) -> b l n a d", d=3)
        edge_vectors = X_gather - repeat(X, "b l a d -> b l n a d", n=E_idx.shape[2])
        edge_vectors = edge_vectors[:, :, :, 2, :].unsqueeze(-2)
        edge_vectors = edge_vectors / (
            torch.norm(edge_vectors, dim=-1, keepdim=True) + 1e-7
        )
    else:
        edge_vectors = None
    return vectors, edge_vectors


def to_pyg(h_V, h_E, E_idx, mask, vectors=None):
    h = rearrange(h_V, "b l f -> (b l) f")
    if vectors is not None:
        vectors_ = rearrange(vectors, "b l k d -> (b l) k d")
    else:
        vectors_ = None
    edge_attr = rearrange(h_E, "b l n f -> (b l n) f")
    n_index = rearrange(E_idx, "b l n -> 1 b (l n)")
    s_index = repeat(
        torch.tensor(range(E_idx.shape[1])),
        "l -> 1 b (l n)",
        b=E_idx.shape[0],
        n=E_idx.shape[2],
    ).to(E_idx.device)
    edge_index = torch.cat([n_index, s_index], dim=0)
    edge_index += (
        repeat(
            torch.tensor(range(E_idx.shape[0])),
            "b -> k b (l n)",
            l=E_idx.shape[1],
            n=E_idx.shape[2],
            k=2,
        ).to(E_idx.device)
        * E_idx.shape[1]
    )
    edge_index = rearrange(edge_index, "e b k -> e (b k)")
    mask_edge = rearrange(mask, "b l -> (b l)")[edge_index[0]]
    batch = repeat(
        torch.tensor(range(E_idx.shape[0])), "b -> (b l)", l=E_idx.shape[1]
    ).to(E_idx.device)
    return h, edge_attr, edge_index, vectors_, batch, mask_edge


def from_pyg(h, edge_attr, vectors, b, l, n):
    if h is not None:
        h_V = rearrange(h, "(b l) f -> b l f", l=l)
    else:
        h_V = None
    if edge_attr is not None:
        h_E = rearrange(edge_attr, "(b l n) f -> b l n f", l=l, n=n)
    else:
        h_E = None
    if vectors is not None:
        vectors_ = rearrange(vectors, "(b l) k d -> b l k d", l=l)
    else:
        vectors_ = None
    return h_V, h_E, vectors_
