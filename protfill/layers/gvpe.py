from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from protfill.layers.base_models import *
from protfill.utils.model_utils import get_vectors, pna_aggregate, gather_nodes
from math import sqrt


def batch_select(input, indexes):
    """
    Select elements in a batched tensor given indexes. Selection is performed along dimension 1 of the input tensor.
    :param input: `torch.Tensor` of size B x N x ...
    :param indexes: `torch.Tensor` of integers >= 0 and < N.
                    If indexes is 1D, it is repeated along the batch dimension.
                    If indexes is 2D, it must be of shape B x K, with K being as large as desired (index repetitions are allowed)
    :return: `torch.Tensor` of size B x K x ... (last dimensions are the same as in the input tensor)
    """
    batch_size = input.shape[0]
    if len(indexes.shape) == 1:
        indexes = repeat(indexes, "n -> b n", b=batch_size)
    flat_input = torch.flatten(input, start_dim=0, end_dim=1)
    index = indexes + repeat(
        input.shape[1] * torch.arange(batch_size).to(input.device),
        "b -> b n",
        n=indexes.shape[1],
    )
    index = torch.flatten(index, start_dim=0, end_dim=1)
    return torch.index_select(flat_input, 0, index).view(
        batch_size, indexes.shape[1], *input.shape[2:]
    )


def tuple_batch_select(input, indexes):
    """
    Perform the `batch_select` operation on a tuple (s, V) of `torch.Tensor`.
    """
    if type(input) == tuple:
        return (batch_select(input[0], indexes), batch_select(input[1], indexes))
    return batch_select(input, indexes)


def tuple_edgify(input, n_neighbors):
    shape = input[0].shape if type(input) == tuple else input.shape
    if shape[1] < n_neighbors**2:
        k = int(sqrt(shape[1]))
    else:
        k = n_neighbors
    if type(input) == tuple:
        return (
            rearrange(input[0], "b (n k) c -> b n k c", k=k),
            rearrange(input[1], "b (n k) c d -> b n k c d", k=k),
        )
    return rearrange(input, "b (n k) c -> b n k c", k=k)


def tuple_nodify(input):
    if type(input) == tuple:
        return (
            rearrange(input[0], "b n k c -> b (n k) c"),
            rearrange(input[1], "b n k c d -> b (n k) c d"),
        )
    return rearrange(input, "b n k c -> b (n k) c")


def tuple_node_expand(nodes, n_neighbors):
    """
    Expand the node tuple to make it match the edges dimensions.
    Handles both cases if nodes is just a scalar features `torch.Tensor` or if it is a (s, V) tuple.
    """
    if type(nodes) == tuple:
        return (
            rearrange(
                repeat(nodes[0], "b n c -> b n k c", k=n_neighbors),
                "b n k c -> b (n k) c",
            ),
            rearrange(
                repeat(nodes[1], "b n c d -> b n k c d", k=n_neighbors),
                "b n k c d -> b (n k) c d",
            ),
        )
    return rearrange(
        repeat(nodes, "b n c -> b n k c", k=n_neighbors), "b n k c -> b (n k) c"
    )


def tuple_sum(*args):
    """
    Sums any number of tuples (s, V) elementwise.
    """
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def tuple_node_mask(x, mask):
    """
    Put to 0 the elements of x ((s, V) tuple) that are masked in the mask

    :param x: (s, V) tuple of nodes
    :param mask: boolean `torch.Tensor` with zeros at the positions that are masked
    """
    if type(x) == tuple:
        s, v = x
        s[~mask] = torch.zeros_like(s[~mask]).to(s.device)
        v[~mask] = torch.zeros_like(v[~mask]).to(v.device)
        return s, v

    x[~mask] = torch.zeros_like(s[~mask]).to(x.device)
    return x


def tuple_edge_mask(x, mask, idx):
    mask_expanded_ = repeat(mask, "b n1 -> b n2 n1", n2=mask.shape[1])
    mask_expanded = mask_expanded_.detach().clone()  # to avoid Warning
    mask_expanded[~mask] = (
        torch.zeros_like(mask_expanded[~mask]).bool().to(mask_expanded.device)
    )
    mask_expanded = rearrange(mask_expanded, "b n1 n2 -> (b n1) n2")
    mask_expanded = rearrange(
        batch_select(mask_expanded, rearrange(idx, "b n k -> (b n) k")),
        "(b n) k -> b n k",
        b=mask.shape[0],
    )

    if type(x) == tuple:
        s, v = x
        s[~mask_expanded] = torch.zeros_like(s[~mask_expanded]).to(s.device)
        v[~mask_expanded] = torch.zeros_like(v[~mask_expanded]).to(v.device)
        return s, v

    x[~mask_expanded] = torch.zeros_like(x[~mask_expanded]).to(x.device)
    return x


def randn(n, dims, device="cpu"):
    """
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    """
    return torch.randn(n, dims[0], device=device), torch.randn(
        n, dims[1], 3, device=device
    )


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    v = torch.reshape(x[..., -3 * nv :], x.shape[:-1] + (nv, 3))
    s = x[..., : -3 * nv]
    return s, v


def _merge(s, v):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate)
            * torch.ones(x.shape[:-1], device=self.dummy_param.device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims, n_dims=None, norm_divide=False):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.norm_divide = norm_divide
        if self.v:
            # self.gamma = nn.Parameter(torch.ones(shape))
            self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        if self.norm_divide:
            vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
            vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
            v = v / vn
        return self.scalar_norm(s), self.gamma * v


class GVP2(nn.Module):
    """
    Geometric Vector Perceptron returning both edges and nodes tensors.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_node_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    :param node_and_edges: if `True`, returns both edge and node outputs. Otherwise, return only edge outputs (default `True`).
    """

    def __init__(
        self,
        in_dims,
        out_node_dims,
        out_edge_dims,
        n_neighbors,
        activations=(F.gelu, torch.sigmoid),
        vector_gate=True,
        node_and_edges=True,
        drop_rate=0.1,
        use_pna=False,
    ):
        super(GVP2, self).__init__()

        self.si, self.vi = in_dims
        self.sno, self.vno = out_node_dims
        self.seo, self.veo = out_edge_dims
        self.n_neighbors = n_neighbors
        self.vector_gate = vector_gate
        if not node_and_edges:
            self.sno, self.vno = 0, 0

        if self.vi:
            self.h_dim = max(self.vi, self.vno, self.veo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.sno + self.seo)
            if self.vno + self.veo:
                self.wv = nn.Linear(self.h_dim, self.vno + self.veo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.sno + self.seo, self.vno + self.veo)
        else:
            self.ws = nn.Linear(self.si, self.sno + self.seo)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.scalar_dropout = nn.Dropout(drop_rate)
        self.vector_dropout = _VDropout(drop_rate)

        self.use_pna = use_pna

    def forward(self, x, mask):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if in_dims[1] is 0), a single `torch.Tensor`
        :return: tuple (tuple (s, V), tuple (s, V)) of `torch.Tensor`,
                 or (if out_node_dims[1] is 0), a single `torch.Tensor` instead of the first tuple
                 or (if out_edge_dims[1] is 0), a single `torch.Tensor` instead of the second tuple
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn_ = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn_], -1))
            if self.sno:
                sn, se = s[:, :, : self.sno], s[:, :, self.sno :]
            else:
                se = s

            if self.vno + self.veo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    # v = v * torch.sigmoid(gate).unsqueeze(-1)
                    v = v * F.softmax(gate, dim=-1).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))

                if self.vno and self.veo:
                    vn, ve = v[:, :, : self.vno], v[:, :, self.vno :]
                elif self.vno:
                    vn = v
                else:
                    ve = v

        else:
            s = self.ws(x)
            if self.sno:
                sn, se = s[:, :, : self.sno], s[:, :, self.sno :]
            else:
                se = s
            if self.vno:
                vn = torch.zeros(
                    sn.shape[0], self.vno, 3, device=self.dummy_param.device
                )
            if self.veo:
                ve = torch.zeros(
                    se.shape[0], self.veo, 3, device=self.dummy_param.device
                )
        if s.shape[1] < self.n_neighbors**2:
            n_neighbors = int(sqrt(s.shape[1]))
        else:
            n_neighbors = self.n_neighbors

        if self.scalar_act:
            if self.sno:
                sn = self.scalar_act(sn)
            se = self.scalar_act(se)

        if self.sno:
            sn = rearrange(
                self.scalar_dropout(sn), "b (n k) c -> b n k c", k=n_neighbors
            )
            if self.use_pna:
                sn = pna_aggregate(sn, mask)
            else:
                sn = sn.sum(dim=2) / (rearrange(mask, "... -> ... 1").sum(dim=2) + 1e-7)
        if self.vno:
            vn = rearrange(
                self.vector_dropout(vn), "b (n k) c d -> b n k c d", k=n_neighbors
            )
            vn = vn.sum(dim=2) / (rearrange(mask, "... -> ... 1 1").sum(dim=2) + 1e-7)
        if self.sno:
            out_n = (sn, vn) if self.vno else sn
        out_e = (se, ve) if self.veo else se

        if self.sno:
            return out_n, out_e
        return out_e


class GVPLayer(nn.Module):
    def __init__(
        self,
        in_node_dims,
        out_node_dims,
        in_edge_dims,
        out_edge_dims,
        n_neighbors,
        n_layers=1,
        activations=(F.gelu, torch.sigmoid),
        vector_gate=True,
        layer_norm=True,
        drop_rate=0.1,
        norm_divide=False,
        update_edges=True,
        use_pna=False,
        graph_context_dim=0,
        less_dropout=False,
        use_node_dropout=False,
    ):
        super(GVPLayer, self).__init__()

        self.sni, self.vni = in_node_dims
        self.sno, self.vno = out_node_dims
        self.sei, self.vei = in_edge_dims
        self.seo, self.veo = out_edge_dims
        self.n_neighbors = n_neighbors
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.update_edges = update_edges
        self.use_pna = use_pna
        self.less_dropout = less_dropout
        self.use_node_dropout = use_node_dropout
        in_dims = (2 * self.sni + self.sei, 2 * self.vni + self.vei)
        if graph_context_dim > 0:
            in_dims = (in_dims[0] + graph_context_dim, in_dims[1])

        if self.n_layers == 1:
            gvps = [GVP2(
                in_dims,
                out_node_dims,
                out_edge_dims,
                n_neighbors,
                activations=activations,
                vector_gate=vector_gate,
                drop_rate=0 if self.less_dropout else drop_rate,
                use_pna=self.use_pna,
            )]
        else:
            gvps = [
                GVP2(
                    in_dims,
                    (0, 0),
                    in_dims,
                    n_neighbors,
                    activations=activations,
                    vector_gate=vector_gate,
                    node_and_edges=False,
                    drop_rate=0 if self.less_dropout else drop_rate,
                    use_pna=self.use_pna,
                )
            ]
            for _ in range(1, self.n_layers - 1):
                gvps.append(
                    GVP2(
                        in_dims,
                        (0, 0),
                        in_dims,
                        n_neighbors,
                        activations=activations,
                        vector_gate=vector_gate,
                        node_and_edges=False,
                        drop_rate=0 if self.less_dropout else drop_rate,
                        use_pna=self.use_pna,
                    )
                )
            gvps.append(
                GVP2(
                    in_dims,
                    out_node_dims,
                    out_edge_dims,
                    n_neighbors,
                    activations=activations,
                    vector_gate=vector_gate,
                    drop_rate=0 if self.less_dropout else drop_rate,
                    use_pna=self.use_pna,
                )
            )
        self.gvp = nn.ModuleList(gvps)

        aggr_factor = 3 if self.use_pna else 1

        self.wns = nn.Linear(self.sni + self.sno * aggr_factor, self.sno)
        self.wes = nn.Linear(self.sei + self.seo, self.seo)
        if self.vno:
            self.wnv = nn.Linear(self.vni + self.vno, self.vno, bias=False)
        if self.veo:
            self.wev = nn.Linear(self.vei + self.veo, self.veo, bias=False)

        if self.layer_norm:
            self.node_norm = LayerNorm(
                (self.sno, self.vno), n_dims=4, norm_divide=norm_divide
            )  # nn.LayerNorm(self.sno)
            self.edge_norm = LayerNorm(
                (self.seo, self.veo), n_dims=5, norm_divide=norm_divide
            )  # nn.LayerNorm(self.seo)

        self.node_dropout = Dropout(drop_rate)
        self.edge_dropout = Dropout(drop_rate)

    def combine(self, nodes, edges, indexes, graph_context):
        indexes_flat = rearrange(indexes, "b n k -> b (n k)")
        nodes_select_out = tuple_batch_select(nodes, indexes_flat)
        nodes_select_in = tuple_node_expand(
            nodes, min(self.n_neighbors, nodes[0].shape[1])
        )
        edges_flat = tuple_nodify(edges)
        if graph_context is not None:
            in_scalar = torch.cat([nodes_select_in[0], repeat(graph_context, "b d -> b l d", l=nodes_select_in[0].shape[1])], dim=-1)
            nodes_select_in = (in_scalar, nodes_select_in[1])

        if self.vni and self.vei:
            input = tuple_cat(nodes_select_in, edges_flat, nodes_select_out)
        elif self.vni:
            scalar_list = [nodes_select_in[0], nodes_select_out[0]]
            vector_list = [nodes_select_in[1], nodes_select_out[1]]
            if isinstance(edges_flat, tuple):
                scalar_list.append(edges_flat[0])
                vector_list.append(edges_flat[1])
            else:
                scalar_list.append(edges_flat)
            input = (
                torch.cat(scalar_list, dim=-1),
                torch.cat(vector_list, dim=-2),
            )
        elif self.vei:
            input = (
                torch.cat([nodes_select_in, edges_flat[0], nodes_select_out], dim=-1),
                edges_flat[1],
            )
        else:
            input = torch.cat([nodes_select_in, edges_flat, nodes_select_out], dim=-1)
        return input

    def forward(self, nodes, edges, indexes, mask=None, global_tokens=None, graph_context=None):
        input = self.combine(nodes, edges, indexes, graph_context)

        if mask is not None:
            mask_edge = gather_nodes(mask.unsqueeze(-1), indexes).squeeze(-1)
        else:
            mask_edge = None

        input_ = (input[0].clone(), input[1].clone())
        for gvp in self.gvp:
            out_nodes, out_edges = gvp(input_, mask_edge)
            input_ = self.combine(nodes, edges, indexes, graph_context)
        out_edges = tuple_edgify(out_edges, self.n_neighbors)
        if self.use_node_dropout:
            out_nodes = self.node_dropout(out_nodes)
        out_edges = self.edge_dropout(out_edges)

        if self.update_edges:
            s_ni = nodes[0] if self.vni else nodes
            s_no = out_nodes[0] if self.vno else out_nodes
            s_ei = edges[0] if self.vei else edges
            s_eo = out_edges[0] if self.veo else out_edges

            no = F.gelu(self.wns(torch.cat([s_ni, s_no], dim=-1)))
            eo = F.gelu(self.wes(torch.cat([s_ei, s_eo], dim=-1)))
            # if self.layer_norm:
            #    no, eo = self.node_norm(no), self.edge_norm(eo)

            if self.vno:
                v_no = torch.transpose(
                    self.wnv(
                        torch.transpose(
                            torch.cat([nodes[1], out_nodes[1]], dim=-2), -1, -2
                        )
                    ),
                    -1,
                    -2,
                )
                no = (no, v_no)

            if self.veo:
                v_eo = torch.transpose(
                    self.wev(
                        torch.transpose(
                            torch.cat([edges[1], out_edges[1]], dim=-2), -1, -2
                        )
                    ),
                    -1,
                    -2,
                )
                eo = (eo, v_eo)

        else:
            no = out_nodes
            eo = out_edges

        if self.layer_norm:
            no, eo = self.node_norm(no), self.edge_norm(eo)

        if mask is not None:
            no = tuple_node_mask(no, mask)
            eo = tuple_edge_mask(eo, mask, indexes)

        return no, eo, global_tokens


class GVPNet(nn.Module):
    def __init__(
        self,
        node_dims,
        edge_dims,
        n_neighbors,
        n_aggr_layers=3,
        n_mpnn_layers=1,
        drop_rate=0.1,
        activations=(F.relu, torch.sigmoid),
        vector_gate=True,
        norm_divide=True,
        update_edges=True,
        use_pna=False,
        use_attention=False,
        return_edge_features=True,
        vector_angles=False,
        linear_layers_num=0,
        edge_compute_func=None,
        use_edge_vectors=False,
        predict_oxygens=False,
        graph_context_dim=0,
        use_node_dropout=False,
        less_dropout=False,
    ):
        super(GVPNet, self).__init__()

        self.n_aggr_layers = n_aggr_layers
        self.linear_layers_num = linear_layers_num
        self.no_linear = False
        self.edge_compute_func = edge_compute_func
        self.use_edge_vectors = use_edge_vectors

        if type(node_dims) == list:
            assert (
                len(node_dims) == n_aggr_layers + 1
            ), f"If you provide a list of dimensions for the node features and vectors, the list must have exactly n_aggr_layers + 1 elements. Found {len(node_dims)} elements and {n_aggr_layers} aggregation layers."
        else:
            node_dims = [node_dims for _ in range(n_aggr_layers + 1)]

        if type(edge_dims) == list:
            assert (
                len(edge_dims) == n_aggr_layers + 1
            ), f"If you provide a list of dimensions for the edge features and vectors, the list must have exactly n_aggr_layers + 1 elements. Found {len(edge_dims)} elements and {n_aggr_layers} aggregation layers."
        else:
            edge_dims = [edge_dims for _ in range(n_aggr_layers + 1)]

        if not return_edge_features and linear_layers_num == 0:
            edge_dims[-1] = (0, 0)
        if vector_angles and not return_edge_features:
            node_dims[-1] = (node_dims[-1][0], 2)
        gvps = []
        for k in range(n_aggr_layers):
            gvps.append(
                GVPLayer(
                    node_dims[k],
                    node_dims[k + 1],
                    edge_dims[k],
                    edge_dims[k + 1],
                    n_neighbors,
                    n_layers=n_mpnn_layers,
                    activations=activations,
                    vector_gate=vector_gate,
                    layer_norm=True,
                    drop_rate=drop_rate,
                    norm_divide=norm_divide,
                    update_edges=update_edges,
                    use_pna=use_pna,
                    graph_context_dim=graph_context_dim,
                    use_node_dropout=use_node_dropout,
                    less_dropout=less_dropout,
                )
            )
        self.gvps = nn.ModuleList(gvps)
        linear = []
        for k in range(linear_layers_num):
            linear.append(
                GVPLayer(
                    node_dims[-1],
                    node_dims[-1],
                    (node_dims[0][0], edge_dims[-1][-1]),
                    (node_dims[0][0], edge_dims[-1][-1]),
                    3,
                    n_layers=n_mpnn_layers,
                    activations=activations,
                    vector_gate=vector_gate,
                    layer_norm=True,
                    drop_rate=drop_rate,
                    norm_divide=norm_divide,
                    update_edges=update_edges,
                    use_pna=use_pna,
                    graph_context_dim=graph_context_dim,
                    use_node_dropout=use_node_dropout,
                    less_dropout=less_dropout,
                )
            )
        self.linear = nn.ModuleList(linear)

    def forward(
        self, nodes, edges, indexes, coords, residue_idx, chain_labels, mask=None, graph_context=None,
    ):
        nodes_out, edges_out = nodes, edges
        global_tokens = None
        for gvp in self.gvps:
            nodes_out, edges_out, global_tokens = gvp(
                nodes_out, edges_out, indexes, mask=mask, global_tokens=global_tokens, graph_context=graph_context,
            )
        # recompute edges_out and indexes
        if len(self.linear) > 0 and not self.no_linear:
            edges_out, indexes, *_ = self.edge_compute_func(
                coords, mask.to(torch.int), residue_idx, chain_labels, linear=True
            )
            _, edge_vectors = get_vectors(
                coords, mask, indexes, edge=self.use_edge_vectors
            )
            edges_out = (edges_out, edge_vectors)
            for gvp in self.linear:
                nodes_out, edges_out, global_tokens = gvp(
                    nodes_out, edges_out, indexes, mask=mask, global_tokens=global_tokens, graph_context=graph_context,
                )
        return nodes_out, edges_out


class GVPe_Encoder(Encoder):
    def __init__(self, args) -> None:
        super().__init__()
        self.return_X = False
        self.use_edge_vectors = args.use_edge_vectors
        self.pass_edge_vectors = args.pass_edge_vectors
        if self.return_X:
            vector_dim = 1
        else:
            vector_dim = args.vector_dim

        node_dims = (
            [(args.hidden_dim, args.vector_dim)]
            + [(args.hidden_dim, args.vector_dim) for _ in range(args.num_encoder_layers - 1)]
            + [(args.hidden_dim, vector_dim)]
        )

        self.encoder = GVPNet(
            node_dims=node_dims,
            edge_dims=(args.hidden_dim, 0 if not args.use_edge_vectors else 1),
            n_neighbors=args.num_neighbors,
            n_aggr_layers=args.num_encoder_layers,
            n_mpnn_layers=args.num_encoder_mpnn_layers
            if args.num_encoder_mpnn_layers is not None
            else 1,
            drop_rate=args.dropout,
            norm_divide=False,
            update_edges=args.update_edges,
            use_pna=args.use_pna_in_encoder,
            use_attention=args.use_attention_in_encoder,
            use_edge_vectors=args.use_edge_vectors,
            graph_context_dim=16 if args.use_graph_context else 0,
            use_node_dropout=args.use_node_dropout,
            less_dropout=args.less_dropout,
        )

    def forward(
        self,
        h_V,
        h_E,
        E_idx,
        mask,
        X,
        residue_idx,
        chain_encoding_all,
        global_context,
        coords,
    ):
        vectors, edge_vectors = get_vectors(X, mask, E_idx, edge=self.use_edge_vectors)
        if self.use_edge_vectors:
            edge = (h_E, edge_vectors)
        else:
            edge = h_E
        (h_V, upd), h_E = self.encoder(
            (h_V, vectors),
            edge,
            E_idx,
            coords,
            residue_idx,
            chain_encoding_all,
            mask=mask.bool(),
            graph_context=global_context,
        )
        if self.return_X:
            X = X + upd
        else:
            X = upd
        if self.use_edge_vectors:
            if self.return_X or not self.pass_edge_vectors:
                h_E = h_E[0]
        return (
            h_V,
            h_E,
            X,
            E_idx,
            coords,
        )


class GVPe_Decoder(Decoder):
    def __init__(self, args) -> None:
        super().__init__()
        self.use_edge_vectors = args.use_edge_vectors
        self.pass_edge_vectors = args.pass_edge_vectors
        node_dims = [
            (args.hidden_dim, args.vector_dim) for _ in range(args.num_decoder_layers - 1)
        ] + [(args.hidden_dim, 1)]
        node_dims = [(args.hidden_dim, args.vector_dim)] + node_dims

        self.accept_X = False

        self.decoder = GVPNet(
            node_dims=node_dims,
            edge_dims=(args.in_dim, 0 if not args.use_edge_vectors else 1),
            n_neighbors=args.num_neighbors,
            n_aggr_layers=args.num_decoder_layers,
            n_mpnn_layers=args.num_decoder_mpnn_layers,
            drop_rate=args.dropout,
            norm_divide=False,
            update_edges=args.update_edges,
            use_pna=args.use_pna_in_decoder,
            use_attention=args.use_attention_in_decoder,
            return_edge_features=args.keep_edge_model,
            vector_angles=args.vector_angles,
            linear_layers_num=args.linear_layers_num,
            edge_compute_func=args.edge_compute_func
            if args.linear_layers_num > 0
            else None,
            use_edge_vectors=args.use_edge_vectors,
            predict_oxygens=args.predict_oxygens,
            graph_context_dim=16 if args.use_graph_context else 0,
            use_node_dropout=args.use_node_dropout,
            less_dropout=args.less_dropout,
        )

    def forward(
        self,
        h_V,
        h_E,
        E_idx,
        mask,
        X,
        residue_idx,
        chain_encoding_all,
        global_context,
        coords,
    ):
        if self.accept_X:
            vectors, edge_vectors = get_vectors(
                X, mask, E_idx, edge=self.use_edge_vectors
            )
        else:
            vectors = X
            if not self.pass_edge_vectors:
                _, edge_vectors = get_vectors(
                    coords, mask, E_idx, edge=self.use_edge_vectors
                )
        if self.use_edge_vectors and not self.pass_edge_vectors:
            edge = (h_E, edge_vectors)
        else:
            edge = h_E
        (h_V, upd), h_E = self.decoder(
            (h_V, vectors),
            edge,
            E_idx,
            coords,
            residue_idx,
            chain_encoding_all,
            mask=mask.bool(),
            graph_context=global_context,
        )
        if self.use_edge_vectors:
            h_E = h_E[0]
        return (
            h_V,
            h_E,
            upd,
            E_idx,
            coords,
        )
