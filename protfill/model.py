from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import esm
from einops import repeat, rearrange
from copy import deepcopy
import os
from proteinflow.data import ProteinEntry

from protfill.layers.gvp import GVPOrig_Decoder, GVPOrig_Encoder
from protfill.layers.gvp_new import GVP_Decoder, GVP_Encoder
from protfill.utils.model_utils import *
from protfill.diffusion import Diffuser, get_orientations
from torch.utils.checkpoint import checkpoint


def combine_decoders(coords_decoder, seq_decoder, predict_angles):
    class CombinedDecoder(nn.Module):
        def __init__(self):
            super(CombinedDecoder, self).__init__()
            self.seq_decoder = seq_decoder
            self.coords_decoder = coords_decoder
            self.predict_angles = predict_angles

        def forward(self, *args):
            h_V, *_ = self.seq_decoder(*args)
            angles, h_E, vectors, E_idx, coords = self.coords_decoder(*args)
            if self.predict_angles:
                h_V = (h_V, angles)
            return h_V, h_E, vectors, E_idx, coords

    return CombinedDecoder()


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        force_neighbor_edges=False,
        no_oxygen=False,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.no_oxygen = no_oxygen

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        num_dist = 16 if no_oxygen else 25
        edge_in = num_positional_embeddings + num_rbf * num_dist
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.timestep = None
        self.force_neighbor_edges = force_neighbor_edges

    def _dist(self, X, mask, eps=1e-6, exclude_neighbors=False):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        if exclude_neighbors:
            D_adjust[
                :, range(1, D_adjust.shape[1]), range(D_adjust.shape[1] - 1)
            ] = 1000
            D_adjust[
                :, range(D_adjust.shape[1] - 1), range(1, D_adjust.shape[1])
            ] = 1000
        D_neighbors, E_idx = torch.topk(
            D_adjust, int(np.minimum(self.top_k, X.shape[1])), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        if len(D.shape) == 4:
            D_mu = D_mu.view([1, 1, 1, -1])
        else:
            D_mu = D_mu.view([1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _dihedral_angle(self, crd, msk):
        p0 = crd[..., 0, :]
        p1 = crd[..., 1, :]
        p2 = crd[..., 2, :]
        p3 = crd[..., 3, :]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= torch.unsqueeze(torch.norm(b1, dim=-1), -1) + 1e-7

        v = b0 - torch.unsqueeze(torch.sum(b0 * b1, dim=-1), -1) * b1
        w = b2 - torch.unsqueeze(torch.sum(b2 * b1, dim=-1), -1) * b1

        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1, v) * w, dim=-1)
        dh = torch.rad2deg(torch.atan2(y, x))
        return dh

    def _dihedral(self, crd, msk):
        angles = []
        # N, C, Ca, O
        # psi
        p = crd[:, :-1, [0, 2, 1], :]
        p = torch.cat([p, crd[:, 1:, [0], :]], 2)
        p = F.pad(p, (0, 0, 0, 0, 0, 1))
        angles.append(self._dihedral_angle(p, msk))
        # phi
        p = crd[:, :-1, [1], :]
        p = torch.cat([p, crd[:, 1:, [0, 2, 1]]], 2)
        p = F.pad(p, (0, 0, 0, 0, 1, 0))
        angles.append(self._dihedral_angle(p, msk))
        angles = torch.stack(angles, -1)
        angles = torch.cat([torch.sin(angles), torch.cos(angles)], -1)
        return angles

    def forward(
        self,
        X,
        mask,
        residue_idx,
        chain_labels,
        timestep=None,
        feature_types=None,
        linear=False,
    ):
        if feature_types is None:
            feature_types = []
        if timestep is not None:
            self.timestep = timestep
        b = X[:, :, 2, :] - X[:, :, 0, :]
        c = X[:, :, 1, :] - X[:, :, 2, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 2, :]
        C = X[:, :, 1, :]
        N = X[:, :, 0, :]
        Ca = X[:, :, 2, :]
        O = X[:, :, 3, :]

        if linear:
            prev_ind = torch.tensor([0] + list(range(X.shape[1] - 1)), device=X.device)
            cur_ind = torch.tensor(list(range(X.shape[1])), device=X.device)
            next_ind = torch.tensor(
                list(range(1, X.shape[1])) + [X.shape[1] - 1], device=X.device
            )
            E_idx = repeat(
                torch.stack([prev_ind, cur_ind, next_ind], -1),
                "n k -> b n k",
                b=X.shape[0],
            )
            x1 = rearrange(C, "b n d -> b n 1 d")
            x2 = gather_nodes(C, E_idx)
            D_neighbors = torch.norm(x1 - x2, dim=-1)
        else:
            D_neighbors, E_idx = self._dist(
                C, mask, exclude_neighbors=self.force_neighbor_edges
            )
            if self.force_neighbor_edges:
                prev_ind = torch.tensor(
                    [0] + list(range(X.shape[1] - 1)), device=X.device
                )
                next_ind = torch.tensor(
                    list(range(1, X.shape[1])) + [X.shape[1] - 1], device=X.device
                )
                E_idx_ = repeat(
                    torch.stack([prev_ind, next_ind], -1),
                    "n k -> b n k",
                    b=X.shape[0],
                )
                x1 = rearrange(C, "b n d -> b n 1 d")
                x2 = gather_nodes(C, E_idx_)
                D_neighbors_ = torch.norm(x1 - x2, dim=-1)
                D_neighbors[:, 1:-1, -2:] = D_neighbors_[:, 1:-1, :]
                E_idx[:, 1:-1, -2:] = E_idx_[:, 1:-1, :]
                D_neighbors[:, 0, -1] = D_neighbors_[:, 0, 1]
                E_idx[:, 0, -1] = E_idx_[:, 0, 1]
                D_neighbors[:, -1, -1] = D_neighbors_[:, -1, 0]
                E_idx[:, -1, -1] = E_idx_[:, -1, 0]

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # C-C
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(C, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # C-Cb
        if not self.no_oxygen:
            RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
            RBF_all.append(self._get_rbf(C, O, E_idx))  # Ca-O
            RBF_all.append(self._get_rbf(O, C, E_idx))  # O-Ca
            RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
            RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
            RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
            RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-C
            RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
            RBF_all.append(self._get_rbf(Ca, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)  # + 24 or 16

        if self.timestep is not None:
            timestep_embedding = self._rbf(
                repeat(self.timestep.to(X.device), "b -> b 1")
            ).squeeze(1)
        else:
            timestep_embedding = None

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        vectors = []
        if "backbone_orientation" in feature_types:
            orientation = torch.zeros((X.shape[0], X.shape[1], 2, 3)).to(X.device)
            orientation[:, 1:, 0] = X[:, 1:, 2] - X[:, :-1, 2]
            orientation[:, :-1, 1] = X[:, :-1, 2] - X[:, 1:, 2]
            orientation /= torch.norm(orientation, dim=-1, keepdim=True) + 1e-7
            orientation *= mask[..., None, None]
            orientation[:, 1:, 0] *= mask[:, :-1, None]
            orientation[:, :-1, 1] *= mask[:, 1:, None]
            vectors.append(orientation)
        if "c_beta" in feature_types:
            diff = (Cb - Ca).unsqueeze(2)
            vectors.append((diff) / (torch.norm(diff, dim=-1, keepdim=True) + 1e-7))
        if len(vectors) > 0:
            vectors = torch.cat(vectors, dim=2)
        else:
            vectors = None
        scalars = []
        if "dihedral_recomputed" in feature_types:
            scalars.append(self._dihedral(X, mask))
        if len(scalars) > 0:
            scalars = torch.cat(scalars, dim=2)
        else:
            scalars = None
        return E, E_idx, vectors, scalars, timestep_embedding


class ProtFill(nn.Module):
    def __init__(
        self,
        args,
        encoder_type,
        decoder_type,
        hidden_dim=128,
        vocab=21,
        k_neighbors=32,
        augment_eps=0.2,
        noise_std=None,
        embedding_dim=128,
        ignore_unknown=False,
        node_features_type="zeros",
        n_cycles: int = 1,
        no_sequence_in_encoder: bool = False,
        double_sequence_features: bool = False,
        seq_init_mode: str = "zeros",
        predict_structure: bool = False,
        use_global_feats_attention_dim: bool = False,
        separate_modules_num: int = 1,
        predict_node_vectors: str = "no_prediction",  # ["only_vectors", "with_others", "no_prediction"]
        predict_angles: bool = False,
        co_design: str = "none",
    ):
        super(ProtFill, self).__init__()
        encoders = {
            "gvp": GVP_Encoder,
            "gvp_orig": GVPOrig_Encoder,
        }
        decoders = {
            "gvp": GVP_Decoder,
            "gvp_orig": GVPOrig_Decoder,
        }
        auto_decoders = ["mpnn_auto"]
        force_update_decoders = ["mpnn", "mpnn_enc"]
        vector_update_decoders = ["gvp", "gvp_orig"]
        self.vector_encoders = ["gvp", "gvp_orig"]
        self.vector_decoders = ["gvp", "gvp_orig"]
        # self.vector_encoders = []
        # self.vector_decoders = []
        num_letters = 20 if ignore_unknown else 21

        self.diffusion = (
            Diffuser(
                num_tokens=num_letters,
                num_steps=args.num_diffusion_steps,
                schedule_name=args.variance_schedule,
                seq_diffusion_type=args.seq_diffusion_type,
                recover_x0=True,
                linear_interpolation=False,
                diff_predict=args.diffusion_parameterization,
                weighted_diff_loss=args.weighted_diff_loss,
                pos_std=args.noise_std,
                noise_around_interpolation=args.noise_around_interpolation,
                no_added_noise=args.no_added_diff_noise,
                cosine_cutoff=0.99,
            )
            if args.diffusion
            else None
        )
        self.diffusion_ = Diffuser(
            num_tokens=num_letters,
            num_steps=args.num_diffusion_steps,
            schedule_name=args.variance_schedule,
            seq_diffusion_type=args.seq_diffusion_type,
            linear_interpolation=False,
            pos_std=args.noise_std,
        )
        self.num_diffusion_steps = args.num_diffusion_steps
        self.num_letters = num_letters
        self.predict_structure = predict_structure
        self.predict_sequence = not predict_structure
        self.predict_node_vectors = predict_node_vectors != "no_prediction"
        if predict_node_vectors == "only_vectors":
            self.predict_structure = False
            self.predict_sequence = False
        if co_design != "none":
            self.predict_structure = True
            self.predict_sequence = True

        if self.predict_structure and decoder_type in auto_decoders:
            raise NotImplementedError(
                "Predicting structure with AR decoders is not implemented!"
            )
        if self.predict_structure and decoder_type in force_update_decoders:
            raise NotImplementedError(
                f"Predicting structure with {decoder_type} decoder is not implemented!"
            )

        if co_design == "seq":
            n_cycles *= 2
            separate_modules_num *= 2

        if node_features_type is None:
            node_features_type = ""
        self.num_letters = num_letters
        self.node_features_type = node_features_type
        self.augment_eps = augment_eps
        self.noise_std = noise_std
        self.n_cycles = n_cycles
        self.no_sequence_in_encoder = no_sequence_in_encoder
        self.seq_init_mode = seq_init_mode
        self.use_global_feats_attention_dim = use_global_feats_attention_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_checkpointing = args.use_checkpointing
        self.predict_angles = predict_angles
        self.co_design = co_design
        self.pass_features = not args.not_pass_features
        self.quaternion = args.quaternion
        self.scale_timestep = args.scale_timestep
        self.reset_masked = args.reset_masked
        self.diff_predict = args.diffusion_parameterization
        self.variance_scale = args.variance_scale
        self.use_graph_context = args.use_graph_context
        self.alternative_noising = args.alternative_noising
        self.no_oxygen = args.no_oxygen

        self.one_shot_decoder = decoder_type not in auto_decoders
        self.use_sequence_in_encoder = (
            self.one_shot_decoder if not no_sequence_in_encoder else False
        )
        self.add_sequence_in_decoder = (
            not self.use_sequence_in_encoder or double_sequence_features
        )
        self.probability_mode = self.seq_init_mode in [
            "esm_probabilities",
            "uniform_probabilities",
        ]
        self.force_update = (
            decoder_type in force_update_decoders
        ) and predict_structure
        self.vector_update = decoder_type in vector_update_decoders

        if separate_modules_num > n_cycles:
            separate_modules_num = n_cycles

        self.str_features = False
        self.seq_features = False

        self.features = ProteinFeatures(
            hidden_dim,
            top_k=k_neighbors,
            force_neighbor_edges=args.force_neighbor_edges,
            no_oxygen=args.no_oxygen,
        )
        args.edge_compute_func = self.features

        n_vectors = {"sidechain_orientation": 1, "backbone_orientation": 2, "c_beta": 1}
        args.vector_dim = 6
        if (
            self.encoder_type in self.vector_encoders
            or self.decoder_type in self.vector_decoders
        ):
            args.vector_dim += sum(
                [n_vectors[x] for x in node_features_type.split("+") if x in n_vectors]
            )
        args.norm_divide = self.predict_sequence

        add_dim = 0
        if self.diffusion and not self.use_graph_context:
            add_dim += 16

        self.W_e = nn.Linear(hidden_dim + add_dim, hidden_dim, bias=True)
        self.W_s = (
            nn.Sequential(
                nn.Linear(vocab, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            if self.probability_mode
            else nn.Embedding(vocab, embedding_dim)
        )
        if node_features_type is not None:
            d_structure = {
                "dihedral": 4,
                "topological": 9,
                "mask": 1,
                "esm": 320,
                "secondary_structure": 3,
            }
            d_sequence = {"chemical": 6, "chem_topological": 105}
            input_f_structure = sum(
                [
                    d_structure[x]
                    for x in node_features_type.split("+")
                    if x in d_structure
                ]
            )
            input_f_seq = sum(
                [
                    d_sequence[x]
                    for x in node_features_type.split("+")
                    if x in d_sequence
                ]
            )
            if input_f_structure > 0:
                self.str_features = True
            if self.use_sequence_in_encoder:
                if input_f_seq > 0:
                    input_f_structure += hidden_dim
                else:
                    input_f_structure += embedding_dim
            if input_f_structure > 0:
                self.W_v_str = nn.Linear(input_f_structure, hidden_dim, bias=True)
            if input_f_seq > 0:
                self.seq_features = True
                self.W_v_seq = nn.Linear(
                    input_f_seq + embedding_dim, hidden_dim, bias=True
                )
        if "esm" in self.node_features_type.split("+") or "esm" in self.seq_init_mode:
            self.esm, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.W_v_esm = nn.Linear(320, hidden_dim)
            tok_to_idx = self.alphabet.tok_to_idx
            idx_to_tok = {tok_to_idx[k]: k for k in tok_to_idx.keys()}
            for k in idx_to_tok.keys():
                if len(idx_to_tok[k]) > 1:
                    idx_to_tok[k] = "X"
            self.idx_esm_to_idx_mpnn = {
                k: REVERSE_ALPHABET_DICT[idx_to_tok[k]]
                if idx_to_tok[k] in REVERSE_ALPHABET_DICT.keys()
                else 0
                for k in idx_to_tok.keys()
            }
            idx_mpnn_to_idx_esm = {
                self.idx_esm_to_idx_mpnn[k]: k
                for k in self.idx_esm_to_idx_mpnn.keys()
                if self.idx_esm_to_idx_mpnn[k] != 0
            }
            self.idx_selector = torch.LongTensor(
                [
                    idx_mpnn_to_idx_esm[k]
                    for k in sorted(list(idx_mpnn_to_idx_esm.keys()))
                ]
            )
            self.idx_for_unknown = torch.LongTensor(
                [
                    k
                    for k in range(len(list(self.idx_esm_to_idx_mpnn.keys())))
                    if k not in self.idx_selector
                ]
            )

        self.separate_modules_num = separate_modules_num
        self.encoders = nn.ModuleList([encoders[encoder_type](args)])
        if separate_modules_num > 1:
            self.encoders += nn.ModuleList(
                [
                    encoders[encoder_type](args)
                    for i in range(separate_modules_num - 1)
                ]
            )

        # Decoder layers
        in_dim = hidden_dim  # edge features
        if self.add_sequence_in_decoder or not self.one_shot_decoder:
            if not self.seq_features:
                in_dim += embedding_dim
            else:
                in_dim += hidden_dim
        args.in_dim = in_dim

        self.decoders = [
            decoders[decoder_type](args)
            for i in range(
                separate_modules_num
                if co_design != "share_enc"
                else separate_modules_num * 2
            )
        ]
        if self.co_design == "share_enc":
            self.decoders = [combine_decoders(
                coords_decoder=self.decoders[c * 2 + 1],  # coords
                seq_decoder=self.decoders[c * 2],  # seq
                predict_angles=self.predict_angles,
            ) for c in range(separate_modules_num)]
        self.decoders = nn.ModuleList(self.decoders)
        if self.predict_sequence:
            self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        if self.force_update:
            force_dim = hidden_dim * 2
            self.W_force = nn.Linear(force_dim, 1)

        self.W_vector = nn.Linear(args.vector_dim, 4)

        self.vector_angles = False
        self.angle_layer = None
        if predict_angles:
            if decoder_type == "gvp" and args.vector_angles:
                self.vector_angles = True
            else:
                self.angle_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, 3 if not args.predict_oxygens else 4),
                )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # self.translation_weight = torch.nn.Parameter(torch.tensor(1.))
        # self.translation_weight.requires_grad = True

    def random_rotation(self, X, chain_labels, lim=torch.pi):
        n = chain_labels.sum()
        angles = torch.randn((n, 3)) * lim - lim / 2
        R, _ = self.rotation_matrices_from_angles(angles)
        R = R.to(X.device)
        mean = X[chain_labels].mean(-2).unsqueeze(-2)
        out = torch.einsum("ndj,naj->nad", R, X[chain_labels] - mean) + mean
        return out.float()

    @staticmethod
    def R_x(alpha, vector=False):
        """
        Computes a tensor of 3D rotation matrices around the x-axis.
        """
        if not vector:
            ca = torch.cos(alpha)
            sa = torch.sin(alpha)
        else:
            ca = alpha[:, :, 0]
            sa = alpha[:, :, 1]
        zeros = torch.zeros_like(ca)
        ones = torch.ones_like(ca)
        return torch.stack(
            (
                torch.cat((ones, zeros, zeros), dim=1),
                torch.cat((zeros, ca, -sa), dim=1),
                torch.cat((zeros, sa, ca), dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def R_y(beta, vector=False):
        """
        Computes a tensor of 3D rotation matrices around the y-axis.
        """
        if not vector:
            cb = torch.cos(beta)
            sb = torch.sin(beta)
        else:
            cb = beta[:, :, 0]
            sb = beta[:, :, 1]
        zeros = torch.zeros_like(cb)
        ones = torch.ones_like(cb)
        return torch.stack(
            (
                torch.cat((cb, zeros, sb), dim=1),
                torch.cat((zeros, ones, zeros), dim=1),
                torch.cat((-sb, zeros, cb), dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def R_z(gamma, vector=False):
        """
        Computes a tensor of 3D rotation matrices around the z-axis.
        """
        if not vector:
            cg = torch.cos(gamma)
            sg = torch.sin(gamma)
        else:
            cg = gamma[..., 0]
            sg = gamma[..., 1]
        zeros = torch.zeros_like(cg)
        ones = torch.ones_like(cg)
        return torch.stack(
            (
                torch.cat((cg, -sg, zeros), dim=1),
                torch.cat((sg, cg, zeros), dim=1),
                torch.cat((zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def rotation_matrices_from_angles(angles, vector=False, oxy_angle=None):
        """
        Computes a tensor of 3D rotation matrices from a tensor of 3D rotation angles.
        """
        # Check if the angles are in the correct shape (B * N, 3) or (B * N, 3, 3) for vector angles
        if (not vector and len(angles.shape) == 3) or (
            vector and len(angles.shape) == 4
        ):
            angles = rearrange(angles, "b n ... -> (b n) ...")
        # Extract the rotation angles for each matrix
        alpha, beta, gamma = torch.chunk(angles, 3, dim=1)

        # Compute the rotation matrices
        R_x = ProtFill.R_x(alpha, vector=vector)
        R_y = ProtFill.R_y(beta, vector=vector)
        R_z = ProtFill.R_z(gamma, vector=vector)

        # Return the composed rotation matrices
        return torch.einsum("nij,njk,nkl->nil", R_z, R_y, R_x), oxy_angle

    @staticmethod
    def rotation_matrices_from_quaternions(quaternions):
        """
        Computes a tensor of 3D rotation matrices from a tensor of 4D quaternions.
        """
        oxy_angle = None
        # Extract the quaternion components
        x, y, z = torch.chunk(quaternions, 3, dim=1)
        norm = 1 / torch.sqrt(x**2 + y**2 + z**2 + 1)
        w, x, y, z = torch.ones_like(x) / norm, x / norm, y / norm, z / norm

        # Compute the rotation matrices
        R = torch.stack(
            (
                torch.cat(
                    (
                        w**2 + x**2 - y**2 - z**2,
                        2 * (x * y - w * z),
                        2 * (x * z + w * y),
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        2 * (x * y + w * z),
                        w**2 - x**2 + y**2 - z**2,
                        2 * (y * z - w * x),
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        2 * (x * z - w * y),
                        2 * (y * z + w * x),
                        w**2 - x**2 - y**2 + z**2,
                    ),
                    dim=1,
                ),
            ),
            dim=1,
        )

        # Return the rotation matrices
        return R, oxy_angle

    def noise_coords(
        self,
        X,
        V_structure,
        vector_node_struct,
        chain_M,
        chain_encoding,
        mask,
        transform=None,
        timestep=None,
    ):
        """
        Add noise to the coordinates (augmentation + masking for structure prediction)
        """

        rotation, translation = None, None

        if self.diffusion:
            X, rotation, translation, _ = self.diffusion.noise_structure(
                X, chain_M, self.predict_angles, timestep
            )
        else:
            chain_M_bool = chain_M.bool()
            if not self.alternative_noising:
                X, *_ = self.diffusion_.noise_structure(
                    X,
                    chain_M,
                    self.predict_angles,
                    (self.num_diffusion_steps)
                    * torch.ones(X.shape[0], dtype=torch.long),
                    inference=True,
                    variance_scale=self.variance_scale,
                )
            else:
                coords_X = X[:, :, :4]
                masked_X = coords_X[chain_M_bool].clone()
                masked_X += self.noise_std * torch.randn_like(
                    masked_X[:, 0, :]
                ).unsqueeze(1)
                coords_X[chain_M_bool] = masked_X
                coords_X[chain_M_bool] = self.random_rotation(coords_X, chain_M_bool)
                X[:, :, :4] = coords_X
            if V_structure is not None:
                V_structure[chain_M_bool] = 0
            if vector_node_struct is not None:
                vector_node_struct[chain_M_bool] = self.random_unit_vectors_like(
                    vector_node_struct[chain_M_bool]
                )
        if not self.no_oxygen:
            X[:, :, :4] = self.rotate_oxygens(X[:, :, :4], chain_encoding, mask)
        if transform is not None:
            X[:, :, :4] = transform(X[:, :, :4])

        return X, V_structure, vector_node_struct, rotation, translation

    def generate_embeddings(self, X, S, mask, E, E_idx, vector_node_struct, V_struct, chain_M):
        if not self.str_features:
            h_V = torch.zeros(
                (E.shape[0], E.shape[1], self.hidden_dim), device=E.device
            )  # node embeddings = zeros
        else:
            h_V = self.W_v_str(V_struct)

        h_E = self.W_e(E)
        return h_V, h_E, E_idx, vector_node_struct

    def find_chains_idx(self, residue_idx):
        diffs = residue_idx[:, 1:] - residue_idx[:, :-1]
        idxs = torch.nonzero(diffs > 1)
        idxs = [
            np.array(
                [w.item() for w in idxs[idxs[:, 0] == k][:, 1]]
                + [len(diffs[k][diffs[k] > 0])]
            )
            for k in range(residue_idx.shape[0])
        ]
        return idxs

    def prepare_seqs_for_esm(self, seqs, idxs):
        seqs = [
            "".join([ALPHABET_DICT[int(s)] for s in seq]) for k, seq in enumerate(seqs)
        ]
        max_len = len(seqs[0])
        for k in range(len(seqs)):
            for idx in idxs[k][-2::-1]:
                seqs[k] = seqs[k][: idx + 1] + "<eos><cls>" + seqs[k][idx + 1 :]
            seqs[k] = seqs[k][: len(seqs[k]) - max_len + idxs[k][-1] + 1] + "<pad>" * (
                max_len - idxs[k][-1] - 1
            )
            seqs[k] = (str(k), seqs[k].replace("X", "<mask>"))
        return seqs

    def retrieve_outputs_from_esm(self, outputs, idxs):
        idxs = [
            [0]
            + [idx + 2 * (k + 1) for k, idx in enumerate(idxs[l][:-1])]
            + [1 + idx + 2 * (k + 1) for k, idx in enumerate(idxs[l][:-1])]
            + [outputs.shape[1] - 1]
            for l in range(len(idxs))
        ]
        idxs = [F.one_hot(torch.LongTensor(idx)).sum(dim=0) for idx in idxs]
        out = [outputs[k, ~idxs[k].bool()] for k in range(len(idxs))]
        min_len = np.min([len(o) for o in out])
        out = torch.stack([out[k][:min_len] for k in range(len(out))])
        return out

    def run_esm(self, S, mask, residue_idx, return_logits=False):
        device = S.device
        self.esm.eval()
        masked_seq = S.detach().clone()
        masked_seq[mask == 0] = 0
        idxs = self.find_chains_idx(residue_idx)
        seqs = self.prepare_seqs_for_esm(masked_seq, idxs)
        _, _, batch_tokens = self.batch_converter(seqs)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = self.esm(batch_tokens, repr_layers=[6], return_contacts=False)

        if return_logits:
            return self.retrieve_outputs_from_esm(results["logits"], idxs)
        else:
            return self.retrieve_outputs_from_esm(results["representations"][6], idxs)

    def esm_probabilities(self, S, mask, residue_idx):
        logits = self.run_esm(S, mask, residue_idx, return_logits=True)
        logits_mpnn = logits[:, :, self.idx_selector]
        logits_unknown = torch.sum(
            logits[:, :, self.idx_for_unknown], dim=-1, keepdim=True
        )
        logits_mpnn = torch.cat([logits_unknown, logits_mpnn], dim=-1)
        return F.softmax(logits_mpnn, dim=-1)

    def esm_one_hot(self, S, mask, residue_idx):
        return self.esm_probabilities(S, mask, residue_idx).argmax(dim=-1)

    def random_unit_vectors_like(self, tensor):
        # rand_vecs = torch.randn_like(tensor)
        # norms = torch.norm(rand_vecs, dim=-1, keepdim=True)
        # return rand_vecs / norms
        return torch.zeros_like(tensor)

    def initialize_sequence(
        self,
        seq,
        V_sequence,
        chain_M,
        mask,
        residue_idx,
        vector_node_seq,
        timestep=None,
    ):
        """
        Initialize the sequence values for the masked regions
        """

        distribution = None
        if self.one_shot_decoder:
            if V_sequence is not None:
                V_sequence[chain_M.bool()] = 0
        if self.diffusion:
            seq, distribution = self.diffusion.noise_sequence(
                seq, chain_M, timestep
            )
        elif self.seq_init_mode == "zeros":
            seq[chain_M.bool()] = 0
        elif self.seq_init_mode == "random":
            seq[chain_M.bool()] = torch.randint_like(
                seq[chain_M.bool()], low=1, high=21
            )
        elif self.seq_init_mode == "esm_one_hot":
            seq[chain_M.bool()] = self.esm_one_hot(
                seq, (1 - chain_M) * mask, residue_idx
            )[chain_M.bool()]
        elif self.seq_init_mode == "esm_probabilities":
            esm_probabilities = self.esm_probabilities(
                seq, (1 - chain_M) * mask, residue_idx
            )
            seq = F.one_hot(seq, num_classes=len(ALPHABET)).float()
            seq[chain_M.bool()] = esm_probabilities[chain_M.bool()]
        elif self.seq_init_mode == "uniform_probabilities":
            seq = F.one_hot(seq, num_classes=len(ALPHABET)).float()
            seq[chain_M.bool()] = 1 / 21
        else:
            raise RuntimeError(
                f"The {self.seq_init_mode} sequence init mode is unrecognized"
            )

        if self.one_shot_decoder and vector_node_seq is not None:
            vector_node_seq[chain_M.bool()] = self.random_unit_vectors_like(
                vector_node_seq[chain_M.bool()]
            )

        return seq, distribution, V_sequence, vector_node_seq

    def extract_features(
        self,
        seq,
        chain_M,
        optional_features,
        mask,
        residue_idx,
        chain_encoding_all,
        X,
        cycle,
        transform=None,
        timestep=None,
        corrupt=True,
    ):
        """
        Extract features from the input sequence and structure
        """

        vector_node_seq = optional_features.get("vector_node_seq")
        vector_node_struct = optional_features.get("vector_node_struct")
        V_sequence = optional_features["scalar_seq"]
        V_structure = optional_features["scalar_struct"]
        rotation_gt, seq_t, translation_gt, distribution = None, None, None, None

        if cycle == 0 and corrupt:
            if isinstance(timestep, int):
                timestep = timestep * torch.ones(X.shape[0], device=X.device)
            seq, distribution, V_sequence, vector_node_seq = self.initialize_sequence(
                seq,
                V_sequence,
                chain_M,
                mask,
                residue_idx,
                vector_node_seq,
                timestep=timestep,
            )
            seq_t = seq.clone()
            (
                X,
                V_structure,
                vector_node_struct,
                rotation_gt,
                translation_gt,
            ) = self.noise_coords(
                X,
                V_structure,
                vector_node_struct,
                chain_M,
                chain_encoding_all,
                mask,
                transform=transform,
                timestep=timestep,
            )
        timestep_factor = 1 if not self.scale_timestep else 25 / self.num_diffusion_steps
        timestep_ = None if timestep is None else timestep * timestep_factor
        E, E_idx, vector_node_struct, V_structure_new, timestep_rbf = self.features(
            X[:, :, :4],
            mask,
            residue_idx,
            chain_encoding_all,
            feature_types=self.node_features_type.split("+"),
            timestep=timestep_,
        )
        if timestep is not None and not self.use_graph_context:
            E = torch.cat([E, repeat(timestep_rbf, "b d -> b l k d", l=E.shape[1], k=E.shape[2])], dim=-1)
        struct_features = []
        if V_structure_new is not None:
            struct_features.append(V_structure_new)
        if V_structure is not None:
            struct_features.append(V_structure)
        if len(struct_features) > 0:
            V_structure = torch.cat(struct_features, dim=-1)

        if (
            self.encoder_type not in self.vector_encoders
            and self.decoder_type not in self.vector_decoders
        ):
            seq_features = []
            if vector_node_seq is not None:
                seq_features.append(rearrange(vector_node_seq, "b n f d -> b n (f d)"))
                vector_node_seq = None
            if V_sequence is not None:
                seq_features.append(V_sequence)
            if len(seq_features) > 0:
                V_sequence = torch.cat(seq_features, dim=-1)
            else:
                V_sequence = None
            struct_features = []
            if vector_node_struct is not None:
                struct_features.append(
                    rearrange(vector_node_struct, "b n f d -> b n (f d)")
                )
                vector_node_struct = None
            if V_structure is not None:
                struct_features.append(V_structure)
            if len(struct_features) > 0:
                V_structure = torch.cat(struct_features, dim=-1)
            else:
                V_structure = None
        h_S = self.W_s(seq)
        if self.seq_features:
            h_S = torch.cat([V_sequence, h_S], -1)
            h_S = self.W_v_seq(h_S)
        if "esm" in self.node_features_type:
            if V_structure is not None:
                V_structure = torch.cat(
                    [V_structure, self.run_esm(seq, mask, residue_idx)], -1
                )
            else:
                V_structure = self.run_esm(seq, mask, residue_idx)

        if self.use_sequence_in_encoder:
            if self.str_features:
                V_structure = torch.cat([V_structure, h_S], -1)
            else:
                V_structure = h_S

        # Prepare node and edge embeddings
        h_V, h_E, E_idx, vector_node_struct = self.generate_embeddings(
            X, seq, mask, E, E_idx, vector_node_struct, V_structure, chain_M
        )

        vectors = [X]
        if vector_node_seq is not None:
            vectors.append(vector_node_seq)
        if vector_node_struct is not None:
            vectors.append(vector_node_struct)
        vectors = torch.cat(vectors, -2)

        return (
            h_V,
            h_E,
            E_idx,
            vectors,
            h_S,
            X,
            translation_gt,
            rotation_gt,
            seq_t,
            distribution,
            timestep_rbf
        )

    def update_coords(self, coords, h_V, h_E, E_idx):
        """
        Update coordinates (for invariant networks)
        """

        h_V1 = gather_nodes(h_V, E_idx)
        h_V2 = h_V.unsqueeze(-2).expand(-1, -1, h_V1.size(-2), -1)
        h = torch.cat([h_V1, h_V2], -1)
        f = self.W_force(h)
        c1 = gather_nodes(coords[:, :, 2, :], E_idx)
        c2 = coords[:, :, 2, :].unsqueeze(-2).expand(-1, -1, c1.size(-2), -1)
        diff = c2 - c1
        f = (f * diff).mean(-2)
        coords[:, :, 2, :] = coords[:, :, 2, :] + f
        return coords

    def construct_coords(
        self, coords_ca, orientations, chain_encoding_all, local_coords, mask
    ):
        """
        Construct full coordinates from CA and orientations
        """

        coords = repeat(coords_ca, "b n d -> b n k d", k=4)
        # basic_frame = torch.tensor(
        #     [
        #         [-0.526,    1.361,  0], # N
        #         [1.525,     0,      0], # C
        #         [0,         0,      0], # CA
        #         [0,         0,      0], # we will add oxygens separately
        #     ]
        # ).to(coords.device)
        coords = coords + torch.einsum(
            "b n i d, b n k d -> b n k i", orientations, local_coords
        )
        coords[~mask.bool()] = 0
        # coords = self.rotate_oxygens(coords, chain_encoding_all, mask)
        return coords

    def rotate(
        self,
        coords,
        angles,
        chain_encoding,
        mask_exist,
        quaternion=False,
        vector=False,
        matrix=False,
    ):
        batch_size = coords.size(0)
        oxy_angle = None
        coords = rearrange(coords, "b n k d -> (b n) k d")

        # center the coordinates on CA
        center = coords[:, 2, :].unsqueeze(1)
        coords = coords - center

        # compute rotation matrices
        if quaternion:
            R, oxy_angle = self.rotation_matrices_from_quaternions(angles)
        elif matrix:
            R = rearrange(angles, "b n ... -> (b n) ...")
        elif vector:
            if self.diffusion:
                R = Diffuser()._so3vec_to_rot(angles)
            else:
                R, oxy_angle = self.rotation_matrices_from_angles(angles, vector=True)
        else:
            R, oxy_angle = self.rotation_matrices_from_angles(angles, vector=False)

        # rotate so that CA-N is the z-axis and (CA-N x CA-C) is the y-axis
        new_oz = coords[:, 0]  # CA-N
        eps = 1e-7  # to avoid numerical issues
        cos = new_oz[:, 2] / (
            (new_oz[:, 1] ** 2 + new_oz[:, 2] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        a = torch.arccos(cos)
        mask = new_oz[:, 1] < 0
        a[mask] = -a[mask]
        R_x = self.R_x(repeat(a, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_x)
        new_oz = coords[:, 0]  # CA-N
        cos = new_oz[:, 2] / (
            (new_oz[:, 0] ** 2 + new_oz[:, 2] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        b = torch.arccos(cos)
        mask = new_oz[:, 0] > 0
        b[mask] = -b[mask]
        R_y = self.R_y(repeat(b, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_y)
        new_oy = torch.cross(coords[:, 0], coords[:, 1])  # (CA-N x CA-C)
        cos = new_oy[:, 1] / (
            (new_oy[:, 1] ** 2 + new_oy[:, 0] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        g = torch.arccos(cos)
        mask = new_oy[:, 0] < 0
        g[mask] = -g[mask]
        R_z = self.R_z(repeat(g, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_z)

        # apply the rotation
        coords = torch.einsum("lkj,lij->lki", coords, R)

        # go back to global orientations
        R_z = self.R_z(repeat(-g, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_z)
        R_y = self.R_y(repeat(-b, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_y)
        R_x = self.R_x(repeat(-a, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_x)

        # uncenter the coordinates
        coords = coords + center
        coords = rearrange(coords, "(b n) k d -> b n k d", b=batch_size)

        # rotate C=O so that it is in the plane defined by C-N' and C-CA
        coords = self.rotate_oxygens(
            coords, chain_encoding, mask_exist, angle=oxy_angle, vector=vector
        )
        return coords

    def rotate_oxygens(
        self, coords, chain_encoding, mask_exist, angle=None, vector=False
    ):
        # set values
        if angle is None:
            vector = False
        else:
            raise NotImplementedError

        # add the next N to the atoms
        eps = 1e-7
        coords_oxy = torch.cat([coords[:, :-1, :], coords[:, 1:, [0]]], dim=-2)

        # rotation axis: C-N' x C-CA, angles: 121°
        new_oz = torch.cross(
            coords_oxy[:, :, 4] - coords_oxy[:, :, 1],
            coords_oxy[:, :, 2] - coords_oxy[:, :, 1],
        )  # C-N' x C-CA
        if angle is None:
            angle = torch.ones_like(mask_exist[:, :-1]) * 121 * torch.pi / 180

        # center the coordinates on C
        center = coords_oxy[:, :, 1, :].unsqueeze(-2)
        coords_oxy = coords_oxy - center

        # rotate so that the new axis is in the (x,z) plane
        cos = new_oz[:, :, 2] / (
            (new_oz[:, :, 1] ** 2 + new_oz[:, :, 2] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        a = torch.arccos(cos)
        mask = new_oz[:, :, 1] < 0
        a[mask] = -a[mask]
        R_x = self.R_x(rearrange(a, "b l -> (b l) 1"))
        R_x = rearrange(R_x, "(b l) k j -> b l k j", b=coords_oxy.shape[0])
        coords_oxy = torch.einsum("blkj,blij->blki", coords_oxy, R_x)

        # recompute the new axis
        new_oz = torch.cross(
            coords_oxy[:, :, 4] - coords_oxy[:, :, 1],
            coords_oxy[:, :, 2] - coords_oxy[:, :, 1],
        )  # C-N' x C-CA

        # rotate so that the new axis is in the (y,z) plane
        cos = new_oz[:, :, 2] / (
            (new_oz[:, :, 0] ** 2 + new_oz[:, :, 2] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        b = torch.arccos(cos)
        mask = new_oz[:, :, 0] > 0
        b[mask] = -b[mask]
        R_y = self.R_y(rearrange(b, "b l -> (b l) 1"))
        R_y = rearrange(R_y, "(b l) k j -> b l k j", b=coords_oxy.shape[0])
        coords_oxy = torch.einsum("blkj,blij->blki", coords_oxy, R_y)

        # rotate C-CA unit vector by 121° around the new axis and multiply by 1.24 to make it a C=O bond
        R_z = self.R_z(rearrange(angle, "b l ... -> (b l) 1 ..."), vector=vector)
        R_z = rearrange(R_z, "(b l) k j -> b l k j", b=coords.shape[0])
        n = coords_oxy[:, :, [2]] - coords_oxy[:, :, [1]]
        n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-7)
        coords_O = torch.einsum("blkj,blij->blki", n, R_z) * 1.24
        coords_oxy = torch.cat([coords_oxy[:, :, :3], coords_O], dim=-2)

        # go back to global orientations
        R_y = self.R_y(rearrange(-b, "b l -> (b l) 1"))
        R_y = rearrange(R_y, "(b l) k j -> b l k j", b=coords_oxy.shape[0])
        coords_oxy = torch.einsum("blkj,blij->blki", coords_oxy, R_y)
        R_x = self.R_x(rearrange(-a, "b l -> (b l) 1"))
        R_x = rearrange(R_x, "(b l) k j -> b l k j", b=coords_oxy.shape[0])
        coords_oxy = torch.einsum("blkj,blij->blki", coords_oxy, R_x)
        coords_oxy = coords_oxy + center

        # remove updates in residues that neighbor a gap or a chain change
        mask_ = ~(mask_exist[:, 1:].bool())
        coords_oxy[mask_] = coords[:, :-1][mask_]
        chain_change = torch.diff(chain_encoding, dim=1) != 0
        coords_oxy[chain_change] = coords[:, :-1][chain_change]

        # add the last aminoacid
        coords = torch.cat([coords_oxy, coords[:, -1:]], dim=1)

        return coords

    def apply_encoder(
        self,
        h_V,
        h_E,
        E_idx,
        mask,
        vectors,
        residue_idx,
        chain_encoding_all,
        global_context,
        coords,
        cycle,
    ):
        if self.use_checkpointing:
            h_V, h_E, vectors, E_idx, coords = checkpoint(
                self.encoders[min(cycle, len(self.encoders) - 1)],
                h_V,
                h_E,
                E_idx,
                mask,
                vectors,
                residue_idx,
                chain_encoding_all,
                global_context,
                coords,
            )
        else:
            h_V, h_E, vectors, E_idx, coords = self.encoders[
                min(cycle, len(self.encoders) - 1)
            ](
                h_V,
                h_E,
                E_idx,
                mask,
                vectors,
                residue_idx,
                chain_encoding_all,
                global_context,
                coords,
            )
        return h_V, h_E, vectors, E_idx, coords

    def apply_decoder(
        self,
        h_V,
        h_E,
        E_idx,
        mask,
        vectors,
        residue_idx,
        chain_encoding_all,
        global_context,
        coords,
        h_S,
        chain_M,
        seq,
        test,
        cycle,
    ):
        decoder_module = self.decoders[min(cycle, len(self.decoders) - 1)]
        if self.one_shot_decoder:
            if self.add_sequence_in_decoder:
                h_E = cat_neighbors_nodes(h_S, h_E, E_idx)
            args = (
                h_V,
                h_E,
                E_idx,
                mask,
                vectors,
                residue_idx,
                chain_encoding_all,
                global_context,
                coords,
            )
            if self.use_checkpointing:
                (
                    h_V,
                    h_E,
                    vectors,
                    E_idx,
                    coords,
                ) = checkpoint(decoder_module, *args)
            else:
                (
                    h_V,
                    h_E,
                    vectors,
                    E_idx,
                    coords,
                ) = decoder_module(*args)
        else:
            h_V, h_E, vectors, E_idx = decoder_module(
                h_V,
                h_E,
                h_S,
                E_idx,
                mask,
                vectors,
                chain_M,
                test=test,
                out=self.W_out,
                out_seq=self.W_s,
                num_letters=self.num_letters,
                temperature=self.temperature,
                seq=seq.clone(),
            )
        return h_V, h_E, vectors, E_idx, coords

    def run_cycle(
        self,
        cycle,
        seq,
        coords,
        chain_M,
        optional_features,
        mask,
        residue_idx,
        chain_encoding_all,
        transform,
        timestep,
        test=False,
        corrupt=True,
    ):
        (
            global_context,
            translation,
            angles,
            logits,
            seq_t,
            translation_gt,
            rotation_gt,
            distribution_t,
        ) = [None] * 8
        if self.co_design == "seq":
            if cycle % 2 == 0:
                self.predict_structure = True
                self.predict_sequence = False
            else:
                self.predict_structure = False
                self.predict_sequence = True
        seq = seq.detach()
        coords = coords.detach()
        (
            h_V,
            h_E,
            E_idx,
            vectors,
            h_S,
            coords,
            translation_gt,
            rotation_gt,
            seq_t,
            distribution_t,
            global_context,
        ) = self.extract_features(
            seq.clone(),
            chain_M,
            optional_features,
            mask,
            residue_idx,
            chain_encoding_all,
            coords,
            cycle,
            transform=transform,
            timestep=timestep,
            corrupt=corrupt,
        )
        if not self.use_graph_context:
            global_context = None
        coords_t = coords.clone()
        if self.predict_angles and not self.vector_angles and self.diffusion:
            orientations, orientations_inverse, _ = get_orientations(coords)
        h_V, h_E, vectors, E_idx, coords = self.apply_encoder(
            h_V,
            h_E,
            E_idx,
            mask,
            vectors,
            residue_idx,
            chain_encoding_all,
            global_context,
            coords,
            cycle,
        )
        if not self.pass_features:
            h_V, h_E, *_ = self.extract_features(
                seq.clone(),
                chain_M,
                optional_features,
                mask,
                residue_idx,
                chain_encoding_all,
                vectors,
                cycle,
                transform=transform,
                corrupt=corrupt,
            )
        h_V, h_E, vectors, E_idx, coords = self.apply_decoder(
            h_V,
            h_E,
            E_idx,
            mask,
            vectors,
            residue_idx,
            chain_encoding_all,
            global_context,
            coords,
            h_S,
            chain_M,
            seq,
            test,
            cycle,
        )
        if isinstance(h_V, tuple):
            h_V, angles = h_V
        else:
            angles = h_V
        if self.predict_sequence:
            logits, seq = self.process_sequence(
                h_V,
                seq,
                chain_M,
                cycle,
            )
        if self.predict_structure:
            coords, angles, translation = self.process_structure(
                coords, angles, vectors, chain_encoding_all, mask, chain_M, cycle
            )
            if self.predict_angles and not self.vector_angles and self.diffusion:
                angles = self.diffusion.get_global_rotation(
                    angles, orientations, orientations_inverse, return_so3=True
                )
        return (
            coords,
            translation,
            angles,
            logits,
            seq_t,
            translation_gt,
            rotation_gt,
            distribution_t,
            coords_t,
        )

    def diffuse(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        optional_features,
        save_path=None,
        **kwargs,
    ):
        seq = deepcopy(S)
        coords = deepcopy(X)
        if self.predict_sequence or self.co_design != "none":
            seq, distribution = self.diffusion.noise_sequence(
                seq,
                chain_M,
                self.num_diffusion_steps * torch.ones(X.shape[0], dtype=torch.long),
                inference=True,
            )
        if self.predict_structure or self.co_design != "none":
            coords, *_, std_coords = self.diffusion.noise_structure(
                coords,
                chain_M,
                self.predict_angles,
                (self.num_diffusion_steps) * torch.ones(X.shape[0], dtype=torch.long),
                variance_scale=self.variance_scale,
                inference=True,
            )
        if self.predict_angles:
            orientations, _, local_coords = get_orientations(coords)
        else:
            orientations = None
        coords_ca = coords[:, :, 2, :]
        chain_M_bool = chain_M.bool()
        if save_path is not None:
            self._save_step(
                coords[0],
                seq[0],
                chain_M[0],
                mask[0],
                chain_encoding_all[0],
                kwargs["chain_dict"][0],
                save_path,
                0,
            )
        for t in list(range(1, self.num_diffusion_steps + 1))[::-1]:
            timestep = t * torch.ones(X.shape[0], dtype=torch.long)
            coords_ = coords.clone()
            for cycle in range(self.n_cycles):
                coords_predicted, translation, angles, logits, *_ = self.run_cycle(
                    cycle,
                    seq=seq,
                    coords=coords_,
                    chain_M=chain_M,
                    optional_features=optional_features,
                    mask=mask,
                    residue_idx=residue_idx,
                    chain_encoding_all=chain_encoding_all,
                    transform=None,
                    timestep=timestep,
                    corrupt=False,
                    test=True,
                )
                if cycle < self.n_cycles - 1:
                    if logits is not None:
                        seq = deepcopy(S)
                        seq[chain_M.bool()] = torch.max(logits[chain_M.bool()], -1)[1]
                    coords_ = deepcopy(X)
                    coords_[chain_M.bool()] = coords[chain_M.bool()]

            if self.predict_sequence:
                seq_new, distribution = self.diffusion.denoise_sequence(
                    distribution, logits, chain_M, timestep
                )
                seq[chain_M_bool] = seq_new[chain_M_bool]

            if self.diff_predict != "noise":
                translation = coords_predicted

            # translation_gt = repeat(self.diffusion._get_v(x0=X[:, :, 2], noise=translation_gt, timestep=timestep), "b n d -> b n 4 d")

            if self.predict_structure:
                coords[~chain_M.bool()] = X[~chain_M.bool()]
                coords_ca_new, orientations_new = self.diffusion.denoise_structure(
                    coords=coords,
                    orientations=orientations,
                    translation_predicted=translation,
                    rotation_predicted=angles,
                    std_coords=std_coords,
                    predict_angles=self.predict_angles,
                    timestep=timestep,
                    chain_M=chain_M,
                    mask=mask,
                )
                if self.reset_masked:
                    coords_ca_new[~mask.bool()] = 0
                coords_ca[chain_M_bool] = coords_ca_new[chain_M_bool]
                if self.predict_angles:
                    orientations[chain_M_bool] = orientations_new[chain_M_bool]
                    coords_new = self.construct_coords(
                        coords_ca,
                        orientations,
                        chain_encoding_all,
                        local_coords,
                        mask,
                    )
                else:
                    coords_new = (
                        repeat(coords_ca_new, "b l d -> b l n d", n=4)
                        + coords
                        - coords[:, :, [2], :]
                    )
                coords[chain_M_bool] = coords_new[chain_M_bool]
            if save_path is not None:
                self._save_step(
                    coords[0],
                    seq[0],
                    chain_M[0],
                    mask[0],
                    chain_encoding_all[0],
                    kwargs["chain_dict"][0],
                    save_path,
                    self.num_diffusion_steps - t + 1,
                )
        out = {}
        if self.predict_sequence:
            out["seq"] = torch.log(distribution + 1e-7)
        if self.predict_structure:
            out["coords"] = coords
        return [out]

    def _save_step(
        self,
        coords,
        seq,
        chain_M,
        mask,
        chain_encoding_all,
        chain_dict,
        save_path,
        t,
    ):
        predicted_protein_entry = ProteinEntry.from_arrays(
            seq,
            coords,
            mask,
            chain_dict,
            chain_encoding_all,
            mask * chain_M,
        )
        predicted_protein_entry.to_pickle(os.path.join(save_path, f"step_{t}.pickle"))

    def process_sequence(self, h_V, seq, chain_M, cycle):
        logits = self.W_out(h_V)
        seq = seq.clone()
        if self.probability_mode:
            if cycle == 0:
                seq = F.one_hot(seq, num_classes=len(ALPHABET)).half()
            if self.num_letters == 21:
                seq[chain_M.bool()] = logits[chain_M.bool()]
            else:
                seq[chain_M.bool(), 1:] = logits[chain_M.bool()]
        else:
            seq[chain_M.bool()] = torch.max(logits.detach(), -1)[1][chain_M.bool()]
            if self.num_letters == 20:
                seq[chain_M.bool()] += 1
        return logits, seq

    def process_structure(
        self, coords, angles, vectors, chain_encoding_all, mask, chain_M, cycle
    ):
        if self.angle_layer is not None:
            angles = self.angle_layer(angles)
        elif self.vector_angles:
            angles = vectors[:, :, 1, :]
        if not self.diffusion:
            angles = rearrange(angles, "b n ... -> (b n) ...")

        chain_M_bool = chain_M.bool()
        if self.vector_update:
            coords[chain_M_bool] = (
                coords[chain_M_bool] + vectors[:, :, [0], :][chain_M_bool]
            )
            translation = vectors[:, :, 0, :]
        else:
            coords_orig = coords.clone()
            coords[chain_M_bool] = vectors[:, :, :4, :][chain_M_bool]
            translation = coords[:, :, 2, :] - coords_orig[:, :, 2, :]

        if self.predict_angles and not self.diffusion:
            coords = self.rotate(
                coords,
                angles,
                chain_encoding_all,
                mask,
                quaternion=self.quaternion,
                vector=self.vector_angles,
            )
        return coords, angles, translation

    def get_mu_t(self, x0, xt, x0_pred, timestep):
        return self.diffusion._get_mu_t(x0, xt, x0_pred, timestep)

    def load_state_dict(self, state_dict):
        strict = False
        if self.co_design == "seq" and len(self.state_dict()) != len(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                name_split = k.split(".")
                if name_split[0] in ["encoders", "decoders"]:
                    name_split[1] = str(int(name_split[1]) * 2)
                    new_state_dict[".".join(name_split)] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            strict = False
        if self.co_design == "share_enc" and len(self.state_dict()) != len(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                name_split = k.split(".")
                if name_split[0] == "decoders":
                    name_split = name_split[: 2] + ["seq_decoder"] + name_split[2:]
                    new_state_dict[".".join(name_split)] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            strict = False
        super().load_state_dict(state_dict, strict=strict)
        if self.co_design =="share_enc" and len(self.state_dict()) != len(state_dict): # freeze the encoder and the seq decoder
            for param in self.parameters():
                param.requires_grad = False
            for dec in self.decoders:
                for param in dec.coords_decoder.parameters():
                    param.requires_grad = True
            if self.predict_angles:
                for param in self.angle_layer.parameters():
                    param.requires_grad = True
            
            # for param in self.encoders.parameters():
            #     param.requires_grad = False
            # for param in self.W_out.parameters():
            #     param.requires_grad = False
            # for param in self.W_s.parameters():
            #     param.requires_grad = False
            # for param in self.W_e.parameters():
            #     param.requires_grad = False
            # for param in self.features.parameters():
            #     param.requires_grad = False
            # for dec in self.decoders[::2]:
            #     for param in dec.parameters():
            #         param.requires_grad = False

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        optional_features,
        test=False,
        transform=None,
    ):
        """Graph-conditioned sequence model"""

        output = []
        seq = deepcopy(S)
        coords = deepcopy(X)
            # timestep = torch.ones(X.shape[:1])
        if self.diffusion is not None:
            timestep = torch.randint(1, self.num_diffusion_steps + 1, size=X.shape[:1])
        else:
            timestep = None
        translation_gt, rotation_gt, seq_t, distribution = None, None, None, None
        for cycle in range(self.n_cycles):
            (
                coords,
                translation,
                angles,
                logits,
                seq_t_,
                translation_gt_,
                rotation_gt_,
                distribution_,
                coords_t,
            ) = self.run_cycle(
                cycle,
                seq,
                coords,
                chain_M,
                optional_features,
                mask,
                residue_idx,
                chain_encoding_all,
                transform,
                timestep,
                test=test,
            )
            if seq_t_ is not None:
                seq_t = seq_t_
            if translation_gt_ is not None:
                translation_gt = translation_gt_
            if rotation_gt_ is not None:
                rotation_gt = rotation_gt_
            if distribution_ is not None:
                distribution = distribution_
            out = {"timestep": timestep}
            if self.predict_sequence:
                out["seq"] = logits
                out["seq_t"] = distribution
            if self.predict_structure:
                if self.diffusion:
                    if self.predict_angles:
                        out["rotation"] = angles
                        out["rotation_gt"] = rotation_gt
                    if self.diff_predict == "noise":
                        out["CA"] = translation
                        out["CA_gt"] = translation_gt
                    elif self.diff_predict == "x0":
                        out["CA"] = coords[:, :, 2]
                        out["CA_gt"] = X[:, :, 2]
                    elif self.diff_predict == "mu_t":
                        coords_ca_gt = self.diffusion._get_mu_t(
                            x0=X[:, :, 2], xt=coords_t[:, :, 2], timestep=timestep
                        )
                        out["CA"] = coords[:, :, 2]
                        out["CA_gt"] = coords_ca_gt
                    elif self.diff_predict == "v":
                        out["CA"] = coords[:, :, 2]
                        out["CA_gt"] = self.diffusion._get_v(
                            x0=X[:, :, 2], noise=translation_gt, timestep=timestep
                        )
                else:
                    out["coords"] = coords.clone()
            output.append(out)
            if cycle < self.n_cycles - 1:
                if logits is not None:
                    seq[chain_M.bool()] = torch.max(logits[chain_M.bool()], -1)[1]
                elif seq_t is not None:
                    seq[chain_M.bool()] = seq_t[chain_M.bool()]
                coords_ = deepcopy(X)
                coords_[chain_M.bool()] = coords[chain_M.bool()]
                coords = coords_

        return output
