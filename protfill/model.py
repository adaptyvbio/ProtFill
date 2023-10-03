from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange
from copy import deepcopy

from protfill.layers.gvp import GVP_Decoder, GVP_Encoder
from protfill.layers.gvpe import GVPe_Decoder, GVPe_Encoder
from protfill.utils.model_utils import *
from protfill.diffusion import Diffuser, get_orientations


class Rotation:
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
    def rotation_matrices_from_angles(angles):
        """
        Computes a tensor of 3D rotation matrices from a tensor of 3D rotation angles.
        """
        # Check if the angles are in the correct shape (B * N, 3) or (B * N, 3, 3) for vector angles
        if len(angles.shape) == 3:
            angles = rearrange(angles, "b n ... -> (b n) ...")
        # Extract the rotation angles for each matrix
        alpha, beta, gamma = torch.chunk(angles, 3, dim=1)

        # Compute the rotation matrices
        R_x = Rotation.R_x(alpha, vector=False)
        R_y = Rotation.R_y(beta, vector=False)
        R_z = Rotation.R_z(gamma, vector=False)

        # Return the composed rotation matrices
        return torch.einsum("nij,njk,nkl->nil", R_z, R_y, R_x)
    
    @staticmethod
    def rotate(
        coords,
        angles,
        chain_encoding,
        mask_exist,
    ):
        batch_size = coords.size(0)
        coords = rearrange(coords, "b n k d -> (b n) k d")

        # center the coordinates on CA
        center = coords[:, 2, :].unsqueeze(1)
        coords = coords - center

        # compute rotation matrices
        R = Rotation.rotation_matrices_from_angles(angles)

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
        R_x = Rotation.R_x(repeat(a, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_x)
        new_oz = coords[:, 0]  # CA-N
        cos = new_oz[:, 2] / (
            (new_oz[:, 0] ** 2 + new_oz[:, 2] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        b = torch.arccos(cos)
        mask = new_oz[:, 0] > 0
        b[mask] = -b[mask]
        R_y = Rotation.R_y(repeat(b, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_y)
        new_oy = torch.cross(coords[:, 0], coords[:, 1])  # (CA-N x CA-C)
        cos = new_oy[:, 1] / (
            (new_oy[:, 1] ** 2 + new_oy[:, 0] ** 2 + 1e-7).sqrt() + 1e-7
        )
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        g = torch.arccos(cos)
        mask = new_oy[:, 0] < 0
        g[mask] = -g[mask]
        R_z = Rotation.R_z(repeat(g, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_z)

        # apply the rotation
        coords = torch.einsum("lkj,lij->lki", coords, R)

        # go back to global orientations
        R_z = Rotation.R_z(repeat(-g, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_z)
        R_y = Rotation.R_y(repeat(-b, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_y)
        R_x = Rotation.R_x(repeat(-a, "l -> l 1"))
        coords = torch.einsum("lkj,lij->lki", coords, R_x)

        # uncenter the coordinates
        coords = coords + center
        coords = rearrange(coords, "(b n) k d -> b n k d", b=batch_size)

        # rotate C=O so that it is in the plane defined by C-N' and C-CA
        coords = Rotation.rotate_oxygens(
            coords, chain_encoding, mask_exist,
        )
        return coords

    @staticmethod
    def rotate_oxygens(
        coords, chain_encoding, mask_exist,
    ):
        # add the next N to the atoms
        eps = 1e-7
        coords_oxy = torch.cat([coords[:, :-1, :], coords[:, 1:, [0]]], dim=-2)

        # rotation axis: C-N' x C-CA, angles: 121°
        new_oz = torch.cross(
            coords_oxy[:, :, 4] - coords_oxy[:, :, 1],
            coords_oxy[:, :, 2] - coords_oxy[:, :, 1],
        )  # C-N' x C-CA
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
        R_x = Rotation.R_x(rearrange(a, "b l -> (b l) 1"))
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
        R_y = Rotation.R_y(rearrange(b, "b l -> (b l) 1"))
        R_y = rearrange(R_y, "(b l) k j -> b l k j", b=coords_oxy.shape[0])
        coords_oxy = torch.einsum("blkj,blij->blki", coords_oxy, R_y)

        # rotate C-CA unit vector by 121° around the new axis and multiply by 1.24 to make it a C=O bond
        R_z = Rotation.R_z(rearrange(angle, "b l ... -> (b l) 1 ..."))
        R_z = rearrange(R_z, "(b l) k j -> b l k j", b=coords.shape[0])
        n = coords_oxy[:, :, [2]] - coords_oxy[:, :, [1]]
        n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-7)
        coords_O = torch.einsum("blkj,blij->blki", n, R_z) * 1.24
        coords_oxy = torch.cat([coords_oxy[:, :, :3], coords_O], dim=-2)

        # go back to global orientations
        R_y = Rotation.R_y(rearrange(-b, "b l -> (b l) 1"))
        R_y = rearrange(R_y, "(b l) k j -> b l k j", b=coords_oxy.shape[0])
        coords_oxy = torch.einsum("blkj,blij->blki", coords_oxy, R_y)
        R_x = Rotation.R_x(rearrange(-a, "b l -> (b l) 1"))
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
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        num_dist = 16 
        edge_in = num_positional_embeddings + num_rbf * num_dist
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.timestep = None

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
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
        linear=False,
    ):
        if timestep is not None:
            self.timestep = timestep
        b = X[:, :, 2, :] - X[:, :, 0, :]
        c = X[:, :, 1, :] - X[:, :, 2, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 2, :]
        C = X[:, :, 1, :]
        N = X[:, :, 0, :]
        Ca = X[:, :, 2, :]

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
                C, mask
            )

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
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)  # + 16

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

        return E, E_idx, timestep_embedding


class ProtFill(nn.Module):
    def __init__(
        self,
        args,
        encoder_type,
        decoder_type,
        hidden_dim=128,
        embedding_dim=128,
        num_letters=21,
        k_neighbors=32,
        noise_std=None,
        n_cycles: int = 1,
        separate_modules_num: int = 1,
    ):
        super(ProtFill, self).__init__()
        encoders = {
            "gvpe": GVPe_Encoder,
            "gvp": GVP_Encoder,
        }
        decoders = {
            "gvpe": GVPe_Decoder,
            "gvp": GVP_Decoder,
        }

        self.diffusion = (
            Diffuser(
                num_tokens=num_letters,
                num_steps=args.num_diffusion_steps,
                schedule_name="cosine",
                seq_diffusion_type="mask",
                recover_x0=True,
                linear_interpolation=False,
                diff_predict="x0",
                weighted_diff_loss=False,
                pos_std=args.noise_std,
                noise_around_interpolation=False,
                no_added_noise=True,
            )
            if args.diffusion
            else None
        )
        self.diffusion_ = Diffuser(
            num_tokens=num_letters,
            num_steps=args.num_diffusion_steps,
            schedule_name="cosine",
            seq_diffusion_type="mask",
            linear_interpolation=False,
            pos_std=args.noise_std,
        )
        self.num_diffusion_steps = args.num_diffusion_steps
        self.num_letters = num_letters
        self.predict_structure = True
        self.predict_sequence = True

        if noise_std is None:
            noise_std = 5 if self.predict_structure else 0.
        self.num_letters = num_letters
        self.noise_unknown = noise_std
        self.n_cycles = n_cycles
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.old_noise = args.alternative_noising

        if separate_modules_num > n_cycles:
            separate_modules_num = n_cycles

        self.str_features = False
        self.seq_features = False

        self.features = ProteinFeatures(
            hidden_dim,
            top_k=k_neighbors,
        )
        args.edge_compute_func = self.features
        args.vector_dim = 6

        add_dim = 16 if self.diffusion else 0
        self.W_e = nn.Linear(hidden_dim + add_dim, hidden_dim, bias=True)
        self.W_s = (
            nn.Embedding(num_letters, embedding_dim)
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
        if not self.seq_features:
            in_dim += embedding_dim
        else:
            in_dim += hidden_dim
        args.in_dim = in_dim

        self.decoders = [
            decoders[decoder_type](args)
            for _ in range(
                separate_modules_num * 2
            )
        ]
        self.decoders = [combine_decoders(
            coords_decoder=self.decoders[c * 2 + 1],  # coords
            seq_decoder=self.decoders[c * 2],  # seq
            predict_angles=True,
        ) for c in range(separate_modules_num)]
        self.decoders = nn.ModuleList(self.decoders)
        if self.predict_sequence:
            self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        self.W_vector = nn.Linear(args.vector_dim, 4)

        self.angle_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def random_rotation(self, X, chain_labels, lim=torch.pi):
        n = chain_labels.sum()
        angles = torch.randn((n, 3)) * lim - lim / 2
        R, _ = self.rotation_matrices_from_angles(angles)
        R = R.to(X.device)
        mean = X[chain_labels].mean(-2).unsqueeze(-2)
        out = torch.einsum("ndj,naj->nad", R, X[chain_labels] - mean) + mean
        return out.float()

    def noise_coords(
        self,
        X,
        chain_M,
        timestep=None,
    ):
        """
        Add noise to the coordinates (augmentation + masking for structure prediction)
        """

        rotation, translation = None, None

        if self.predict_structure or self.co_design != "none" or self.noise_structure:
            if self.diffusion:
                X, rotation, translation, _ = self.diffusion.noise_structure(
                    X, chain_M, True, timestep
                )
            else:
                chain_M_bool = chain_M.bool()
                if not self.old_noise:
                    X, *_ = self.diffusion_.noise_structure(
                        X,
                        chain_M,
                        True,
                        (self.num_diffusion_steps)
                        * torch.ones(X.shape[0], dtype=torch.long),
                        inference=True,
                        variance_scale=1.,
                    )
                else:
                    coords_X = X[:, :, :4]
                    masked_X = coords_X[chain_M_bool].clone()
                    masked_X += self.noise_unknown * torch.randn_like(
                        masked_X[:, 0, :]
                    ).unsqueeze(1)
                    coords_X[chain_M_bool] = masked_X
                    coords_X[chain_M_bool] = self.random_rotation(coords_X, chain_M_bool)
                    X[:, :, :4] = coords_X

        return X, rotation

    def initialize_sequence(
        self,
        seq,
        chain_M,
        timestep=None,
    ):
        """
        Initialize the sequence values for the masked regions
        """

        distribution = None
        if self.predict_sequence or self.co_design != "none" or self.mask_sequence:
            if self.diffusion:
                seq, distribution = self.diffusion.noise_sequence(
                    seq, chain_M, timestep
                )
            else:
                seq[chain_M.bool()] = 0

        return seq

    def extract_features(
        self,
        seq,
        chain_M,
        mask,
        residue_idx,
        chain_encoding_all,
        X,
        cycle,
        timestep=None,
        corrupt=True,
    ):
        """
        Extract features from the input sequence and structure
        """

        rotation_gt = None

        if cycle == 0 and corrupt:
            if isinstance(timestep, int):
                timestep = timestep * torch.ones(X.shape[0], device=X.device)
            seq = self.initialize_sequence(
                seq,
                chain_M,
                timestep=timestep,
            )
            (
                X,
                rotation_gt,
            ) = self.noise_coords(
                X,
                chain_M,
                timestep=timestep,
            )
        timestep_factor = 25 / self.num_diffusion_steps
        timestep_ = None if timestep is None else timestep * timestep_factor
        E, E_idx, timestep_rbf = self.features(
            X[:, :, :4],
            mask,
            residue_idx,
            chain_encoding_all,
            timestep=timestep_,
        )
        if timestep is not None:
            E = torch.cat([E, repeat(timestep_rbf, "b d -> b l k d", l=E.shape[1], k=E.shape[2])], dim=-1)

        h_S = self.W_s(seq)

        # Prepare node and edge embeddings
        h_V = torch.zeros(
            (E.shape[0], E.shape[1], self.hidden_dim), device=E.device
        )  # node embeddings = zeros

        h_E = self.W_e(E)

        return (
            h_V,
            h_E,
            E_idx,
            h_S,
            X,
            rotation_gt,
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
        # coords = Rotation.rotate_oxygens(coords, chain_encoding_all, mask)
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
        (
            h_V,
            h_E,
            vectors,
            E_idx,
            coords,
        ) = decoder_module(*args)
        return h_V, h_E, vectors, E_idx, coords

    def run_cycle(
        self,
        cycle,
        seq,
        coords,
        chain_M,
        mask,
        residue_idx,
        chain_encoding_all,
        timestep,
        test=False,
        corrupt=True,
    ):
        (
            global_context,
            angles,
            logits,
            rotation_gt,
        ) = [None] * 4
        seq = seq.detach()
        coords = coords.detach()
        (
            h_V,
            h_E,
            E_idx,
            h_S,
            coords,
            rotation_gt,
        ) = self.extract_features(
            seq.clone(),
            chain_M,
            mask,
            residue_idx,
            chain_encoding_all,
            coords,
            cycle,
            timestep=timestep,
            corrupt=corrupt,
        )
        global_context = None
        if self.diffusion:
            orientations, orientations_inverse, _ = get_orientations(coords)
        h_V, h_E, vectors, E_idx, coords = self.apply_encoder(
            h_V,
            h_E,
            E_idx,
            mask,
            coords,
            residue_idx,
            chain_encoding_all,
            global_context,
            coords,
            cycle,
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
            logits = self.W_out(h_V)
        if self.predict_structure:
            coords, angles = self.process_structure(
                coords, angles, vectors, chain_encoding_all, mask, chain_M
            )
            if self.diffusion:
                angles = self.diffusion.get_global_rotation(
                    angles, orientations, orientations_inverse, return_so3=True
                )
        return (
            coords,
            angles,
            logits,
            rotation_gt,
        )

    def diffuse(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        save_path=None,
        **kwargs,
    ):
        seq = deepcopy(S)
        coords = deepcopy(X)
        seq, distribution = self.diffusion.noise_sequence(
            seq,
            chain_M,
            self.num_diffusion_steps * torch.ones(X.shape[0], dtype=torch.long),
            inference=True,
        )
        coords, *_, std_coords = self.diffusion.noise_structure(
            coords,
            chain_M,
            True,
            (self.num_diffusion_steps) * torch.ones(X.shape[0], dtype=torch.long),
            variance_scale=1.,
            inference=True,
        )
        orientations, _, local_coords = get_orientations(coords)
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
                coords_predicted, angles, logits, _ = self.run_cycle(
                    cycle,
                    seq=seq,
                    coords=coords_,
                    chain_M=chain_M,
                    mask=mask,
                    residue_idx=residue_idx,
                    chain_encoding_all=chain_encoding_all,
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

            # translation_gt = repeat(self.diffusion._get_v(x0=X[:, :, 2], noise=translation_gt, timestep=timestep), "b n d -> b n 4 d")

            if self.predict_structure:
                coords[~chain_M.bool()] = X[~chain_M.bool()]
                coords_ca_new, orientations_new = self.diffusion.denoise_structure(
                    coords=coords,
                    orientations=orientations,
                    translation_predicted=coords_predicted,
                    rotation_predicted=angles,
                    std_coords=std_coords,
                    predict_angles=True,
                    timestep=timestep,
                    chain_M=chain_M,
                    mask=mask,
                )
                coords_ca[chain_M_bool] = coords_ca_new[chain_M_bool]
                orientations[chain_M_bool] = orientations_new[chain_M_bool]
                coords_new = self.construct_coords(
                    coords_ca,
                    orientations,
                    chain_encoding_all,
                    local_coords,
                    mask,
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

    def process_structure(
        self, coords, angles, vectors, chain_encoding_all, mask, chain_M
    ):
        angles = self.angle_layer(angles)

        chain_M_bool = chain_M.bool()
        coords[chain_M_bool] = (
            coords[chain_M_bool] + vectors[:, :, [0], :][chain_M_bool]
        )

        if not self.diffusion:
            angles = rearrange(angles, "b n ... -> (b n) ...")
            coords = Rotation.rotate(
                coords,
                angles,
                chain_encoding_all,
                mask,
            )
        return coords, angles

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        test=False,
    ):
        """Graph-conditioned sequence model"""

        output = []
        seq = deepcopy(S)
        coords = deepcopy(X)
        if self.diffusion is not None:
            timestep = torch.randint(1, self.num_diffusion_steps + 1, size=X.shape[:1])
        else:
            timestep = None
        for cycle in range(self.n_cycles):
            (
                coords,
                angles,
                logits,
                angles_gt
            ) = self.run_cycle(
                cycle,
                seq,
                coords,
                chain_M,
                mask,
                residue_idx,
                chain_encoding_all,
                timestep,
                test=test,
            )
            out = {"timestep": timestep}
            if self.predict_sequence:
                out["seq"] = logits
                out["seq_t"] = S
            if self.predict_structure:
                if self.diffusion:
                    out["rotation"] = angles
                    out["rotation_gt"] = angles_gt
                    out["CA"] = coords[:, :, 2]
                    out["CA_gt"] = X[:, :, 2]
                else:
                    out["coords"] = coords.clone()
            output.append(out)
            if cycle < self.n_cycles - 1:
                if logits is not None:
                    seq[chain_M.bool()] = torch.max(logits[chain_M.bool()], -1)[1]
                coords_ = deepcopy(X)
                coords_[chain_M.bool()] = coords[chain_M.bool()]
                coords = coords_

        return output
