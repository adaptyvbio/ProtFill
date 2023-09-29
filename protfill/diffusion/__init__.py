from torch.nn import functional as F
import torch
from torch import nn
import math
from einops import repeat, rearrange
import time
from protfill.utils.model_utils import get_seq_loss
import pandas as pd


def linear_interpolation(coords, mask):
    coords_ = coords.clone()
    coords_[(1 - mask).bool()] = torch.nan
    coords_ = coords_.detach().cpu().numpy()
    coords_ = rearrange(coords_, "b n d -> n (b d)")
    coords_ = pd.DataFrame(coords_)
    coords_ = coords_.interpolate(method="linear", limit_direction="both", axis=0)
    coords_ = rearrange(coords_.values, "n (b d) -> b n d", b=coords.shape[0])
    coords = torch.from_numpy(coords_).to(coords.device)
    return coords


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_around_axis(u, theta):
    """
    Compute the rotation matrix around the axis u by theta

    Parameters
    ----------
    u : torch.Tensor
        The axis of rotation `(..., 3)`
    theta : torch.Tensor
        The rotation angle `(...)`

    Returns
    -------
    R: torch.Tensor
        The rotation matrix `(..., 3, 3)`

    """

    # normalize the axis
    u = u / torch.norm(u, dim=-1, keepdim=True)

    # compute the rotation matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    R = torch.zeros_like(u)[..., None, None] + torch.eye(3, device=u.device)
    R = [
        cos_theta + ux**2 * (1 - cos_theta),
        ux * uy * (1 - cos_theta) - uz * sin_theta,
        ux * uz * (1 - cos_theta) + uy * sin_theta,
        uy * ux * (1 - cos_theta) + uz * sin_theta,
        cos_theta + uy**2 * (1 - cos_theta),
        uy * uz * (1 - cos_theta) - ux * sin_theta,
        uz * ux * (1 - cos_theta) - uy * sin_theta,
        uz * uy * (1 - cos_theta) + ux * sin_theta,
        cos_theta + uz**2 * (1 - cos_theta),
    ]
    R = rearrange(torch.stack(R, dim=-1), "... (k l) -> ... k l", l=3)
    return R


def safe_inverse(tensor):
    """
    Invert tensor where the first element of the last dimension is not zero
    """

    if len(tensor.shape) == 3:
        mask = tensor[:, :, 0] != 0
    else:
        mask = tensor[:, :, 0, 0] != 0
    tensor_inv = torch.zeros_like(tensor)
    tensor_inv[mask] = torch.inverse(tensor[mask])
    return tensor_inv


def get_orientations(coords):
    """
    Get inverse frame orientations from coordinates
    """

    v1 = coords[:, :, 1, :] - coords[:, :, 2, :]
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-7)
    v2 = coords[:, :, 0, :] - coords[:, :, 2, :]
    u2 = v2 - torch.einsum("b n d, b n d -> b n", v2, e1).unsqueeze(-1) * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-7)
    e3 = torch.cross(e1, e2, dim=-1)
    orientations = torch.stack([e1, e2, e3], dim=-1)
    orientations_inverse = safe_inverse(orientations)
    local_coords = torch.einsum(
        "b n k d, b n m d -> b n m k",
        orientations_inverse,
        coords - coords[:, :, [2], :],
    )
    return orientations, orientations_inverse, local_coords


class VarianceSchedule(nn.Module):
    """
    from https://github.com/luost26/diffab/blob/main/diffab/modules/diffusion/transition.py
    """

    def __init__(
        self, num_steps=100, s=0.01, nu=1, schedule="linear", seq_diffusion_type="mask", cosine_cutoff=0.8
    ):
        super().__init__()

        assert schedule in ["cosine", "linear", "quadratic"]
        T = num_steps
        if schedule == "cosine":
            t = torch.arange(0, num_steps + 1)
            f_t = torch.cos((torch.pi / 2) * (((t / T) + s) ** nu) / (1 + s)) ** 2
            alpha_bars = f_t / f_t[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = betas.clamp_max(cosine_cutoff)
        elif schedule == "linear":
            t = torch.arange(0, num_steps)
            betas = 1e-4 + (0.02 - 1e-4) * t / T
        elif schedule == "quadratic":
            t = torch.arange(0, num_steps)
            betas = (math.sqrt(0.00085) + (math.sqrt(0.012) - 1e-4) * t / T) ** 2
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alpha_bars = torch.cumprod(1 - betas, dim=0)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        Qs = repeat(torch.eye(21), "i j -> t i j", t=num_steps + 1)
        Qs = Qs * (1 - betas)[:, None, None]
        if seq_diffusion_type == "mask":
            Qs[:, :, 0] = betas[:, None]
            Qs[:, 0, 0] = 1
        elif seq_diffusion_type == "uniform":
            Qs += betas[:, None, None] / 21

        Q_bars = [torch.eye(21)]
        for Q in Qs[1:]:
            Q_bars.append(Q_bars[-1] @ Q)
        Q_bars = torch.stack(Q_bars, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("Qs", Qs)
        self.register_buffer("Q_bars", Q_bars)


class ApproxAngularDistribution(nn.Module):
    """
    from https://github.com/luost26/diffab/blob/main/diffab/modules/diffusion/transition.py
    """

    def __init__(self, stddevs, std_threshold=0.1, num_bins=8192, num_iters=1024):
        super().__init__()
        self.std_threshold = std_threshold
        self.num_bins = num_bins
        self.num_iters = num_iters
        self.register_buffer("stddevs", torch.FloatTensor(stddevs))
        self.register_buffer("approx_flag", self.stddevs <= std_threshold)
        self._precompute_histograms()

    @staticmethod
    def _pdf(x, e, L):
        """
        Args:
            x:  (N, )
            e:  Float
            L:  Integer
        """
        x = x[:, None]  # (N, *)
        c = (1 - torch.cos(x)) / math.pi  # (N, *)
        l = torch.arange(0, L)[None, :]  # (*, L)
        a = (2 * l + 1) * torch.exp(-l * (l + 1) * (e**2))  # (*, L)
        b = (torch.sin((l + 0.5) * x) + 1e-6) / (torch.sin(x / 2) + 1e-6)  # (N, L)

        f = (c * a * b).sum(dim=1)
        return f

    def _precompute_histograms(self):
        X, Y = [], []
        for std in self.stddevs:
            std = std.item()
            x = torch.linspace(0, math.pi, self.num_bins)  # (n_bins,)
            y = self._pdf(x, std, self.num_iters)  # (n_bins,)
            y = torch.nan_to_num(y).clamp_min(0)
            X.append(x)
            Y.append(y)
        self.register_buffer("X", torch.stack(X, dim=0))  # (n_stddevs, n_bins)
        self.register_buffer("Y", torch.stack(Y, dim=0))  # (n_stddevs, n_bins)

    def sample(self, std_idx):
        """
        Args:
            std_idx:  Indices of standard deviation.
        Returns:
            samples:  Angular samples [0, PI), same size as std.
        """

        size = std_idx.size()
        std_idx = std_idx.flatten()  # (N,)
        self.X = self.X.to(std_idx.device)
        self.Y = self.Y.to(std_idx.device)
        self.stddevs = self.stddevs.to(std_idx.device)
        self.approx_flag = self.approx_flag.to(std_idx.device)

        # Samples from histogram
        prob = self.Y[std_idx]  # (N, n_bins)
        bin_idx = torch.multinomial(prob[:, :-1], num_samples=1).squeeze(-1)  # (N,)
        bin_start = self.X[std_idx, bin_idx]  # (N,)
        bin_width = self.X[std_idx, bin_idx + 1] - bin_start
        samples_hist = bin_start + torch.rand_like(bin_start) * bin_width  # (N,)

        # Samples from Gaussian approximation
        std_gaussian = self.stddevs[std_idx]
        mean_gaussian = std_gaussian * 2
        samples_gaussian = (
            mean_gaussian + torch.randn_like(mean_gaussian) * std_gaussian
        )
        samples_gaussian = samples_gaussian.abs() % math.pi

        # Choose from histogram or Gaussian
        gaussian_flag = self.approx_flag[std_idx]
        samples = torch.where(gaussian_flag, samples_gaussian, samples_hist)

        return samples.reshape(size)


class Diffuser:
    def __init__(
        self,
        num_steps=100,
        num_tokens=20,
        schedule_name="linear",
        seq_diffusion_type="mask",
        recover_x0=True,
        linear_interpolation=False,
        diff_predict="noise",
        weighted_diff_loss=False,
        nu_pos=1.0,
        nu_rot=1.0,
        nu_seq=1.0,
        pos_std=None,
        noise_around_interpolation=False,
        no_added_noise=False,
        cosine_cutoff=0.8,
        all_x0=False,
    ):
        """
        Parameters
        ----------
        num_steps : int
            Number of diffusion steps
        num_tokens : int
            number of tokens in the vocabulary

        """

        self.var_sched_rot = VarianceSchedule(
            num_steps=num_steps, schedule=schedule_name, nu=nu_rot, cosine_cutoff=cosine_cutoff
        )
        self.var_sched_pos = VarianceSchedule(
            num_steps=num_steps, schedule=schedule_name, nu=nu_pos, cosine_cutoff=cosine_cutoff
        )
        self.var_sched_seq = VarianceSchedule(
            num_steps=num_steps,
            schedule=schedule_name,
            seq_diffusion_type=seq_diffusion_type,
            nu=nu_seq,
            cosine_cutoff=cosine_cutoff,
        )
        c1 = torch.sqrt(1 - self.var_sched_rot.alpha_bars)  # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist())
        sigma = self.var_sched_rot.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist())
        self.num_tokens = num_tokens
        self.recover_x0 = recover_x0
        self.seq_diffusion_type = seq_diffusion_type
        self.linear_interpolation = linear_interpolation
        self.num_steps = num_steps
        self.diff_predict = diff_predict
        self.weighted_diff_loss = weighted_diff_loss
        self.pos_std = pos_std
        self.noise_around_interpolation = noise_around_interpolation
        self.no_added_diff_noise = no_added_noise
        self.all_x0 = all_x0

    @staticmethod
    def random_uniform_so3(size, device="cpu"):
        """
        Generate random rotation matrix from uniform distribution
        """

        q = F.normalize(
            torch.randn(
                list(size)
                + [
                    4,
                ],
                device=device,
            ),
            dim=-1,
        )  # (..., 4)
        return quaternion_to_rotation_matrix(q)

    def get_global_rotation(
        self, rotation_local, orientations, orientations_inverse=None, return_so3=False
    ):
        """
        Go from local rotation to global

        Parameters
        ----------
        rotation_local : torch.Tensor
            local rotation so(3) vector or SO(3) matrix
        orientations : torch.Tensor
            orientations of the local frame
        orientations_inverse : torch.Tensor, optional
            inverse of the orientations
        return_so3 : bool, default False
            return so(3) vector instead of SO(3) matrix if `True`

        Returns
        -------
        rotation_global : torch.Tensor
            global rotation so(3) vector or SO(3) matrix

        """

        if orientations.shape[-2] == 4:
            orientations, orientations_inverse, _ = get_orientations(orientations)
        elif orientations_inverse is None:
            orientations_inverse = safe_inverse(orientations)
        if len(rotation_local.shape) == 3:
            rotation_global = torch.einsum(
                "...ij,...j->...i", orientations, rotation_local
            )
            if not return_so3:
                rotation_global = self._so3vec_to_rot(rotation_global)
        else:
            rotation_global = torch.einsum(
                "...ij,...jk->...ik", orientations, rotation_local
            )
            rotation_global = torch.einsum(
                "...ij,...jk->...ik", rotation_global, orientations_inverse
            )
            if return_so3:
                rotation_global = self._rot_to_so3vec(rotation_global)
        return rotation_global

    def get_local_rotation(
        self, rotation_global, orientations, orientations_inverse=None, return_so3=False
    ):
        """
        Go from global rotation to local

        Parameters
        ----------
        rotation_global : torch.Tensor
            global rotation so(3) vector or SO(3) matrix
        orientations : torch.Tensor
            orientations of the local frame
        orientations_inverse : torch.Tensor, optional
            inverse of the orientations
        return_so3 : bool, default False
            return so(3) vector instead of SO(3) matrix if `True`

        Returns
        -------
        rotation_local : torch.Tensor
            local rotation so(3) vector or SO(3) matrix

        """

        if orientations.shape[-2] == 4:
            orientations, orientations_inverse, _ = get_orientations(orientations)
        elif orientations_inverse is None:
            orientations_inverse = safe_inverse(orientations)
        if len(rotation_global.shape) == 3:
            rotation_local = torch.einsum(
                "...ij,...j->...i", orientations_inverse, rotation_global
            )
            if not return_so3:
                rotation_local = self._so3vec_to_rot(rotation_local)
        else:
            rotation_local = torch.einsum(
                "...ij,...jk->...ik", orientations_inverse, rotation_global
            )
            rotation_local = torch.einsum(
                "...ij,...jk->...ik", rotation_local, orientations
            )
            if return_so3:
                rotation_local = self._rot_to_so3vec(rotation_local)
        return rotation_local

    def _get_rotation_mean(self, x_0, x_t, timestep):
        """
        Get $/mu_{/theta}^{t-1}(x_0, x_t)$

        Parameters
        ----------
        x_0 : torch.Tensor
            initial orientation (predicted) `(B, total_L, 3, 3)`
        x_t : torch.Tensor
            orientation at step `t` `(B, total_L, 3, 3)`
        timestep : int
            timestep `t`

        """

        alpha_t = repeat(
            self.var_sched_rot.alpha_bars[timestep], "... -> ... 1 1 1"
        ).to(x_0.device)
        alpha_t_1 = repeat(
            self.var_sched_rot.alpha_bars[timestep - 1], "... -> ... 1 1 1"
        ).to(x_0.device)
        beta = repeat(self.var_sched_rot.betas[timestep], "... -> ... 1 1 1").to(
            x_0.device
        )
        a1 = self._scale_rot((torch.sqrt(alpha_t_1) * beta / (1 - alpha_t + 1e-7)), x_0)
        a2 = self._scale_rot(
            (torch.sqrt(alpha_t) * (1 - alpha_t_1) / (1 - alpha_t + 1e-7)), x_t
        )
        return torch.einsum("...ij,...jk->...ik", a1, a2)

    def _so3vec_to_so3mat(self, so3):
        """
        Convert so3 vector representation to so3 matrix representation.

        Parameters
        ----------
        so3 : torch.Tensor
            so3 vector representation `(..., 3)`

        Returns
        -------
        so3 : torch.Tensor
            so3 matrix representation `(..., 3, 3)`
        """

        zeros = torch.zeros_like(so3[..., 0])
        vx, vy, vz = so3[..., 0], so3[..., 1], so3[..., 2]
        so3 = [
            zeros,
            -vz,
            vy,
            vz,
            zeros,
            -vx,
            -vy,
            vx,
            zeros,
        ]
        so3 = torch.stack(so3, dim=-1)
        so3 = rearrange(so3, "... (n d) -> ... n d", n=3)
        return so3

    def _so3mat_to_so3vec(self, so3):
        """
        Convert so3 matrix representation to so3 vector representation.

        Parameters
        ----------
        so3 : torch.Tensor
            so3 matrix representation `(..., 3, 3)`

        Returns
        -------
        so3 : torch.Tensor
            so3 vector representation `(..., 3)`
        """

        so3 = [
            -so3[..., 1, 2],
            so3[..., 0, 2],
            -so3[..., 0, 1],
        ]
        so3 = torch.stack(so3, dim=-1)
        return so3

    def _rot_to_so3vec(self, rot):
        """
        Convert rotation matrix to so3 vector representation.
        Args:
            rot:  (..., 3, 3)
        Returns:
            so3:  (..., 3)
        """

        so3 = self._rot_log(rot)
        so3 = self._so3mat_to_so3vec(so3)
        return so3

    def _so3vec_to_rot(self, so3):
        """
        Convert so3 vector representation to rotation matrix.
        Args:
            so3:  (..., 3)
        Returns:
            rot:  (..., 3, 3)
        """

        so3 = self._so3vec_to_so3mat(so3)
        rot = self._rot_exp(so3)
        return rot

    def _rot_log(self, X):
        """
        Get the log of the rotation matrix

        Parameters
        ----------
        X : torch.Tensor
            the rotation matrix `(..., 3, 3)`

        Returns
        -------
        log : torch.Tensor
            the log of the rotation matrix `(..., 3, 3)`

        """

        cos = (torch.diagonal(X, dim1=-2, dim2=-1).sum(-1) - 1) / 2
        cos = torch.clamp(cos, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        a = repeat(theta / (2 * torch.sin(theta) + 1e-7), "... -> ... 1 1")
        log = a * (X - X.transpose(-1, -2))
        return log

    def _rot_exp(self, log):
        """
        Get the rotation matrix from the log

        Parameters
        ----------
        log : torch.Tensor
            the log of the rotation matrix `(..., 3, 3)`

        Returns
        -------
        X : torch.Tensor
            the rotation matrix `(..., 3, 3)`

        """

        v = [-log[..., 1, 2], log[..., 0, 2], -log[..., 0, 1]]
        v = torch.stack(v, dim=-1)
        theta = torch.sqrt((v**2).sum(-1) + 1e-7)
        theta = repeat(theta, "... -> ... 1 1")
        X = (
            torch.eye(3, device=log.device)
            + (torch.sin(theta) / (theta + 1e-7)) * log
            + ((1 - torch.cos(theta)) / (theta**2 + 1e-7)) * log @ log
        )
        return X

    def _scale_rot(self, alpha, X):
        """
        Scale the rotation matrix by a factor alpha in so(3) space

        Parameters
        ----------
        alpha : torch.Tensor
            the scaling factor `(..., 1)`
        X : torch.Tensor
            the rotation matrix `(..., 3, 3)`

        Returns
        -------
        X : torch.Tensor
            the scaled rotation matrix `(..., 3, 3)`

        """

        log = self._rot_log(X)
        log = alpha * log
        return self._rot_exp(log)

    def _ig_so3(self, var_idx, forward=True, device=None):
        """
        Sample from the Inverse Gamma distribution on SO(3) with identity matrix as mean

        Parameters
        ----------
        var_idx : torch.Tensor
            the variance indices of the distribution `(B, total_L)`

        Returns
        -------
        R : torch.Tensor
            the sampled rotation matrix `(B, total_L, 3, 3)`

        """

        # sample the axis
        axes = torch.normal(
            repeat(torch.zeros_like(var_idx, dtype=float), "b l -> b l 3"),
            repeat(torch.ones_like(var_idx, dtype=float), "b l -> b l 3"),
        )  # (B, total_L, 3)
        # sample the angle
        var_idx = var_idx.to(device)
        if forward:
            angles = self.angular_distrib_fwd.sample(var_idx)  # (B, total_L
        else:
            angles = self.angular_distrib_inv.sample(var_idx)  # (B, total_L)
        # make the rotation matrix
        R = rotation_around_axis(axes.to(device), angles)  # (B, total_L, 3, 3)
        return R

    def _get_v(self, x0, noise, timestep):
        a = repeat(
            torch.sqrt(self.var_sched_pos.alpha_bars[timestep]), "b -> b 1 1"
        ).to(x0.device)
        b = repeat(
            torch.sqrt(1 - self.var_sched_pos.alpha_bars[timestep]), "b -> b 1 1"
        ).to(x0.device)
        return a * noise - b * x0

    def _get_mu_t(self, x0, xt, timestep):
        alpha_bar = repeat(self.var_sched_pos.alpha_bars[timestep], "b -> b 1 1").to(
            x0.device
        )
        alpha_bar_1 = repeat(
            self.var_sched_pos.alpha_bars[timestep - 1], "b -> b 1 1"
        ).to(x0.device)
        beta = repeat(self.var_sched_pos.betas[timestep], "b -> b 1 1").to(x0.device)
        a = ((torch.sqrt(alpha_bar_1) * beta) / (1 - alpha_bar)) * x0
        b = ((torch.sqrt(1 - beta) * (1 - alpha_bar_1)) / (1 - alpha_bar)) * xt
        mu_t = a + b
        mu_t[timestep == 1] = x0[timestep == 1]
        return mu_t

    def _set_x0(self, mu_t, xt, timestep):
        if (timestep != 1).all():
            return mu_t
        alpha_bar = repeat(self.var_sched_pos.alpha_bars[timestep], "b -> b 1 1").to(
            xt.device
        )
        alpha_bar_1 = repeat(
            self.var_sched_pos.alpha_bars[timestep - 1], "b -> b 1 1"
        ).to(xt.device)
        beta = repeat(self.var_sched_pos.betas[timestep], "b -> b 1 1").to(xt.device)
        a = (torch.sqrt(alpha_bar_1) * beta) / (1 - alpha_bar)
        b = ((torch.sqrt(1 - beta) * (1 - alpha_bar_1)) / (1 - alpha_bar)) * xt
        x0 = (mu_t - b) / a
        mu_t[timestep == 1] = x0[timestep == 1]
        return mu_t

    def noise_sequence(self, seq, chain_M, timestep, inference=False):
        """
        Noise the sequence with the diffusion process

        Parameters
        ----------
        seq : torch.Tensor
            the sequence to be noised `(B, total_L)`
        chain_M : torch.Tensor
            the mask (1 for the part that should be diffused, 0 otherwise) `(B, total_L)`
        timestep : torch.Tensor
            the current timestep `(B,)`

        Returns
        -------
        seq : torch.Tensor
            the noised sequence

        """

        chain_M_bool = chain_M.bool()
        Q_bars = self.var_sched_seq.Q_bars[timestep].to(seq.device)
        if inference:
            if self.seq_diffusion_type == "mask":
                Q_bars = torch.zeros_like(Q_bars)
                Q_bars[:, :, 0] = 1
            else:
                Q_bars = torch.ones_like(Q_bars) / Q_bars.shape[-1]
        one_hot_seq = F.one_hot(seq, self.num_tokens).float()
        prob = torch.einsum("b l k, b k i -> b l i", one_hot_seq, Q_bars)
        seq[chain_M_bool] = torch.multinomial(prob[chain_M_bool], 1).squeeze(-1)
        return seq, prob

    def noise_structure(
        self,
        X,
        chain_M,
        predict_angles,
        timestep=None,
        inference=False,
        variance_scale=1,
    ):
        """
        Noise the structure with the diffusion process

        Parameters
        ----------
        X : torch.Tensor
            the structure to be noised `(B, total_L, 4, 3)`
        chain_M : torch.Tensor
            the mask (1 for the part that should be diffused, 0 otherwise) `(B, total_L)`
        predict_angles : bool
            whether to predict the angles or not
        timestep : torch.Tensor
            the current timestep `(B,)`
        inference : bool, default False
            whether to sample from the prior or not

        Returns
        -------
        X : torch.Tensor
            the translated and rotated structure `(B, total_L, 4, 3)`
        rotation : torch.Tensor
            the rotation so(3) vectors `(B, total_L, 3)`

        """

        # standardize
        chain_M_bool = chain_M.bool()
        mask = (X[:, :, 2, 0] != 0).float()
        if self.noise_around_interpolation:
            X_int = linear_interpolation(X[:, :, 2], (1 - chain_M) * mask).unsqueeze(-2)
            X = X - X_int
        if self.pos_std is not None:
            std_coords = self.pos_std * torch.ones(
                (X.shape[0], 1, 1, 3), dtype=torch.float64, device=X.device
            )
        else:
            mask_ = (mask * chain_M).unsqueeze(-1)
            std_coords = torch.sqrt(((X[:, :, 2] ** 2) * mask_).sum(1) / mask_.sum(1))
            std_coords = rearrange(std_coords, "b d -> b 1 1 d") + 1e-7
        X = X.to(dtype=torch.float64) / std_coords
        orientations, _, local_coords = get_orientations(X)

        if inference:
            alpha_bar = torch.zeros(
                (X.shape[0], 1, 1, 1), dtype=torch.float64, device=X.device
            )
        else:
            alpha_bar = repeat(
                self.var_sched_pos.alpha_bars[timestep], "b -> b 1 1 1"
            ).to(X.device, dtype=torch.float64)
        mean = torch.sqrt(alpha_bar) * X[:, :, [2], :]
        std = torch.sqrt(1 - alpha_bar)
        translation = torch.randn_like(mean)

        X_CA = mean + translation * std
        X[chain_M_bool] = repeat(X_CA[chain_M_bool], "n 1 d -> n 4 d")
        rotation_so3 = None
        if inference:
            if self.linear_interpolation:
                X_CA = linear_interpolation(X[:, :, 2], mask=(1 - chain_M) * mask)
                X_CA[(1 - mask).bool()] = 0.0
                X[chain_M_bool] = repeat(X_CA[chain_M_bool], "n d -> n 4 d")
            else:
                X[chain_M_bool] = X[chain_M_bool] * variance_scale
        if inference or not predict_angles:
            orientations_rotated = self.random_uniform_so3(
                size=(X.shape[0], X.shape[1]), device=X.device
            ).to(dtype=X.dtype)
        else:
            alpha_bar = repeat(
                self.var_sched_rot.alpha_bars[timestep], "b -> b 1 1 1"
            ).to(X.device)
            orientations_orig = orientations.clone()
            orientations_rotated_ = self._scale_rot(torch.sqrt(alpha_bar), orientations)
            rotation_matrices = self._ig_so3(
                repeat(timestep, "b -> b l", l=X.shape[1]), device=X.device
            ).to(dtype=X.dtype)
            orientations_rotated = torch.einsum(
                "b l i k, b l k d -> b l i d", rotation_matrices, orientations_rotated_
            )
            if not self.all_x0:
                log_R = self._rot_to_so3vec(rotation_matrices)  # (B, total_L, 3)
                rotation_so3 = log_R / (torch.sqrt(1 - alpha_bar.squeeze(-1)) + 1e-7)
            else:
                R = torch.einsum("b n i j, b n j k -> b n i k", orientations_rotated, safe_inverse(orientations_orig))
                R = safe_inverse(R)
                rotation_so3 = self._rot_to_so3vec(R)
        X[chain_M_bool] = (
            X[:, :, [2]]
            + torch.einsum("b n i d, b n k d -> b n k i", orientations_rotated, local_coords)
        )[chain_M_bool]

        # go back to scale
        translation = translation.to(dtype=torch.float32) * std_coords
        X = (X * std_coords).to(dtype=torch.float32)
        if rotation_so3 is not None:
            rotation_so3 = rotation_so3.to(dtype=torch.float32)
        if self.noise_around_interpolation:
            X = X + X_int

        return X, rotation_so3, translation.squeeze(-2), std_coords

    def denoise_sequence(self, seq_distribution, logits_predicted, chain_M, timestep):
        """
        Get sequence logits at step `t-1` given the model output at step `t`

        Parameters
        ----------
        seq_distribution : torch.Tensor
            the sequence distribution at step `t` `(B, total_L, 21)`
        logits_predicted : torch.Tensor
            the model output at step `t` `(B, total_L, 21)`
        chain_M : torch.Tensor
            the mask (1 for the part that should be diffused, 0 otherwise) `(B, total_L)`
        timestep : torch.Tensor
            the current timestep `(B,)`

        Returns
        -------
        seq : torch.Tensor
            the sampled sequence at step `t - 1` `(B, total_L)`

        """

        prob_predicted_0 = torch.softmax(logits_predicted, -1)
        Qs = self.var_sched_seq.Qs[timestep].to(logits_predicted.device)
        Q_bars_ = self.var_sched_seq.Q_bars[timestep - 1].to(logits_predicted.device)
        Q_bars = self.var_sched_seq.Q_bars[timestep].to(logits_predicted.device)
        all_tokens = repeat(
            torch.eye(21),
            "n j -> n b l j",
            b=seq_distribution.shape[0],
            l=seq_distribution.shape[1],
        ).to(logits_predicted.device)
        a = torch.einsum(
            "b l k, b k i -> b l i", seq_distribution, torch.transpose(Qs, -1, -2)
        )
        b = torch.einsum(
            "n b l k, b k i -> n b l i", all_tokens, torch.transpose(Q_bars_, -1, -2)
        )
        c1 = torch.einsum("b i k, b l k -> b l i", Q_bars, seq_distribution)
        c = torch.einsum("n b l k, b l k -> n b l", all_tokens, c1).unsqueeze(-1)
        a = repeat(a, "b l i -> n b l i", n=21)
        prob_posterior = a * b
        prob_posterior = prob_posterior / (c + 1e-7)
        prob_predicted = torch.einsum(
            "n b l i, b l n -> b l i", prob_posterior, prob_predicted_0
        )
        seq_distribution[chain_M.bool()] = prob_predicted[chain_M.bool()]
        if not self.no_added_diff_noise:
            pred = torch.multinomial(
                rearrange(prob_predicted, "b l f -> (b l) f"), 1
            ).squeeze(-1)
            seq = rearrange(pred, "(b l) -> b l", b=seq_distribution.shape[0])
        else:
            seq = torch.max(seq_distribution, -1)[1]
        return seq, seq_distribution

    def denoise_structure(
        self,
        coords,
        std_coords,
        orientations,
        translation_predicted,
        rotation_predicted,
        predict_angles,
        timestep,
        chain_M,
        mask,
    ):
        """
        Get structure prediction at step `t-1` given the model output at step `t`

        Parameters
        ----------
        coords : torch.Tensor
            the coordinates at step t `(B, total_L, 4, 3)`
        std_coords : torch.Tensor
            the standard deviation of the coordinates `(B, 3)`
        orientations : torch.Tensor
            the orientations at step t `(B, total_L, 3, 3)`
        translation_predicted : torch.Tensor
            the predicted translation `(B, total_L, 3)`
        rotation_predicted : torch.Tensor
            the predicted rotation `(B, total_L, 3)`
        predict_angles : bool
            whether to predict angles or not
        timestep : torch.Tensor
            the current timestep `(B,)`

        Returns
        -------
        coords_ca : torch.Tensor
            the sampled CA coordinates at step `t - 1` `(B, total_L, 3)`
        orientations : torch.Tensor
            the sampled orientation matrices at step `t - 1` `(B, total_L, 3, 3)`
        """

        # standardize
        if self.noise_around_interpolation:
            X_int = linear_interpolation(coords[:, :, 2], (1 - chain_M) * mask).unsqueeze(-2)
            coords = coords - X_int
        coords = coords.to(dtype=torch.float64) / std_coords
        if len(translation_predicted.shape) == 3:
            std_coords = std_coords.squeeze(-2)
        translation_predicted = translation_predicted.to(dtype=torch.float64)
        translation_predicted = translation_predicted / std_coords
        if orientations is not None:
            orientations = orientations.to(dtype=torch.float64)

        # translation
        sigma = repeat(self.var_sched_pos.sigmas[timestep], "b -> b 1 1").to(
            coords.device, dtype=torch.float64
        )
        if self.diff_predict == "noise":
            alpha_bar = repeat(
                self.var_sched_pos.alpha_bars[timestep], "b -> b 1 1"
            ).to(coords.device, dtype=torch.float64)
            beta = repeat(self.var_sched_pos.betas[timestep], "b -> b 1 1").to(
                coords.device, dtype=torch.float64
            )
            coords_ca = (1 / torch.sqrt(1 - beta)) * (
                coords[:, :, 2, :]
                - (beta / torch.sqrt(1 - alpha_bar)) * translation_predicted
            )
        elif self.diff_predict == "x0":
            x0 = translation_predicted[:, :, 2, :]
            xt = coords[:, :, 2, :]
            mu = self._get_mu_t(x0=x0, xt=xt, timestep=timestep)
            coords_ca = mu
        elif self.diff_predict == "mu_t":
            mu_t = translation_predicted[:, :, 2, :]
            xt = coords[:, :, 2, :]
            coords_ca = mu_t
        elif self.diff_predict == "v":
            v = translation_predicted[:, :, 2, :]
            a = repeat(self.var_sched_pos.alpha_bars[timestep], "b -> b 1 1").to(
                coords.device, dtype=torch.float64
            )
            a = torch.sqrt(a)
            b = repeat(1 - self.var_sched_pos.alpha_bars[timestep], "b -> b 1 1").to(
                coords.device, dtype=torch.float64
            )
            b = torch.sqrt(b)
            xt = coords[:, :, 2, :]
            mu = self._get_mu_t(x0=xt * a - b * v, xt=xt, timestep=timestep)
            coords_ca = mu
        if not self.no_added_diff_noise:
            coords_ca[timestep > 1] = (coords_ca + torch.randn_like(coords_ca) * sigma)[
                timestep > 1
            ]
        # rotation
        if predict_angles:
            if not self.all_x0:
                alpha_bar = repeat(
                    self.var_sched_rot.alpha_bars[timestep], "b -> b 1 1 1"
                ).to(coords.device, coords.dtype)
                beta = repeat(self.var_sched_rot.betas[timestep], "b -> b 1 1 1").to(
                    coords.device, coords.dtype
                )
                rotation_predicted = rotation_predicted * torch.sqrt(
                    1 - alpha_bar.squeeze(-1)
                )
                rotation_predicted = self._so3vec_to_rot(rotation_predicted)
                rotation_inverse = safe_inverse(rotation_predicted)
                exp = torch.einsum(
                    "b l i j, b l j k -> b l i k", rotation_inverse, orientations
                )
                x0 = self._scale_rot(1 / torch.sqrt(alpha_bar), exp)
            else:
                R = self._so3vec_to_rot(rotation_predicted).to(dtype=torch.float64)
                x0 = torch.einsum("b n i j, b n j k -> b n i k", R, orientations)
            orientations = self._get_rotation_mean(x0, orientations, timestep=timestep)
            if not self.no_added_diff_noise:
                R = self._ig_so3(
                    repeat(timestep, "b -> b l", l=coords_ca.shape[1]),
                    forward=False,
                    device=coords_ca.device,
                )
                R = R.to(coords_ca.device, dtype=coords_ca.dtype)
                orientations[timestep > 1] = torch.einsum(
                    "b l i j, b l j k -> b l i k", R, orientations
                )[timestep > 1]
        else:
            orientations = self.random_uniform_so3(size=(coords_ca.shape[0], coords_ca.shape[1]), device=coords_ca.device)

        # go back to scale
        if len(translation_predicted.shape) != 3:
            std_coords = std_coords.squeeze(-2)
        coords_ca = (coords_ca * std_coords).to(dtype=torch.float32)
        if self.noise_around_interpolation:
            coords_ca = coords_ca + X_int.squeeze(-2)
        if orientations is not None:
            orientations = orientations.to(dtype=torch.float32)

        return coords_ca, orientations

    def get_ca_loss(self, vectors_true, vectors_predicted, mask, timestep):
        """
        Get the loss for the CA coordinates

        Parameters
        ----------
        translation_true : torch.Tensor
            the true translation `(B, total_L, 3)`
        translation_predicted : torch.Tensor
            the predicted translation `(B, total_L, 3)`
        mask : torch.Tensor
            the mask (1 for the part that is diffused, 0 otherwise) `(B, total_L)`
        timestep : torch.Tensor
            the current timestep

        Returns
        -------
        loss : torch.Tensor
            the loss for the coordinates

        """

        factor = 1
        if self.weighted_diff_loss:
            lambda_t = self.var_sched_pos.alpha_bars[timestep] / (1 - self.var_sched_pos.alpha_bars[timestep])
            # lambda_t = torch.clip(
            #     self.var_sched_pos.alpha_bars[timestep]
            #     / (1 - self.var_sched_pos.alpha_bars[timestep]),
            #     min=1,
            # )
            factor = repeat(lambda_t, "b -> b 1").to(
                vectors_true.device, dtype=torch.float64
            )
        if isinstance(vectors_predicted, list):
            loss_pos = 0
            for pr, tr in zip(vectors_predicted, vectors_true):
                loss = self.get_ca_loss(tr, pr, mask, timestep)
                loss_pos += loss
            loss_pos /= len(vectors_predicted)
        else:
            diff = torch.sum(
                (vectors_predicted - vectors_true) ** 2,
                dim=-1,
            )
            loss_pos = (factor * diff * mask).sum() / (mask.sum().float() + 1e-8)
        return loss_pos

    def get_rotation_loss(self, rotation_true, rotation_predicted, mask, timestep):
        """
        Get the loss for the angles

        Parameters
        ----------
        rotation_true : torch.Tensor
            the true rotation so(3) vectors `(B, total_L, 3)`
        rotation_predicted : torch.Tensor
            the predicted rotation so(3) vectors `(B, total_L, 3)`
        mask : torch.Tensor
            the mask (1 for the part that is diffused, 0 otherwise) `(B, total_L)`
        timestep : torch.Tensor
            the current timestep

        Returns
        -------
        loss : torch.Tensor
            the loss for the angles

        """

        loss = (((rotation_true - rotation_predicted) ** 2).sum(-1) * mask).sum(
            1
        ) / mask.sum(1)
        return loss.sum()

    def get_sequence_loss(self, seq_0, seq_t, logits_predicted, mask, timestep):
        """
        Get the loss for the sequence

        Parameters
        ----------
        seq_0 : torch.Tensor
            the true sequence `(B, total_L)`
        seq_t : torch.Tensor
            the true sequence for step `t` `(B, total_L, num_tokens)`
        logits_predicted : torch.Tensor
            the predicted logits `(B, total_L, num_tokens)`
        mask : torch.Tensor
            the mask (1 for the part that is diffused, 0 otherwise) `(B, total_L)`
        timestep : int
            the current timestep

        Returns
        -------
        loss : torch.Tensor
            the loss for the sequence

        """
        loss = get_seq_loss(
            seq_0,
            logits_predicted,
            mask,
            no_smoothing=True,
            ignore_unknown=False,
            weight=0.1,
        )
        return loss


class FlowMatcher(Diffuser):
    def noise_structure(
        self,
        X,
        chain_M,
        predict_angles,
        timestep,
        inference=False,
        variance_scale=1,
        sigma=0,
    ):
        CA_start = X[:, :, 2].clone()
        std, rotation = None, None
        X_noised, *_, std = super().noise_structure(
            X=X,
            chain_M=chain_M,
            predict_angles=False,
            timestep=None,
            inference=True,
            variance_scale=variance_scale,
        )
        orientations, _, local_coords = get_orientations(X_noised)
        timestep = repeat(timestep, "b -> b 1 1").to(X.device)
        if (timestep > 1).any():
            timestep = timestep / self.num_steps
        if inference:
            mu = X_noised[:, :, 2]
            # rotation = None
        else:
            mu = (1 - timestep) * CA_start + timestep * X_noised[:, :, 2, :]
            # rotation_noised = self.random_uniform_so3(size=(X.shape[0], X.shape[1]), device=X.device).to(dtype=X.dtype)
            # rot = safe_inverse(orientations) @ rotation_noised
            # orientations = orientations @ self._scale_rot(timestep.unsqueeze(-1), rot)
            # rotation = safe_inverse(rotation_noised) @ orientations
            # rotation = self._rot_to_so3vec(rotation)
        if not inference and sigma > 0:
            mu = mu + torch.randn_like(mu) * sigma
        translation = X_noised[:, :, 2] - CA_start
        # translation = CA_new - X_start
        X_new = mu.unsqueeze(2) + torch.einsum(
            "b n i d, b n k d -> b n k i", orientations, local_coords
        )
        return X_new, rotation, translation, std

    def denoise_structure(
        self,
        coords,
        std_coords,
        orientations,
        translation_predicted,
        rotation_predicted,
        predict_angles,
        timestep,
        diff_predict=None,
    ):
        raise NotImplementedError
