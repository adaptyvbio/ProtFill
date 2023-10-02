import argparse
import os.path
import time, os
import numpy as np
import torch
import torch.nn as nn
import os.path
import warnings
import random
from proteinflow import ProteinLoader, ProteinDataset
from proteinflow.data import ProteinEntry
from protfill.model import (
    ProtFill,
)
from tqdm import tqdm
import optuna
from protfill.utils.model_utils import *
import sys
from copy import deepcopy
from math import sqrt
import wandb
from itertools import product


def get_mse_loss(att_mse, mask):
    if att_mse is not None:
        new, old, E_idx = att_mse
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        loss_att_mse = (new - old) ** 2
        return loss_att_mse[mask_attend.bool()].mean()
    return torch.zeros(1).to(mask.device)


def compute_vec_angles(vecs1, vecs2):
    inner_product = (vecs1 * vecs2).sum(dim=-1)
    cos = inner_product / (torch.norm(vecs1, dim=-1) * torch.norm(vecs2, dim=-1) + 1e-6)
    cos -= 1e-6 * torch.sign(cos)
    return torch.acos(cos)


def norm_pdf(sigma, x):
    return (1 / (sigma * sqrt(2 * torch.pi))) * torch.exp(-x / (2 * sigma**2))


def get_coords_loss(
    X_pred,
    X_true=None,
    mask=None,
    loss_type="mse",
    predict_all_atoms=False,
    skip_oxygens=False,
):
    if isinstance(X_pred, list):
        loss = {"struct": 0, "bond": 0, "angle": 0, "clash": 0}
        for i, x in enumerate(X_pred):
            loss_i = get_coords_loss(
                x,
                X_true=X_true,
                mask=mask,
                loss_type=loss_type,
                predict_all_atoms=predict_all_atoms,
                skip_oxygens=skip_oxygens,
            )
            for k, v in loss_i.items():
                loss[k] += v
        for k, v in loss.items():
            loss[k] = v / len(X_pred)
        return loss
    if predict_all_atoms:
        dims = [0, 1, 2]
        if not skip_oxygens:
            dims.append(3)
    else:
        dims = [2]
    loss = 0
    if X_true is None:
        X_true = X_pred
    if mask is None:
        mask = torch.ones_like(X_pred[:, :, 0, 0])

    bond_loss = 0
    angle_loss = 0
    clash_loss = 0

    if predict_all_atoms:
        mask_bool = mask.bool()
        mask_ = mask_bool[:, 1:] | mask_bool[:, :-1]
        # bond length loss
        c_n_len = torch.norm(X_pred[:, :-1, 1, :] - X_pred[:, 1:, 0, :], dim=-1)
        true_len = 1.32
        tol = 0.02 * 12
        with_tol = torch.abs(c_n_len - true_len)[mask_] - tol
        bond_loss = torch.relu(with_tol).mean()
        # bond angle loss
        c_ca = X_pred[:, :-1, 2, :] - X_pred[:, :-1, 1, :]
        c_n = X_pred[:, 1:, 0, :] - X_pred[:, :-1, 1, :]
        ca_c_n_angle_cos = (c_ca * c_n).sum(-1) / (
            torch.norm(c_ca, dim=-1) * torch.norm(c_n, dim=-1) + 1e-7
        )
        true_cos = -0.407
        tol = 0.2
        with_tol = torch.abs(ca_c_n_angle_cos - true_cos)[mask_] - tol
        angle_loss = torch.relu(with_tol).mean()
        # clash loss
        vdw_rad = [1.55, 1.7, 1.7, 1.52]
        tol = 1.5
        X = X_pred[:, :, 2, :]
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = torch.sqrt(torch.sum(dX**2, 3) + 1e-7)
        n = D.shape[1]
        D_adjust = D + torch.eye(n).to(D.device) * 100
        _, E_idx = torch.topk(D_adjust, 5, dim=-1, largest=False)
        X = rearrange(X_pred, "b n a d -> b n (a d)")
        x1 = rearrange(X_pred, "b n a d -> b n 1 a d")
        x2 = rearrange(gather_nodes(X, E_idx), "b n k (a d) -> b n k a d", d=3)
        ind = torch.tensor(range(n)).to(E_idx.device).unsqueeze(0).unsqueeze(-1)
        next_mask = E_idx == (ind + 1)
        prev_mask = E_idx == (ind - 1)
        clash_loss = 0
        for i, j in product(dims, repeat=2):  # skip oxygens for now
            diff = x1[..., i, :] - x2[..., j, :]
            d = torch.norm(diff, dim=-1)
            dd = 100 * torch.ones_like(d)
            if i == 0 and j == 1:  # C-N bond is ok
                dd[~prev_mask] = d[~prev_mask]
            elif i == 1 and j == 0:
                dd[~next_mask] = d[~next_mask]
            else:
                dd = d
            with_tol = (vdw_rad[i] + vdw_rad[j] - dd)[mask_bool] - tol
            clash_loss += torch.relu(with_tol).mean()

    if loss_type == "mse":
        diff = torch.sum(
            (X_pred[:, :, dims, :][mask.bool()] - X_true[:, :, dims, :][mask.bool()])
            ** 2,
            dim=-1,
        )
        loss += diff.mean()
    elif loss_type == "huber":
        delta = 1
        diff = torch.sum(
            (X_pred[:, :, dims, :][mask.bool()] - X_true[:, :, dims, :][mask.bool()])
            ** 2,
            dim=-1,
        )
        root = torch.sqrt(diff)
        struct_loss = torch.zeros_like(diff)
        greater_mask = root > delta
        struct_loss[~greater_mask] = (1 / 2) * diff[~greater_mask]
        struct_loss[greater_mask] = delta * (root[greater_mask] - delta / 2)
        loss += struct_loss.mean()
    elif loss_type == "relaxed_mse":
        scale = 0.5
        diff = torch.sum(
            (X_pred[:, :, dims, :] - X_true[:, :, dims, :]) ** 2,
            dim=-1,
        )
        coef = (norm_pdf(scale, torch.tensor(0)) - norm_pdf(scale, diff)) ** 2
        loss += (coef * diff)[mask.bool()].mean()
    else:
        raise NotImplementedError
    return {"struct": loss, "bond": bond_loss, "angle": angle_loss, "clash": clash_loss}


def print_invariance(old_coords, old_seq, model, model_args, transform=None, atol=5e-2):
    torch.manual_seed(0)
    output = model(**model_args, transform=transform)
    if "seq" in output[0]:
        new_seq = output[0]["seq"].detach().clone()
        print(
            f"  seq invariance: {torch.isclose(new_seq, old_seq, atol=atol).all().data}"
        )
    if "coords" in output[0]:
        new_coords = output[0]["coords"].detach().clone()
        base_coords = deepcopy(old_coords)
        if transform is not None:
            base_coords = transform(base_coords)
        print(
            f"  coords equivariance: {torch.isclose(new_coords, base_coords, atol=atol).all().data}"
        )
        # mask = ~torch.isclose(new_coords, base_coords, atol=atol)


def test_invariance(model, model_args, args):
    model.eval()
    model_args["chain_M"][0, 30] = 1
    torch.manual_seed(0)
    output = model(**model_args, test=args.test)
    old_seq, old_coords = None, None
    if "seq" in output[0]:
        old_seq = output[0]["seq"].detach().clone()
    if "coords" in output[0]:
        old_coords = output[0]["coords"].detach().clone()
    old_X = model_args["X"].detach().clone()
    print("\n\nTESTING INVARIANCE:")
    print("Repeat:")
    print_invariance(old_coords, old_seq, model, model_args)
    print("Translation:")
    print_invariance(old_coords, old_seq, model, model_args, transform=lambda x: x + 1)
    print("Rotation:")
    a = torch.tensor(torch.pi / 3)
    R = torch.tensor(
        [[1, 0, 0], [0, torch.cos(a), -torch.sin(a)], [0, torch.sin(a), torch.cos(a)]]
    ).to(old_X.device)
    print_invariance(
        old_coords,
        old_seq,
        model,
        model_args,
        transform=lambda x: torch.einsum("ijkl,ml->ijkm", x, R),
    )
    print("Reflection:")
    print_invariance(old_coords, old_seq, model, model_args, transform=lambda x: -x)
    print("Change:")
    print_invariance(
        old_coords,
        old_seq,
        model,
        model_args,
        transform=lambda x: torch.einsum("ijkl,ml->ijkm", x, R + 1),
    )


def initialize_sequence(seq, chain_M, seq_init_mode):
    if seq_init_mode == "zeros":
        seq[chain_M.bool()] = 0
    elif seq_init_mode == "random":
        seq[chain_M.bool()] = torch.randint(
            size=seq[chain_M.bool()].shape, low=1, high=22
        )
    return seq


def compute_loss(model_args, args, model, epoch):
    output, mask_for_loss, S, X = get_prediction(model, model_args, args, barycenter=True)

    losses = defaultdict(lambda: torch.tensor(0.0).to(args.device))
    v_w = 0.0 if epoch < args.violation_loss_start_epoch else args.violation_loss_weight
    for out in output:
        if "seq" in out:
            if model.diffusion and model.training:
                losses["seq"] += model.diffusion.get_sequence_loss(
                    seq_0=S,
                    seq_t=out["seq_t"],
                    logits_predicted=out["seq"],
                    mask=mask_for_loss,
                    timestep=out["timestep"],
                )
            else:
                losses["seq"] += get_seq_loss(
                    S,
                    out["seq"],
                    mask_for_loss,
                    no_smoothing=False,
                    ignore_unknown=False,
                )
        if "coords" in out and not model.diffusion:
            loss_upd = get_coords_loss(
                out["coords"],
                X,
                mask_for_loss,
                loss_type="mse",
                predict_all_atoms=True,
                skip_oxygens=True,
            )
            for k, v in loss_upd.items():
                losses[k] += v
        if "CA" in out and model.diffusion:
            losses["translation"] += model.diffusion.get_ca_loss(
                vectors_predicted=out["CA"],
                vectors_true=out["CA_gt"],
                mask=mask_for_loss,
                timestep=out["timestep"],
            )
        if "rotation" in out and model.diffusion:
            losses["rotation"] += model.diffusion.get_rotation_loss(
                rotation_predicted=out["rotation"],
                rotation_true=out["rotation_gt"],
                mask=mask_for_loss,
                timestep=out["timestep"],
            )
    seq_predict = output[-1].get("seq")
    coords_predict = output[-1].get("coords")
    if coords_predict is None and len(output) > 1:
        coords_predict = output[-2].get("coords")
    true_false, rmsd, pp = metrics(
        S,
        seq_predict,
        mask_for_loss,
        X,
        coords_predict,
        ignore_unknown=False,
        predict_all_atoms=True,
        predict_oxygens=False,
    )
    full_rmsd = 0
    if isinstance(rmsd, list):
        rmsd, full_rmsd = rmsd
    acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
    weights = torch.sum(mask_for_loss).cpu().data.numpy()
    total_loss = (
        losses["seq"] * args.seq_loss_weight
        + losses["struct"] * args.struct_loss_weight
        + losses.get("translation", 0) * args.struct_loss_weight
        + losses.get("rotation", 0) * args.struct_loss_weight
        + sum(
            [
                losses[k]
                for k in losses
                if k not in ["seq", "struct", "translation", "rotation"]
            ]
        )
        * v_w
    )
    return (
        total_loss,
        {k: v.detach() for k, v in losses.items()},
        rmsd,
        acc,
        pp,
        weights,
        full_rmsd,
    )


def get_model_args(batch, device):
    optional_feature_names = {
        "scalar_seq": ["chemical", "chem_topological"],
        "scalar_struct": [
            "dihedral",
            "topological",
            "mask_feature",
            "secondary_structure",
        ],
        "vector_node_seq": ["sidechain_orientation"],
        "vector_node_struct": ["backbone_orientation", "c_beta"],
        "vector_edge_seq": [],  # not implemented
        "vector_edge_struct": [],  # not implemented
    }
    model_args = {}
    model_args["chain_M"] = batch["masked_res"].to(dtype=torch.float32, device=device)
    model_args["X"] = batch["X"].to(dtype=torch.float32, device=device)
    model_args["S"] = batch["S"].to(dtype=torch.long, device=device)
    model_args["residue_idx"] = batch["residue_idx"].to(dtype=torch.long, device=device)
    model_args["chain_encoding_all"] = batch["chain_encoding_all"].to(
        dtype=torch.long, device=device
    )
    model_args["mask"] = batch["mask"].to(dtype=torch.float32, device=device)
    model_args["optional_features"] = {}
    for k, v in optional_feature_names.items():
        if k.startswith("scalar"):
            model_args["optional_features"][k] = (
                torch.cat([batch[x] for x in v if x in batch], dim=2).to(
                    dtype=torch.float32, device=device
                )
                if any([x in batch for x in v])
                else None
            )
        elif k.startswith("vector"):
            model_args["optional_features"][k] = None
            to_stack = []
            for x in v:
                if x in batch:
                    to_stack.append(batch[x])
                    if len(to_stack[-1].shape) == 3:
                        to_stack[-1] = to_stack[-1].unsqueeze(2)
            if len(to_stack) > 0:
                model_args["optional_features"][k] = torch.cat(to_stack, dim=2).to(
                    dtype=torch.float32, device=device
                )
    return model_args


def get_prediction(model, model_args, args, chain_dict=None, barycenter=False):
    mask = model_args["mask"]
    model_args["X"][~mask.bool()] = 0.0
    chain_M = model_args["chain_M"]
    mask_for_loss = mask * chain_M
    S = model_args["S"]
    X = deepcopy(model_args["X"])

    mu = []
    for i in range(model_args["X"].shape[0]):
        anchor_ind = ProteinDataset.get_anchor_ind(model_args["chain_M"][i], model_args["mask"][i])
        mu.append(model_args["X"][i][[int(x) for x in anchor_ind], 2].mean(dim=0))
    mu = repeat(torch.stack(mu, dim=0), "b d -> b 1 1 d")
    model_args["X"] = model_args["X"] - mu
    model_args["X"][~model_args["mask"].bool()] = 0.0

    if args.diffusion and not model.training:
        if args.predict_file:
            entry_name = os.path.basename(args.predict_file).split(".")[0]
            save_path = os.path.join("output", entry_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = None
        output = model.diffuse(
            **deepcopy(model_args),
            test=True,
            save_path=save_path,
            chain_dict=chain_dict,
        )
    else:
        output = model(**model_args, test=args.test)

    for i, out in enumerate(output):
        if "coords" in out:  # and args.diffusion:
            if isinstance(output[i]["coords"], list):
                output[i]["coords"] = [x + mu for x in output[i]["coords"]]
            else:
                output[i]["coords"] = output[i]["coords"] + mu
    return output, mask_for_loss, S, X


def get_loss(batch, optimizer, args, model, epoch):
    device = args.device
    model_args = get_model_args(batch, device)

    optimizer.zero_grad()

    if not args.no_mixed_precision:
        with torch.cuda.amp.autocast():
            (
                loss,
                losses_sep,
                rmsd,
                acc,
                pp,
                weights,
                full_rmsd,
            ) = compute_loss(model_args, args, model, epoch)
    else:
        (
            loss,
            losses_sep,
            rmsd,
            acc,
            pp,
            weights,
            full_rmsd,
        ) = compute_loss(model_args, args, model, epoch)

    return (
        loss,
        losses_sep,
        acc,
        rmsd,
        pp,
        weights,
        full_rmsd,
    )


def run(args, trial=None):
        
    scaler = torch.cuda.amp.GradScaler()

    args.device = torch.device(args.device)

    base_folder = time.strftime(args.output_path, time.localtime())

    if base_folder[-1] != "/":
        base_folder += "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ["model_weights"]
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.load_experiment or ""

    logfile = base_folder + "log.txt"
    if not PATH:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    LOAD_PARAM = {
        "batch_size": args.batch_size,
        "shuffle_batches": True,
        "pin_memory": False,
        "num_workers": 4,
    }

    DATA_PARAM = {
        "features_folder": args.features_path,
        "max_length": args.max_protein_length,
        "rewrite": True,
        "debug": args.debug,

        "min_cdr_length": 3,
        "lower_limit": args.lower_masked_limit,
        "upper_limit": args.upper_masked_limit,
        "mask_all_cdrs": args.mask_all_cdrs,
        "patch_around_mask": args.patch_around_mask,
        "initial_patch_size": args.initial_patch_size,
    }

    if not os.path.exists(os.path.join(args.dataset_path, "splits_dict")):
        args.skip_clustering = True

    training_dict = os.path.join(args.dataset_path, "splits_dict", "train.pickle")
    validation_dict = os.path.join(args.dataset_path, "splits_dict", "valid.pickle")
    test_dict = os.path.join(args.dataset_path, "splits_dict", "test.pickle")
    excluded_dict = os.path.join(args.dataset_path, "splits_dict", "excluded.pickle")

    print("\nDATA LOADING")
    use_frac = 1.
    if args.test:
        if args.test_excluded:
            folder, clustering_dict_path = "excluded", excluded_dict
        elif args.validate:
            folder, clustering_dict_path = "valid", validation_dict
        else:
            folder, clustering_dict_path = "test", test_dict
        test_set = ProteinDataset(
            dataset_folder=os.path.join(args.dataset_path, folder),
            clustering_dict_path=clustering_dict_path,
            shuffle_clusters=False,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        test_set.set_cdr(args.test_cdr)
        test_loader = ProteinLoader(test_set, **LOAD_PARAM)
    elif args.predict_file is not None:
        test_set = ProteinDataset(
            dataset_folder=os.path.join(args.dataset_path, "test"),
            debug_file_path=args.predict_file,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        test_set.set_cdr(args.test_cdr)
        test_loader = ProteinLoader(test_set, **LOAD_PARAM)
    else:
        train_set = ProteinDataset(
            dataset_folder=os.path.join(args.dataset_path, "train"),
            clustering_dict_path=training_dict,
            use_fraction=use_frac,
            shuffle_clusters=True,
            force_binding_sites_frac=args.train_force_binding_sites_frac,
            **DATA_PARAM,
        )
        train_loader = ProteinLoader(train_set, **LOAD_PARAM)
        # valid_set = train_set
        valid_set = ProteinDataset(
            dataset_folder=os.path.join(args.dataset_path, "valid"),
            clustering_dict_path=validation_dict,
            shuffle_clusters=False,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        valid_loader = ProteinLoader(valid_set, **LOAD_PARAM)

    model = ProtFill(
        args,
        encoder_type=args.message_passing,
        decoder_type=args.message_passing,
        k_neighbors=args.num_neighbors,
        noise_unknown=args.noise_std,
        n_cycles=args.n_cycles,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.hidden_dim,
    )
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(">>> Number of parameters in the model: ", params)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint["step"]  # write total_step from the checkpoint
        epoch = checkpoint["epoch"]  # write epoch from the checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(filter(lambda p: p.requires_grad, model.parameters()), args.hidden_dim, total_step, lr=None)

    if PATH:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            warnings.warn(str(e))

    if args.predict_file is not None:
        print("\nGENERATING")
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            model_args = get_model_args(batch, args.device)
            if args.predict_positions is not None:
                if batch["masked_res"].shape[0] != 1:
                    raise NotImplementedError
                if ":" in args.predict_positions:
                    chain, positions = args.predict_positions.split(":")
                    positions = positions.split(",")
                    if positions[0] == "":
                        positions = []
                else:
                    chain, positions = args.predict_positions, []
                chain_M = torch.zeros_like(batch["masked_res"])
                chain_mask = (
                    batch["chain_encoding_all"] == batch["chain_dict"][0][chain]
                )
                if len(positions) > 0:
                    chain_start = torch.where(chain_mask)[1][0]
                    for pos in positions:
                        if len(pos.split("-")) == 2:
                            start, end = pos.split("-")
                            chain_M[
                                :, int(start) + chain_start : int(end) + chain_start
                            ] = 1
                        else:
                            chain_M[:, int(pos) + chain_start] = 1
                else:
                    chain_M[chain_mask] = 1
                model_args["chain_M"] = chain_M.to(args.device)
            if args.num_predictions > 1 and args.decoder_type != "mpnn_auto":
                model.train()
            for i in range(args.num_predictions // args.batch_size + 1):
                b = min(args.batch_size, args.num_predictions - i * args.batch_size)
                if b <= 0:
                    continue
                m_args = {}
                for k, v in model_args.items():
                    if k == "chain_dict":
                        pass
                    if isinstance(v, torch.Tensor):
                        m_args[k] = repeat(v, "n ... -> (b n) ...", b=b).clone()
                    else:
                        m_args[k] = {}
                        for kk, vv in v.items():
                            if vv is not None:
                                m_args[k][kk] = repeat(
                                    vv, "n ... -> (b n) ...", b=b
                                ).clone()
                            else:
                                m_args[k][kk] = None

                output, mask_for_loss, *_ = get_prediction(
                    model, m_args, args, chain_dict=batch["chain_dict"], barycenter=args.barycenter
                )
                out = output[-1]
                true_protein_entry = ProteinEntry.from_arrays(
                    batch["S"][0],
                    batch["X"][0],
                    batch["mask"][0],
                    batch["chain_dict"][0],
                    batch["chain_encoding_all"][0],
                    mask_for_loss[0],
                    batch.get("cdr", [None for _ in range(b)])[0],
                )
                basename = os.path.basename(args.predict_file)
                if not os.path.exists(os.path.join("output", basename.split(".")[0])):
                    os.makedirs(os.path.join("output", basename.split(".")[0]))
                true_protein_entry.to_pickle(
                    os.path.join("output", basename.split(".")[0], f"true.pickle")
                )
                for j in range(b):
                    mask_ = mask_for_loss[j].bool()
                    coords_ = batch["X"][j].float()
                    coords_[mask_] = out.get("coords", batch["X"])[j][mask_].to(
                        coords_.device
                    )
                    seq_ = batch["S"][j].long()
                    if "seq" in out:
                        seq_[mask_] = torch.argmax(out["seq"][j], dim=-1)[mask_].to(
                            seq_.device
                        )
                    predicted_protein_entry = ProteinEntry.from_arrays(
                        seq_,
                        coords_,
                        batch["mask"][j],
                        batch["chain_dict"][j],
                        batch["chain_encoding_all"][j],
                        mask_for_loss[j],
                        batch.get("cdr", [None for _ in range(b)])[j],
                    )
                    filepath = os.path.join(
                        "output",
                        basename.split(".")[0],
                        f"predicted_{i * args.batch_size + j}.pickle",
                    )
                    predicted_protein_entry.to_pickle(filepath)

    elif args.test:
        print("\nTESTING")
        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            valid_rmsd = 0.0
            valid_rmsd_full = 0.0
            valid_pp = 0.0
            valid_losses = defaultdict(float)
            if args.skip_tqdm:
                loader = test_loader
            else:
                loader = tqdm(test_loader)
            for batch in loader:
                (
                    loss,
                    losses_sep,
                    acc,
                    rmsd,
                    pp,
                    weights,
                    full_rmsd,
                ) = get_loss(batch, optimizer, args, model, 0)
                validation_sum += loss.detach()
                validation_acc += acc
                valid_rmsd += rmsd
                valid_rmsd_full += full_rmsd
                valid_pp += pp
                validation_weights += weights
                for k, v in losses_sep.items():
                    valid_losses[k] += v

        validation_accuracy = validation_acc / validation_weights
        valid_rmsd = valid_rmsd / len(test_set)
        valid_rmsd_full = valid_rmsd_full / len(test_set)
        valid_pp = valid_pp / len(test_set)
        validation_loss = float(validation_sum / len(test_set))
        valid_losses = {k: v / len(test_set) for k, v in valid_losses.items()}

        epoch_string = f"[test], loss: {validation_loss:.2e}"
        if validation_accuracy > 0:
            epoch_string += (
                f", test_acc: {validation_accuracy:.2f}, test_pp: {valid_pp:.2f}"
            )
        if valid_rmsd > 0:
            epoch_string += f", test_rmsd: {valid_rmsd:.2f}"
        if valid_rmsd_full > 0:
            epoch_string += f", test_rmsd_bb: {valid_rmsd_full:.2f}"
        for k, v in valid_losses.items():
            if v > 0:
                epoch_string += f", test_{k}_loss: {v:.2e}"
        epoch_string += "\n"
        print(epoch_string)

    else:
        print("\nTRAINING")

        best_res = 0
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0.0, 0.0
            train_losses = defaultdict(float)
            train_acc = 0.0
            train_rmsd = 0.0
            train_rmsd_full = 0.0
            train_pp = 0.0
            loader = tqdm(train_loader)
            for batch in loader:
                try:
                    (
                        loss,
                        losses_sep,
                        acc,
                        rmsd,
                        pp,
                        weights,
                        full_rmsd,
                    ) = get_loss(batch, optimizer, args, model, e)
                except RuntimeError as e:
                    if "Test over" in str(e):
                        return
                    else:
                        raise e
                if not args.no_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_sum += loss.detach()
                for k, v in losses_sep.items():
                    train_losses[k] += v
                train_acc += acc
                train_weights += weights
                train_rmsd += rmsd
                train_rmsd_full += full_rmsd
                train_pp += pp

                total_step += 1

            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            valid_rmsd = 0.0
            valid_rmsd_full = 0.0
            valid_pp = 0.0
            valid_losses = defaultdict(float)
            validation_accuracy, validation_loss, valid_rmsd, valid_rmsd_full = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            if (e + 1) % args.validate_every_n_epochs == 0:
                model.eval()
                with torch.no_grad():
                    if args.skip_tqdm:
                        loader = valid_loader
                    else:
                        loader = tqdm(valid_loader)
                    for batch in loader:
                        (
                            loss,
                            losses_sep,
                            acc,
                            rmsd,
                            pp,
                            weights,
                            full_rmsd,
                        ) = get_loss(batch, optimizer, args, model, e)
                        validation_sum += loss.detach()
                        validation_acc += acc
                        valid_rmsd += rmsd
                        valid_rmsd_full += full_rmsd
                        valid_pp += pp
                        validation_weights += weights
                        for k, v in losses_sep.items():
                            valid_losses[k] += v

                    validation_accuracy = validation_acc / validation_weights
                    valid_rmsd = valid_rmsd / len(valid_set)
                    valid_rmsd_full = valid_rmsd_full / len(valid_set)
                    valid_pp = valid_pp / len(valid_set)
                    validation_loss = float(validation_sum / len(valid_set))
                    valid_losses = {
                        k: v / len(valid_set) for k, v in valid_losses.items()
                    }

            train_accuracy = train_acc / train_weights
            train_rmsd = train_rmsd / len(train_set)
            train_rmsd_full = train_rmsd_full / len(train_set)
            train_pp = train_pp / len(train_set)
            train_loss = float(train_sum / len(train_set))
            train_losses = {k: v / len(train_set) for k, v in train_losses.items()}

            t1 = time.time()
            dt = np.format_float_positional(
                np.float32(t1 - t0), unique=False, precision=1
            )
            epoch_string = (
                f"epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss:.2e}"
            )
            if e + 1 % args.validate_every_n_epochs == 0:
                epoch_string += f", valid: {validation_loss:.2e}"
            if train_accuracy > 0:
                epoch_string += (
                    f", train_acc: {train_accuracy:.2f}, train_pp: {train_pp:.2f}"
                )
            if validation_accuracy > 0:
                epoch_string += (
                    f", valid_acc: {validation_accuracy:.2f}, valid_pp: {valid_pp:.2f}"
                )
            if train_rmsd > 0:
                epoch_string += f", train_rmsd: {train_rmsd:.2f}"
            if valid_rmsd > 0:
                epoch_string += f", valid_rmsd: {valid_rmsd:.2f}"
            if train_rmsd_full > 0:
                epoch_string += f", train_rmsd_bb: {train_rmsd_full:.2f}"
            if valid_rmsd_full > 0:
                epoch_string += f", valid_rmsd_bb: {valid_rmsd_full:.2f}"
            if len(train_losses) > 1:
                for k, v in train_losses.items():
                    if v > 0:
                        epoch_string += f", train_{k}_loss: {v:.2e}"
            if len(valid_losses) > 1:
                for k, v in valid_losses.items():
                    if v > 0:
                        epoch_string += f", valid_{k}_loss: {v:.2e}"
            epoch_string += "\n"

            with open(logfile, "a") as f:
                f.write(epoch_string)
            print(epoch_string)
            if args.log_wandb_name is not None:
                res = {
                    "epoch": e + 1,
                    "train_loss": train_loss,
                }
                if (e + 1) % args.validate_every_n_epochs == 0:
                    res["valid_loss"] = validation_loss
                if train_rmsd > 0:
                    res["train_rmsd"] = train_rmsd
                if train_rmsd_full > 0:
                    res["train_rmsd_bb"] = train_rmsd_full
                if valid_rmsd > 0:
                    res["valid_rmsd"] = valid_rmsd
                if valid_rmsd_full > 0:
                    res["valid_rmsd_bb"] = valid_rmsd_full
                if train_accuracy > 0:
                    res["train_pp"] = train_pp
                    res["train_acc"] = train_accuracy
                if validation_accuracy > 0:
                    res["valid_pp"] = valid_pp
                    res["valid_acc"] = validation_accuracy
                for k, v in train_losses.items():
                    if v > 0:
                        res[f"train_{k}_loss"] = v
                for k, v in valid_losses.items():
                    if v > 0:
                        res[f"valid_{k}_loss"] = v
                wandb.log(res)

            checkpoint_filename_last = (
                base_folder + "model_weights/epoch_last.pt".format(e + 1, total_step)
            )
            torch.save(
                {
                    "epoch": e + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_filename_last,
            )

            best_epoch = False
            if not args.predict_structure or args.co_design != "none":
                if validation_accuracy > best_res:
                    best_epoch = True
                    best_res = validation_accuracy
            elif valid_rmsd < best_res:
                best_epoch = True
                best_res = valid_rmsd
            if best_epoch:
                checkpoint_filename_best = (
                    base_folder
                    + "model_weights/epoch_best.pt".format(e + 1, total_step)
                )
                torch.save(
                    {
                        "epoch": e + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_filename_best,
                )
            if trial is not None:  # optuna trial for hyperparameter optimization
                trial.report(best_res, e)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if (e + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = (
                    base_folder
                    + "model_weights/epoch{}_step{}.pt".format(e + 1, total_step)
                )
                torch.save(
                    {
                        "epoch": e + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_filename,
                )

        if args.log_wandb_name is not None:
            wandb.finish()
        return best_res


def parse(command=None):
    if command is not None:
        sys.argv = command.split()
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/proteinflow_20230102_stable",
        help="path for loading training data (a folder with training, test and validation subfolders)",
    )
    argparser.add_argument(
        "--features_path",
        type=str,
        default="./data/tmp_features",
        help="path where ProteinMPNN features will be saved",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default="./exp",
        help="path for logs and model weights",
    )
    argparser.add_argument(
        "--load_experiment",
        type=str,
        default=None,
        help="path for previous model weights, e.g. file.pt",
    )
    argparser.add_argument(
        "--load_epoch_mode",
        choices=["last", "best"],
        default="last",
        help="the mode for loading the model weights",
    )
    argparser.add_argument(
        "--num_epochs", type=int, default=100, help="number of epochs to train for"
    )
    argparser.add_argument(
        "--batch_size", type=int, default=8, help="number of tokens for one batch"
    )
    argparser.add_argument(
        "--max_protein_length",
        type=int,
        default=2000,
        help="maximum length of the protein complex",
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="The name of the torch device"
    )
    argparser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set instead of training (make sure to set previous_checkpoint)",
    )
    argparser.add_argument(
        "--test_excluded",
        action="store_true",
        help="Evaluate on the excluded set instead of training (make sure to set previous_checkpoint)",
    )
    argparser.add_argument(
        "--validate",
        action="store_true",
        help="Evaluate on the validation set instead of training (make sure to set previous_checkpoint)",
    )
    argparser.add_argument(
        "--noise_std",
        default=None,
        type=float,
        help="The noise standard deviation",
    )
    argparser.add_argument(
        "--n_cycles",
        default=1,
        type=int,
        help="Number of refinement cycles (1 = only prediction, no refinement)",
    )
    argparser.add_argument(
        "--message_passing",
        choices=[
            "gvp",
            "gvp_orig",
        ],
        default="mpnn_auto",
    )
    argparser.add_argument(
        "--predict_file",
        help="Predict the given file",
    )
    argparser.add_argument(
        "--predict_positions",
        help="Mask specific positions in the given file (only with predict_file); e.g. A, A:5-10,20,30-40 (fasta-based numbering)",
    )
    argparser.add_argument(
        "--linear_layers_num",
        type=int,
        default=0,
        help="The number of linear graph layers to use in the decoder (GVP)",
    )
    argparser.add_argument(
        "--diffusion",
        action="store_true",
        help="Use diffusion",
    )
    argparser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    argparser.add_argument(
        "--mask_all_cdrs",
        action="store_true",
        help="Mask all CDRs",
    )
    argparser.add_argument(
        "--test_cdr",
        choices=["L1", "L2", "L3", "H1", "H2", "H3"],
        default=None,
        help="Test on a single CDR",
    )
    argparser.add_argument(
        "--initial_patch_size",
        type=int,
        default=128,
        help="Initial patch size for patching",
    )
    argparser.add_argument(
        "--alternative_noising",
        action="store_true",
        help="Add noise to coordinates instead of replacing them",
    )

    argparser.add_argument(
        "--debug",
        action="store_true",
    )

    args = argparser.parse_args()

    args.no_mixed_precision = True
    if args.test_excluded or args.validate:
        args.test = True
    args.patch_around_mask = not args.predict_file

    args.use_edge_vectors = True
    # args.use_node_dropout = True
    # args.less_dropout = False
    args.no_oxygen_features = True

    args.validate_every_n_epochs = 10
    args.save_model_every_n_epochs = 10

    args.hidden_dim = 128
    args.num_encoder_layers = 3
    args.num_decoder_layers = 3
    args.num_encoder_mpnn_layers = 1
    args.num_decoder_mpnn_layers = 1
    args.num_neighbors = 32
    args.dropout = 0.2
    args.struct_loss_weight = 1.
    args.seq_loss_weight = 1.
    args.violation_loss_weight = 0.
    args.violation_loss_start_epoch = 0
    args.no_added_diff_noise = True

    args.train_force_binding_sites_frac = 0.5
    args.val_force_binding_sites_frac = 1.
    args.lower_masked_limit = 15
    args.upper_masked_limit = 50

    return args


def main():
    args = parse()
    run(args)


if __name__ == "__main__":
    main()
