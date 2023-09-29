import argparse
import os.path
import time, os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os.path
import warnings
import random
from protfill.utils.data import CustomDataset

# from proteinflow.data.torch import ProteinLoader
from proteinflow import ProteinLoader, ProteinDataset
from proteinflow.data import ProteinEntry
from protfill.model import (
    ProteinModel,
)
from tqdm import tqdm
from protfill.utils.aws_utils import get_trained_model_path
import optuna
from protfill.utils.model_utils import *
import sys
from copy import deepcopy
from math import sqrt
import wandb
import pickle
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
    output, mask_for_loss, S, X = get_prediction(model, model_args, args, barycenter=args.barycenter)

    losses = defaultdict(lambda: torch.tensor(0.0).to(args.device))
    v_w = 0.0 if epoch < args.violation_loss_epoch else args.violation_loss_weight
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
                    no_smoothing=args.no_smoothing,
                    ignore_unknown=args.ignore_unknown_residues,
                )
        if "coords" in out and not model.diffusion:
            loss_upd = get_coords_loss(
                out["coords"],
                X,
                mask_for_loss,
                loss_type=args.struct_loss_type,
                predict_all_atoms=args.predict_angles,
                skip_oxygens=not args.predict_oxygens,
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
        ignore_unknown=args.ignore_unknown_residues,
        predict_all_atoms=args.predict_angles,
        predict_oxygens=args.predict_oxygens,
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
    if args.ignore_unknown_residues:
        mask_for_loss *= S != 0

    if args.test_invariance:
        test_invariance(model, model_args, args)
        raise RuntimeError("Test over")

    # if args.diffusion:
    if barycenter:
        mu = []
        for i in range(model_args["X"].shape[0]):
            anchor_ind = ProteinDataset.get_anchor_ind(model_args["chain_M"][i], model_args["mask"][i])
            mu.append(model_args["X"][i][[int(x) for x in anchor_ind], 2].mean(dim=0))
        mu = repeat(torch.stack(mu, dim=0), "b d -> b 1 1 d")
    else:
        mask_ = model_args["chain_M"] * model_args["mask"]
        mask_ = rearrange(mask_, "b l -> b l 1")
        mu = torch.sum(model_args["X"][:, :, 2] * mask_, dim=1) / torch.sum(mask_, dim=1)
        mu = repeat(mu, "b d -> b 1 1 d")
    model_args["X"] = model_args["X"] - mu
    model_args["X"][~model_args["mask"].bool()] = 0.0

    if args.diffusion and not model.training:
        if args.skip_diffuse:
            output = model(**model_args, test=args.test)
        else:
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
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
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

    PATH = ""
    if args.load_experiment is not None:
        PATH = get_trained_model_path(args.load_experiment, args.load_epoch_mode)

    logfile = base_folder + "log.txt"
    if not PATH:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    LOAD_PARAM = {
        "batch_size": args.batch_size,
        "shuffle_batches": True,
        "pin_memory": False,
        "num_workers": 4,
        
        # "lower_limit": args.lower_masked_limit,
        # "upper_limit": args.upper_masked_limit,
        # "mask_whole_chains": args.mask_whole_chains,
        # "force_binding_sites_frac": args.train_force_binding_sites_frac,
        # "mask_frac": args.masking_probability,
        # "mask_all_cdrs": args.mask_all_cdrs,
    }

    DATA_PARAM = {
        "features_folder": args.features_path,
        "max_length": args.max_protein_length,
        "rewrite": True,
        "debug": args.debug,
        "load_to_ram": args.load_to_ram,
        "interpolate": args.interpolate,
        "node_features_type": args.node_features,

        "min_cdr_length": 3,
        "lower_limit": args.lower_masked_limit,
        "upper_limit": args.upper_masked_limit,
        "mask_whole_chains": args.mask_whole_chains,
        "mask_frac": args.masking_probability,
        "mask_all_cdrs": args.mask_all_cdrs,
        "patch_around_mask": args.patch_around_mask,
        "initial_patch_size": args.initial_patch_size,
        "mask_sequential": args.mask_sequential,
    }

    if not os.path.exists(os.path.join(args.dataset_path, "splits_dict")):
        args.skip_clustering = True

    training_dict = (
        None
        if args.skip_clustering
        else os.path.join(args.dataset_path, "splits_dict", "train.pickle")
    )
    validation_dict = (
        None
        if args.skip_clustering
        else os.path.join(args.dataset_path, "splits_dict", "valid.pickle")
    )
    test_dict = (
        None
        if args.skip_clustering
        else os.path.join(args.dataset_path, "splits_dict", "test.pickle")
    )
    excluded_dict = (
        None
        if args.skip_clustering
        else os.path.join(args.dataset_path, "splits_dict", "excluded.pickle")
    )

    print("\nDATA LOADING")
    if args.tiny_dataset:
        use_frac = 0.02
    elif args.small_dataset:
        use_frac = 0.1
    elif args.half_dataset:
        use_frac = 0.5
    else:
        use_frac = 1.0
    if args.debug_file:
        train_set = CustomDataset(
            dataset_folder=os.path.join(args.dataset_path, "train"),
            debug_file_path=args.debug_file,
            force_binding_sites_frac=args.train_force_binding_sites_frac,
            **DATA_PARAM,
        )
        train_loader = ProteinLoader(train_set, **LOAD_PARAM)
        valid_set = CustomDataset(
            dataset_folder=os.path.join(args.dataset_path, "valid"),
            debug_file_path=args.debug_file,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        valid_loader = ProteinLoader(valid_set, **LOAD_PARAM)
    elif args.test:
        if args.test_excluded:
            folder, clustering_dict_path = "excluded", excluded_dict
        elif args.validate:
            folder, clustering_dict_path = "valid", validation_dict
        else:
            folder, clustering_dict_path = "test", test_dict
        test_set = CustomDataset(
            dataset_folder=os.path.join(args.dataset_path, folder),
            clustering_dict_path=clustering_dict_path,
            shuffle_clusters=False,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        test_set.set_cdr(args.test_cdr)
        test_loader = ProteinLoader(test_set, **LOAD_PARAM)
    elif args.predict_file is not None:
        test_set = CustomDataset(
            dataset_folder=os.path.join(args.dataset_path, "test"),
            debug_file_path=args.predict_file,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        test_set.set_cdr(args.test_cdr)
        test_loader = ProteinLoader(test_set, **LOAD_PARAM)
    else:
        train_set = CustomDataset(
            dataset_folder=os.path.join(args.dataset_path, "train"),
            clustering_dict_path=training_dict,
            use_fraction=use_frac,
            shuffle_clusters=not args.not_shuffle_clusters,
            force_binding_sites_frac=args.train_force_binding_sites_frac,
            **DATA_PARAM,
        )
        train_loader = ProteinLoader(train_set, **LOAD_PARAM)
        # valid_set = train_set
        valid_set = CustomDataset(
            dataset_folder=os.path.join(args.dataset_path, "valid"),
            clustering_dict_path=validation_dict,
            shuffle_clusters=False,
            force_binding_sites_frac=args.val_force_binding_sites_frac,
            **DATA_PARAM,
        )
        valid_loader = ProteinLoader(valid_set, **LOAD_PARAM)

    model = ProteinModel(
        args,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        k_neighbors=args.num_neighbors,
        augment_eps=args.backbone_noise,
        embedding_dim=args.embedding_dim,
        ignore_unknown=args.ignore_unknown_residues,
        node_features_type=args.node_features,
        predict_structure=args.predict_structure,
        noise_unknown=args.noise_unknown,
        n_cycles=args.n_cycles,
        no_sequence_in_encoder=args.no_sequence_in_encoder,
        cycle_over_embedding=args.cycle_over_embedding,
        seq_init_mode=args.seq_init_mode,
        double_sequence_features=args.double_sequence_features,
        hidden_dim=args.hidden_dim,
        separate_modules_num=args.separate_modules_num,
        only_cycle_over_decoder=args.only_cycle_over_decoder,
        random_connections_frac=args.random_connections_frac,
        skip_weight_init=args.skip_weight_init,
        predict_angles=args.predict_angles,
        co_design=args.co_design,
    )
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if args.log_wandb_name is not None:
        wandb.init(project="ML4", config=args, name=args.log_wandb_name, entity="adaptyvbio")
        wandb.watch(model)
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

    if args.freeze_structure and args.co_design == "seq":
        for name, param in model.named_parameters():
            name_split = name.split(".")
            if (
                name_split[0] in ["encoders", "decoders"]
                and int(name_split[1]) % 2 == 1
            ):
                continue
            if "W_out" in name:
                continue
            param.requires_grad = False

    optimizer = get_std_opt(filter(lambda p: p.requires_grad, model.parameters()), args.hidden_dim, total_step, lr=args.lr)

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
        if not args.test_invariance:
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
            if args.skip_tqdm or args.test_invariance:
                loader = train_loader
            else:
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
        "--skip_clustering",
        action="store_true",
        help="skip using sequence identity clusters",
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
        "--save_model_every_n_epochs",
        type=int,
        default=10,
        help="save model weights every n epochs",
    )
    argparser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=1,
        help="validate model every n epochs",
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
        "--hidden_dim", type=int, default=128, help="hidden model dimension"
    )
    argparser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="number of encoder layers"
    )
    argparser.add_argument(
        "--num_decoder_layers", type=int, default=3, help="number of decoder layers"
    )
    argparser.add_argument(
        "--num_encoder_mpnn_layers",
        type=int,
        default=1,
        help="Number of stacked GVPs to use in one aggregation GVP layer inside the encoder (need encoder_type='gvp' to be used)",
    )
    argparser.add_argument(
        "--num_decoder_mpnn_layers",
        type=int,
        default=1,
        help="Number of stacked GVPs to use in one aggregation GVP layer inside the decoder (need decoder_type='gvp' to be used)",
    )
    argparser.add_argument(
        "--num_neighbors",
        type=int,
        default=32,
        help="number of neighbors for the sparse graph",
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout"
    )
    argparser.add_argument(
        "--gradient_norm",
        type=float,
        default=-1.0,
        help="clip gradient norm, set to negative to omit clipping",
    )
    argparser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="train without mixed precision",
    )
    argparser.add_argument(
        "--violation_loss_epoch",
        type=int,
        default=0,
        help="epoch after which to start using the violation loss",
    )

    # New parameters
    argparser.add_argument(
        "--lower_masked_limit",
        type=int,
        default=15,
        help="The minimum number of residues to mask in each protein (ignored if mask_whole_chains is true or masking_probability is specified)",
    )
    argparser.add_argument(
        "--upper_masked_limit",
        type=int,
        default=100,
        help="The maximum number of residues to mask in each protein (ignored if mask_whole_chains is true or masking_probability is specified)",
    )
    argparser.add_argument(
        "--masking_probability",
        type=float,
        help="The probability of masking a residue (if specified, overrides lower_masked_limit and upper_masked_limit)",
    )
    argparser.add_argument(
        "--mask_whole_chains",
        action="store_true",
        help="if true, lower_masked_limit, upper_masked_limit, masking_probability are ignored and entire chains are masked",
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="The name of the torch device"
    )
    argparser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="The dimension of the residue type embedding",
    )
    argparser.add_argument(
        "--no_smoothing",
        action="store_true",
        help="Use a regular cross-entropy loss without smooting the one-hot encoding",
    )
    argparser.add_argument(
        "--use_mean",
        action="store_true",
        help="Use mean over existing neighbors for aggregation instead of sum",
    )
    argparser.add_argument(
        "--use_egnn_for_refinement",
        action="store_true",
        help="Use an additional EGNN pre-processing layer for sructure refinement",
    )
    argparser.add_argument(
        "--predict_structure",
        action="store_true",
        help="Predict the structure of the protein instead of the sequence",
    )
    argparser.add_argument(
        "--predict_angles",
        action="store_true",
        help="Predict the frame orientations",
    )
    argparser.add_argument(
        "--small_dataset", action="store_true", help="Use 0.1 of the training clusters"
    )
    argparser.add_argument(
        "--tiny_dataset", action="store_true", help="Use 0.03 of the training clusters"
    )
    argparser.add_argument(
        "--half_dataset", action="store_true", help="Use 0.5 of the training clusters"
    )
    argparser.add_argument(
        "--train_force_binding_sites_frac",
        type=float,
        default=0.15,
        help="If > 0, this fraction of regions sampled in polymer chains will be forced to be around the binding sites (in training)",
    )
    argparser.add_argument(
        "--val_force_binding_sites_frac",
        type=float,
        default=0.15,
        help="If > 0, this fraction of regions sampled in polymer chains will be forced to be around the binding sites (in validation)",
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
        "--ignore_unknown_residues",
        action="store_true",
        help="Predict 20 aminoacids; if the residue type is unknown in the ground truth, mask it in loss calculation",
    )
    argparser.add_argument(
        "--debug", action="store_true", help="Only process 1000 files per subset"
    )
    argparser.add_argument(
        "--load_to_ram",
        action="store_true",
        help="Load the data to RAM (use with caution! make sure the RAM is big enough!)",
    )
    argparser.add_argument(
        "--interpolate",
        choices=["none", "only_middle", "all"],
        default="none",
        help="Choose none for no interpolation, only_middle for only linear interpolation in the middle, all for linear interpolation + ends generation",
    )
    argparser.add_argument(
        "--node_features",
        default=None,
        help='The node features type; choices = ["dihedral", "chemical", "topological", "esm", "sidechain_orientation", "backbone_orientation", "secondary_structure", "c_beta"] and combinations (e.g. "chemical+sidechain_orientation")',
    )
    argparser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If None, NoamOpt is used, otherwise Adam with this starting learning rate",
    )
    argparser.add_argument(
        "--debug_file",
        default=None,
        type=str,
        help="If not None, open a specific file instead of loading the dataset",
    )
    argparser.add_argument(
        "--noise_unknown",
        default=None,
        type=float,
        help="The noise level to apply to masked structure (by default same as backbone_noise)",
    )
    argparser.add_argument(
        "--n_cycles",
        default=1,
        type=int,
        help="Number of refinement cycles (1 = only prediction, no refinement)",
    )
    argparser.add_argument(
        "--seq_init_mode",
        choices=[
            "zeros",
            "random",
            "esm_one_hot",
            "esm_probabilities",
            "uniform_probabilities",
        ],
        default="zeros",
        help="The sequence initialization mode for one-shot decoders",
    )
    argparser.add_argument(
        "--cycle_over_embedding",
        action="store_true",
        help="Run cycles at the embedding level instead of the sequence level",
    )
    argparser.add_argument(
        "--double_sequence_features",
        action="store_true",
        help="If true, sequence is used both in the encoder and in the decoder",
    )
    argparser.add_argument(
        "--decoder_type",
        choices=[
            "mpnn",
            "mpnn_auto",
            "egnn",
            "gvp",
            "gvp_orig",
            "mpnn_enc",
            "genprot",
            "egnn_orig",
            "egnn_refine",
            "egnn_geom",
            "egnn_lucid",
            "real_sake",
        ],
        default="mpnn_auto",
    )
    argparser.add_argument(
        "--encoder_type",
        choices=[
            "mpnn",
            "egnn",
            "gvp",
            "gvp_orig",
            "genprot",
            "egnn_orig",
            "egnn_refine",
            "egnn_geom",
            "egnn_lucid",
            "real_sake",
        ],
        default="mpnn",
    )
    argparser.add_argument(
        "--use_attention_in_encoder",
        action="store_true",
        help="If true, add global node attention to (MPNN) encoder layers",
    )
    argparser.add_argument(
        "--use_attention_in_decoder",
        action="store_true",
        help="If true, add global node attention to (MPNN) decoder layers",
    )
    argparser.add_argument(
        "--use_pna_in_encoder",
        action="store_true",
        help="If true, add PNA aggregation to encoder layers",
    )
    argparser.add_argument(
        "--use_pna_in_decoder",
        action="store_true",
        help="If true, add PNA aggregation to decoder layers",
    )
    argparser.add_argument(
        "--separate_modules_num",
        default=1,
        type=int,
        help="The number of separate modules to use for recycling (if n_cycles > separate_modules_num, the last module is used for all remaining cycles)",
    )
    argparser.add_argument(
        "--struct_loss_type",
        choices=["mse", "huber", "relaxed_mse"],
        help="The type of the structure loss",
        default="mse",
    )
    argparser.add_argument(
        "--skip_tqdm",
        action="store_true",
        help="Skip drawing the tqdm progressbars for epoch progress",
    )
    argparser.add_argument(
        "--only_cycle_over_decoder",
        action="store_true",
        help="If true (and cycle_over_embedding is true, and n_cycles > 1), only cycle over the decoder, not the encoder",
    )
    argparser.add_argument(
        "--not_shuffle_clusters",
        action="store_true",
        help="Use a fixed representative for each cluster instead of shuffling them",
    )
    argparser.add_argument(
        "--random_connections_frac",
        type=float,
        default=0.0,
        help="The fraction of random connections to add to the graph",
    )
    argparser.add_argument(
        "--test_invariance",
        action="store_true",
        help="If true, test the SE(3) sequence invariance and/or coordinate equivariance of the model",
    )
    argparser.add_argument(
        "--skip_weight_init",
        action="store_true",
        help="Skip weight initialization",
    )
    argparser.add_argument(
        "--log_wandb_name",
        help="Experiment name for wandb (if not given, experiment is not logged)",
    )
    argparser.add_argument(
        "--use_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )
    argparser.add_argument(
        "--predict_file",
        help="Predict the given file",
    )
    argparser.add_argument(
        "--predict_positions",
        help="Mask specific positions in the given file (only with predict_file); e.g. A, A:5-10,20,30-40 (fasta-based numbering, ends not included)",
    )
    argparser.add_argument(
        "--num_predictions",
        type=int,
        default=1,
        help="Number of predictions to make (only with predict_file; if > 1, the model is run in train mode)",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (only with predict_file or test and with AR models)",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    argparser.add_argument(
        "--co_design",
        choices=["none", "seq", "share_enc", "share_all"],
        help="Co-design options",
        default="none",
    )
    argparser.add_argument(
        "--struct_loss_weight",
        type=float,
        default=1.0,
        help="Structure loss weight",
    )
    argparser.add_argument(
        "--seq_loss_weight",
        type=float,
        default=1.0,
        help="Sequence loss weight",
    )
    argparser.add_argument(
        "--quaternion",
        action="store_true",
        help="Use quaternions instead of Euler angles",
    )
    argparser.add_argument(
        "--violation_loss_weight",
        type=float,
        default=0.0,
        help="Violations loss weight",
    )
    argparser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.0,
        help="Backbone noise",
    )
    argparser.add_argument(
        "--m_dim",
        type=int,
        default=64,
        help="m_dim in EGNN",
    )
    argparser.add_argument(
        "--m_factor",
        type=int,
        default=1,
        help="m_factor in EGNN",
    )
    argparser.add_argument(
        "--hidden_factor",
        type=int,
        default=1,
        help="hidden_factor in EGNN",
    )
    argparser.add_argument(
        "--detach_between_cycles",
        action="store_true",
        help="Detach the graph between cycles",
    )
    argparser.add_argument(
        "--not_pass_features",
        action="store_true",
        help="Do not pass features between encoder and decoder",
    )
    argparser.add_argument(
        "--keep_edge_model",
        action="store_true",
        help="Keep the edge model in the last layer",
    )
    argparser.add_argument(
        "--vector_angles",
        action="store_true",
        help="Use vector angles instead of Euler angles",
    )
    argparser.add_argument(
        "--linear_layers_num",
        type=int,
        default=0,
        help="The number of linear graph layers to use in the decoder (GVP)",
    )
    argparser.add_argument(
        "--pass_edge_vectors",
        action="store_true",
        help="Pass edge vectors to the decoder (GVP)",
    )

    # Temporary
    argparser.add_argument(
        "--predict_oxygens",
        action="store_true",
        help="Predict oxygen rotation angle (GVP only)",
    )
    argparser.add_argument(
        "--no_edge_update",
        action="store_true",
        help="Do not update edge vectors",
    )
    argparser.add_argument(
        "--diffusion",
        action="store_true",
        help="Use diffusion",
    )
    argparser.add_argument(
        "--flow_matching",
        action="store_true",
        help="Use flow matching",
    )
    argparser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=300,
        help="Number of diffusion steps",
    )
    argparser.add_argument(
        "--mask_all_cdrs",
        action="store_true",
        help="Mask all CDRs",
    )
    argparser.add_argument(
        "--variance_schedule",
        choices=["cosine", "linear", "quadratic"],
        default="linear",
        help="Variance schedule",
    )
    argparser.add_argument(
        "--freeze_structure",
        action="store_true",
        help="Freeze structure modules (when co_design is seq)",
    )
    argparser.add_argument(
        "--mask_sequence",
        action="store_true",
        help="Mask sequence",
    )
    argparser.add_argument(
        "--seq_diffusion_type",
        choices=["mask", "uniform"],
        default="mask",
        help="Sequence diffusion type",
    )
    argparser.add_argument(
        "--scale_timestep",
        action="store_true",
        help="Scale timestep feature",
    )
    argparser.add_argument(
        "--reset_masked",
        action="store_true",
        help="Reset masked positions in diffusion",
    )
    argparser.add_argument(
        "--diffusion_parameterization",
        choices=["noise", "x0", "mu_t", "v"],
        default="noise",
        help="Predict noise, x0 or mu_t with the translation diffusion model",
    )
    argparser.add_argument(
        "--variance_scale",
        type=float,
        default=1.0,
        help="Variance scale for sampling",
    )
    argparser.add_argument(
        "--not_recover_x0",
        action="store_true",
        help="Do not recover x0 in the translation diffusion model (when diffusion_parameterization is mu_t)",
    )
    argparser.add_argument(
        "--force_neighbor_edges",
        action="store_true",
        help="Always form edges between neighbors in the chain",
    )
    argparser.add_argument(
        "--test_cdr",
        choices=["L1", "L2", "L3", "H1", "H2", "H3"],
        default=None,
        help="Test on a single CDR",
    )
    argparser.add_argument(
        "--linear_interpolation",
        action="store_true",
        help="Use linear interpolation for sampling",
    )
    argparser.add_argument(
        "--skip_diffuse",
        action="store_true",
        help="Validate the same way as in the training",
    )
    argparser.add_argument(
        "--add_mask_feature",
        action="store_true",
        help="Add mask feature",
    )
    argparser.add_argument(
        "--add_anchor_feature",
        action="store_true",
        help="Add anchor feature",
    )
    argparser.add_argument(
        "--sequence_in_encoder",
        action="store_true",
        help="Pass sequence to the encoder",
    )
    argparser.add_argument(
        "--weighted_diff_loss",
        action="store_true",
        help="Weighted diffusion loss",
    )
    argparser.add_argument(
        "--pos_cosine_nu",
        type=float,
        default=1.0,
        help="The nu parameter for adaptive cosine variance schedule for CA coordinates",
    )
    argparser.add_argument(
        "--rot_cosine_nu",
        type=float,
        default=1.0,
        help="The nu parameter for adaptive cosine variance schedule for orientations",
    )
    argparser.add_argument(
        "--seq_cosine_nu",
        type=float,
        default=1.0,
        help="The nu parameter for adaptive cosine variance schedule for sequence",
    )
    argparser.add_argument(
        "--barycenter",
        action="store_true",
        help="Use barycenter as mean of noise",
    )
    argparser.add_argument(
        "--noise_around_interpolation",
        action="store_true",
        help="Add noise around linear interpolation",
    )
    argparser.add_argument(
        "--patch_around_mask",
        action="store_true",
        help="Patch around masked positions",
    )
    argparser.add_argument(
        "--only_separate_seq",
        action="store_true",
        help="Only use multiple separate modules for sequence",
    )
    argparser.add_argument(
        "--only_separate_dec",
        action="store_true",
        help="Only use multiple separate modules in the decoder",
    )
    argparser.add_argument(
        "--only_use_linear_for_str",
        action="store_true",
        help="Only use linear layers for structure",
    )
    argparser.add_argument(
        "--initial_patch_size",
        type=int,
        default=128,
        help="Initial patch size for patching",
    )
    argparser.add_argument(
        "--mask_sequential",
        action="store_true",
        help="Mask sequential positions",
    )
    argparser.add_argument(
        "--use_graph_context",
        action="store_true",
        help="Use timestep as global context",
    )
    argparser.add_argument(
        "--norm_coors",
        action="store_true",
        help="Normalize coordinates",
    )
    argparser.add_argument(
        "--old_noise",
        action="store_true",
        help="Add noise to coordinates instead of replacing them",
    )
    argparser.add_argument(
        "--noise_structure",
        action="store_true",
        help="Add noise to structure",
    )
    # argparser.add_argument(
    #     "--no_oxygen",
    #     action="store_true",
    #     help="Do not use oxygen information",
    # )
    argparser.add_argument(
        "--no_added_diff_noise",
        action="store_true",
        help="Do not add diffusion noise",
    )
    argparser.add_argument(
        "--cosine_cutoff",
        default=0.8,
        type=float,
        help="The cosine variance schedule cutoff for the diffusion model",
    )
    argparser.add_argument(
        "--all_x0",
        action="store_true",
        help="Use all x0 parameterization in the diffusion model",
    )

    args = argparser.parse_args()

    if args.test_invariance:
        args.no_mixed_precision = True
        args.debug = True
        args.max_protein_length = 150
    if args.device == "cpu":
        args.no_mixed_precision = True
    for prefix in ["egnn", "gvp"]:
        if args.decoder_type.startswith(prefix) or args.encoder_type.startswith(prefix):
            args.no_mixed_precision = True
    if args.test_excluded or args.validate:
        args.test = True
    if args.flow_matching:
        args.diffusion = True
        args.diffusion_parameterization = "noise"
    # args.predict_angles = args.predict_structure

    args.scale_timestep = True
    args.update_edges = not args.no_edge_update
    args.use_edge_vectors = True
    args.no_sequence_in_encoder = not args.sequence_in_encoder
    args.force = True
    args.use_node_dropout = True
    args.less_dropout = False
    args.no_oxygen = True
    return args


def main():
    args = parse()
    run(args)


if __name__ == "__main__":
    main()
