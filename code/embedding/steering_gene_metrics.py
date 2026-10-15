"""Post-hoc E43 gene-level recovery audit on the frozen steering inputs.

This reruns the two pre-existing control-only nonlinear consumers with identical
seeds and training, then retains per-perturbation gene-level metrics that E43
did not save. No encoder, GCCA, donor rule, or steering vector is refit here.
"""
from __future__ import annotations

import json
import os

import modal


CACHE = "/cache"
VOL = modal.Volume.from_name("vec2vec-cache", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.5.1")
)
app = modal.App("vec2vec-steering-gene-metrics", image=image)
TARGETS = ("scgpt", "gf1", "gf2", "gf3", "scvi", "pca")
TOP_K = (25, 50, 100)


def _cosines(torch, prediction, truth):
    denominator = prediction.norm(dim=1) * truth.norm(dim=1)
    return (prediction * truth).sum(1) / denominator.clamp_min(1e-12)


def _specific(torch, matrix, generic):
    unit = generic / generic.norm().clamp_min(1e-12)
    return matrix - (matrix @ unit).unsqueeze(1) * unit.unsqueeze(0)


def _rank_rows(torch, matrix):
    order = torch.argsort(matrix, dim=1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    values = torch.arange(matrix.shape[1], device=matrix.device,
                          dtype=torch.float32).expand_as(ranks)
    ranks.scatter_(1, order, values)
    return ranks


def _gene_metrics(torch, prediction, truth):
    """Return per-perturbation metrics; truth and prediction are P x genes."""
    error = prediction - truth
    rms_truth = truth.square().mean(1).sqrt().clamp_min(1e-12)
    normalized_rmse = error.square().mean(1).sqrt() / rms_truth
    prediction_centered = prediction - prediction.mean(1, keepdim=True)
    truth_centered = truth - truth.mean(1, keepdim=True)
    pearson = _cosines(torch, prediction_centered, truth_centered)
    pred_rank = _rank_rows(torch, prediction)
    truth_rank = _rank_rows(torch, truth)
    spearman = _cosines(
        torch,
        pred_rank - pred_rank.mean(1, keepdim=True),
        truth_rank - truth_rank.mean(1, keepdim=True),
    )
    out = {
        "cosine": _cosines(torch, prediction, truth).cpu().tolist(),
        "pearson": pearson.cpu().tolist(),
        "spearman": spearman.cpu().tolist(),
        "normalized_rmse": normalized_rmse.cpu().tolist(),
    }
    for k in TOP_K:
        truth_top = torch.topk(truth.abs(), k, dim=1).indices
        pred_top = torch.topk(prediction.abs(), k, dim=1).indices
        membership = (pred_top.unsqueeze(2) == truth_top.unsqueeze(1))
        overlap = membership.any(2).float().mean(1)
        pred_sign = torch.gather(torch.sign(prediction), 1, pred_top)
        matched_truth_pos = membership.float().argmax(2)
        matched_truth_idx = torch.gather(truth_top, 1, matched_truth_pos)
        matched_truth_sign = torch.gather(torch.sign(truth), 1, matched_truth_idx)
        signed_overlap = (
            membership.any(2) & (pred_sign == matched_truth_sign)
        ).float().mean(1)
        truth_sign = torch.gather(torch.sign(truth), 1, truth_top)
        prediction_on_truth_top = torch.gather(
            torch.sign(prediction), 1, truth_top)
        true_top_sign_accuracy = (
            prediction_on_truth_top == truth_sign).float().mean(1)
        out[f"top{k}_overlap"] = overlap.cpu().tolist()
        out[f"top{k}_signed_overlap"] = signed_overlap.cpu().tolist()
        out[f"true_top{k}_sign_accuracy"] = (
            true_top_sign_accuracy.cpu().tolist())
    return out


@app.function(gpu="H100", volumes={CACHE: VOL}, timeout=14400, memory=32768)
def run_target(target: str):
    import numpy as np
    import torch
    from torch import nn

    if target not in TARGETS:
        raise ValueError(target)
    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu.upper():
        raise RuntimeError(f"H100 required, received {gpu}")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    source = f"{CACHE}/steering_inputs/steering_inputs/{target}.npz"
    with np.load(source, allow_pickle=False) as data:
        control_z = data["control_Z"].astype(np.float32)
        control_y = data["control_Y"].astype(np.float32)
        truth = data["true_de"].astype(np.float32)
        generic = data["generic_de"].astype(np.float32)
        deltas = data["deltas"].astype(np.float32)
        arm_names = [str(value) for value in data["arm_names"]]
        pert_names = [str(value) for value in data["pert_names"]]
        perturbed_z = data["perturbed_Z"].astype(np.float32)
        perturbed_bounds = data["perturbed_bounds"].astype(np.int64)

    rng = np.random.default_rng(43001)
    order = rng.permutation(len(control_z))
    train_idx, valid_idx = order[:4800], order[4800:]
    z_mean = control_z[train_idx].mean(0)
    z_std = np.maximum(control_z[train_idx].std(0), 1e-5)
    y_mean = control_y[train_idx].mean(0)
    y_std = np.maximum(control_y[train_idx].std(0), 0.05)
    z_train = torch.from_numpy(
        (control_z[train_idx] - z_mean) / z_std).to(device)
    y_train = torch.from_numpy(
        (control_y[train_idx] - y_mean) / y_std).to(device)
    z_valid = torch.from_numpy(
        (control_z[valid_idx] - z_mean) / z_std).to(device)
    y_valid = torch.from_numpy(
        (control_y[valid_idx] - y_mean) / y_std).to(device)
    y_scale = torch.from_numpy(y_std).to(device)
    truth_t = torch.from_numpy(truth).to(device)
    generic_t = torch.from_numpy(generic).to(device)
    delta_t = torch.from_numpy(deltas / z_std[None, None, :]).to(device)
    z_start = z_valid[:256]

    architectures = {
        "mlp256_silu": (256, 256, "silu"),
        "mlp512_gelu": (512, 256, "gelu"),
    }
    output = {
        "experiment": "E43 post-hoc gene-level audit",
        "target": target,
        "gpu": gpu,
        "n_perturbations": len(pert_names),
        "n_genes": truth.shape[1],
        "top_k": list(TOP_K),
        "perturbations": pert_names,
        "variants": {},
    }

    for variant_index, (variant, (h1, h2, activation)) in enumerate(
            architectures.items()):
        torch.manual_seed(43010 + variant_index)
        act = nn.SiLU if activation == "silu" else nn.GELU
        model = nn.Sequential(
            nn.Linear(z_train.shape[1], h1), act(),
            nn.Linear(h1, h2), act(),
            nn.Linear(h2, y_train.shape[1]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(43100 + variant_index)
        best_loss, best_state, stale, epochs = float("inf"), None, 0, 0
        for epoch in range(160):
            model.train()
            permutation = torch.randperm(
                len(z_train), generator=generator, device="cpu")
            for start in range(0, len(permutation), 256):
                batch = permutation[start:start + 256].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(z_train[batch]) - y_train[batch]) ** 2)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(torch.mean(
                    (model(z_valid) - y_valid) ** 2).item())
            epochs = epoch + 1
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 20:
                break
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            base_start = model(z_start)
            arm_predictions = torch.empty(
                (len(arm_names), len(pert_names), truth.shape[1]),
                dtype=torch.float32, device="cpu")
            for arm in range(len(arm_names)):
                for pert in range(len(pert_names)):
                    change = (model(z_start + delta_t[arm, pert])
                              - base_start) * y_scale
                    arm_predictions[arm, pert] = change.mean(0).cpu()
            decoded_control_mean = (model(z_valid) * y_scale).mean(0)
            actual_predictions = []
            for pert in range(len(pert_names)):
                start, stop = perturbed_bounds[pert:pert + 2]
                values = torch.from_numpy(
                    (perturbed_z[start:stop] - z_mean) / z_std).to(device)
                actual_predictions.append(
                    (model(values) * y_scale).mean(0) - decoded_control_mean)
            actual_predictions = torch.stack(actual_predictions)

        prediction_sets = {
            name: arm_predictions[index].to(device)
            for index, name in enumerate(arm_names)}
        prediction_sets["actual_perturbed_consumer"] = actual_predictions
        raw, generic_removed = {}, {}
        truth_specific = _specific(torch, truth_t, generic_t)
        for name, prediction in prediction_sets.items():
            raw[name] = _gene_metrics(torch, prediction, truth_t)
            generic_removed[name] = _gene_metrics(
                torch, _specific(torch, prediction, generic_t), truth_specific)
        output["variants"][variant] = {
            "epochs": epochs,
            "best_validation_mse_standardized": best_loss,
            "raw": raw,
            "generic_removed": generic_removed,
        }
        print(f"{target}/{variant}: epochs={epochs} gene metrics complete",
              flush=True)

    outdir = f"{CACHE}/steering_gene_metrics"
    os.makedirs(outdir, exist_ok=True)
    destination = f"{outdir}/{target}.json"
    with open(destination, "w") as handle:
        json.dump(output, handle, indent=2)
    VOL.commit()
    return destination


@app.local_entrypoint()
def main():
    for result in run_target.map(TARGETS):
        print(result, flush=True)
