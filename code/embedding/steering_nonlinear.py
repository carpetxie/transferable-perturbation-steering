"""E43: H100 nonlinear-consumer test of representation steering.

Each target gets a control-only MLP decoder.  Frozen perturbation vectors are
added to held-out control-cell embeddings and decoded into gene expression.
The target perturbation outcomes are used only for final scoring.
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
app = modal.App("vec2vec-steering", image=image)
TARGETS = ("scgpt", "gf1", "gf2", "gf3", "scvi", "pca")


def _cosines(torch, prediction, truth):
    denominator = prediction.norm(dim=1) * truth.norm(dim=1)
    return (prediction * truth).sum(1) / denominator.clamp_min(1e-12)


def _specific(torch, matrix, generic):
    unit = generic / generic.norm().clamp_min(1e-12)
    return matrix - (matrix @ unit).unsqueeze(1) * unit.unsqueeze(0)


@app.function(gpu="H100", volumes={CACHE: VOL}, timeout=14400, memory=32768)
def run_target(target: str):
    import numpy as np
    import torch
    from torch import nn

    if target not in TARGETS:
        raise ValueError(target)
    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu.upper():
        raise RuntimeError(f"E43 requires H100, received {gpu}")
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
    z_std = control_z[train_idx].std(0)
    z_std = np.maximum(z_std, 1e-5)
    y_mean = control_y[train_idx].mean(0)
    y_std = control_y[train_idx].std(0)
    y_std = np.maximum(y_std, 0.05)
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
        "target": target,
        "gpu": gpu,
        "n_controls_train": len(train_idx),
        "n_controls_valid": len(valid_idx),
        "n_starting_states": len(z_start),
        "n_perturbations": len(pert_names),
        "n_genes": truth.shape[1],
        "arms": arm_names,
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
        best_loss = float("inf")
        best_state = None
        stale = 0
        epochs = 0
        for epoch in range(160):
            model.train()
            permutation = torch.randperm(
                len(z_train), generator=generator, device="cpu")
            for start in range(0, len(permutation), 256):
                batch = permutation[start:start + 256].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(z_train[batch])
                loss = torch.mean((prediction - y_train[batch]) ** 2)
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
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= 20:
                break
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            valid_prediction = model(z_valid)
            centered_true = y_valid - y_valid.mean(0, keepdim=True)
            centered_pred = valid_prediction - y_valid.mean(0, keepdim=True)
            validation_r2 = 1.0 - float(
                ((valid_prediction - y_valid) ** 2).sum().item()
                / centered_true.square().sum().clamp_min(1e-12).item())
            validation_cosine = float(_cosines(
                torch, centered_pred, centered_true).mean().item())
            base_start = model(z_start)

            # Decode all arms at per-cell starting states. Keeping the cell
            # dimension until after decoding is what makes this a nonlinear
            # steering assay rather than a delta-only linear projection.
            arm_predictions = torch.empty(
                (len(arm_names), len(pert_names), truth.shape[1]),
                dtype=torch.float32, device="cpu")
            nonadditivity = torch.empty(
                (len(arm_names), len(pert_names)), dtype=torch.float32,
                device="cpu")
            for arm in range(len(arm_names)):
                for pert in range(len(pert_names)):
                    change = (model(z_start + delta_t[arm, pert])
                              - base_start) * y_scale
                    mean_change = change.mean(0)
                    arm_predictions[arm, pert] = mean_change.cpu()
                    between_state = (change - mean_change).square().mean().sqrt()
                    signal = mean_change.square().mean().sqrt().clamp_min(1e-12)
                    nonadditivity[arm, pert] = (between_state / signal).cpu()

            # A consumer ceiling: decode actual held-out perturbed cells and
            # subtract decoded controls. This tests whether the consumer can
            # read real perturbation state at all.
            decoded_control_mean = (model(z_valid) * y_scale).mean(0)
            actual_predictions = []
            for pert in range(len(pert_names)):
                start, stop = perturbed_bounds[pert:pert + 2]
                values = torch.from_numpy(
                    (perturbed_z[start:stop] - z_mean) / z_std).to(device)
                actual_predictions.append(
                    (model(values) * y_scale).mean(0) - decoded_control_mean)
            actual_predictions = torch.stack(actual_predictions)

        pred = arm_predictions.to(device)
        raw_scores = _cosines(
            torch, pred.reshape(-1, pred.shape[-1]),
            truth_t.repeat(len(arm_names), 1)).reshape(
                len(arm_names), len(pert_names))
        pred_specific = _specific(
            torch, pred.reshape(-1, pred.shape[-1]), generic_t)
        truth_specific = _specific(torch, truth_t, generic_t)
        specific_scores = _cosines(
            torch, pred_specific,
            truth_specific.repeat(len(arm_names), 1)).reshape(
                len(arm_names), len(pert_names))
        actual_scores = _cosines(torch, actual_predictions, truth_t)
        actual_specific = _cosines(
            torch, _specific(torch, actual_predictions, generic_t),
            truth_specific)

        arm_results = {}
        for arm, name in enumerate(arm_names):
            arm_results[name] = {
                "mean_cosine": float(raw_scores[arm].mean().item()),
                "median_cosine": float(raw_scores[arm].median().item()),
                "generic_removed_cosine": float(
                    specific_scores[arm].mean().item()),
                "mean_state_nonadditivity_ratio": float(
                    nonadditivity[arm].mean().item()),
                "per_perturbation_cosine": raw_scores[arm].cpu().tolist(),
                "per_perturbation_generic_removed_cosine": (
                    specific_scores[arm].cpu().tolist()),
            }
        output["variants"][variant] = {
            "epochs": epochs,
            "best_validation_mse_standardized": best_loss,
            "validation_expression_r2": validation_r2,
            "validation_centered_cell_cosine": validation_cosine,
            "actual_perturbed_consumer_ceiling": {
                "mean_cosine": float(actual_scores.mean().item()),
                "generic_removed_cosine": float(actual_specific.mean().item()),
            },
            "arm_results": arm_results,
        }
        print(
            f"{target}/{variant}: epochs={epochs} R2={validation_r2:.3f} "
            f"single={arm_results['best_single_inner']['mean_cosine']:.3f} "
            f"balanced={arm_results['architecture_balanced']['mean_cosine']:.3f} "
            f"native={arm_results['target_native_positive']['mean_cosine']:.3f}",
            flush=True,
        )

    outdir = f"{CACHE}/steering_results"
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
