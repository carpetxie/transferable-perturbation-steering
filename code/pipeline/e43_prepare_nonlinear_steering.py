"""Prepare leak-free E43 inputs for nonlinear representation steering.

The Modal GPU job only receives frozen control-trained transfer vectors and
target-native embeddings.  No target perturbation outcome is used to choose a
view, donor, dimension, or weighting rule.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import qlib
import vdlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
E42 = ROOT / "artifacts/queue/e42_nested_zero_label_selection.json"
OUT = ROOT / "artifacts/steering_inputs"
ARM_NAMES = (
    "fixed_six_equal_all",
    "best_single_inner",
    "architecture_balanced",
    "selected_joint",
    "target_native_positive",
    "best_single_mismatch",
    "best_single_generic",
    "zero_shift",
)


def ordered(models):
    selected = set(models)
    return tuple(model for model in vdlib.PRIMARY if model in selected)


def predict_from_rule(native, prepared_by_k, spaces, target, rule):
    k = int(rule["k"])
    views = ordered(set(rule["view_pool_rule"]) | {target})
    key = (k, views)
    if key not in spaces:
        spaces[key] = vdlib.fit_space(prepared_by_k[k], views, k=k)
    return vdlib.predict_native_matrix(
        native,
        prepared_by_k[k],
        spaces[key],
        target,
        tuple(rule["donor_pool_rule"]),
        rule["weighting"],
    )


def perturbation_cells(native, target):
    chunks = []
    bounds = [0]
    for pert in native["perts"]:
        Z, lib, ngene = vdlib.load_embedding(pert, target)
        residual = (
            qlib.apply_depth_projector(
                Z, lib, ngene, native["betas"][target])
            - native["means"][target]
        )
        half = len(residual) // 2
        chunks.append(residual[half:].astype(np.float32))
        bounds.append(bounds[-1] + len(residual) - half)
    return np.concatenate(chunks), np.asarray(bounds, dtype=np.int64)


def main():
    with E42.open() as handle:
        e42 = json.load(handle)

    native = vdlib.load_native_deltas(vdlib.PRIMARY)
    gene = vdlib.load_gene_endpoint(native)
    ks = sorted({
        int(result["selection_rule"]["k"])
        for row in e42["outer_rows"]
        for result in (
            row["selected"],
            row["comparators"]["fixed_six_equal_all"],
            row["comparators"]["best_single_inner"],
            row["comparators"]["architecture_balanced"],
        )
    })
    prepared_by_k = {k: vdlib.prepare_controls(native, k=k) for k in ks}
    spaces = {}
    OUT.mkdir(parents=True, exist_ok=True)

    rows = {row["outer_target"]: row for row in e42["outer_rows"]}
    truth = np.stack([gene["true_de"][p] for p in native["perts"]])
    for target in vdlib.PRIMARY:
        print(f"preparing {target}", flush=True)
        row = rows[target]
        rules = {
            "fixed_six_equal_all": row["comparators"]
                ["fixed_six_equal_all"]["selection_rule"],
            "best_single_inner": row["comparators"]
                ["best_single_inner"]["selection_rule"],
            "architecture_balanced": row["comparators"]
                ["architecture_balanced"]["selection_rule"],
            "selected_joint": row["selected"]["selection_rule"],
        }
        predictions = {
            name: predict_from_rule(
                native, prepared_by_k, spaces, target, rule)
            for name, rule in rules.items()
        }
        best = predictions["best_single_inner"]
        native_positive = np.stack([
            native["first"][target][pert] for pert in native["perts"]])
        deltas = np.stack([
            predictions["fixed_six_equal_all"],
            best,
            predictions["architecture_balanced"],
            predictions["selected_joint"],
            native_positive,
            np.roll(best, -1, axis=0),
            np.repeat(best.mean(0, keepdims=True), len(best), axis=0),
            np.zeros_like(best),
        ]).astype(np.float32)
        perturbed, bounds = perturbation_cells(native, target)
        if not all(np.isfinite(array).all() for array in (
                deltas, truth, perturbed, native["controls"][target],
                gene["control_expression"])):
            raise AssertionError(f"{target}: non-finite steering input")
        np.savez_compressed(
            OUT / f"{target}.npz",
            control_Z=native["controls"][target].astype(np.float32),
            control_Y=gene["control_expression"],
            true_de=truth.astype(np.float32),
            generic_de=gene["generic"].astype(np.float32),
            deltas=deltas,
            arm_names=np.asarray(ARM_NAMES),
            pert_names=np.asarray(native["perts"]),
            perturbed_Z=perturbed,
            perturbed_bounds=bounds,
            hvg=gene["hvg"].astype(np.int64),
        )
        print(f"  {deltas.shape=} {perturbed.shape=}", flush=True)
    print("E43_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
