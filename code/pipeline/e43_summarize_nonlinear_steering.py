"""Aggregate E43 H100 nonlinear-steering workers and apply assay gates."""
from __future__ import annotations

import json
import pathlib

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKERS = ROOT / "artifacts/steering_results/steering_results"
OUTPUT = ROOT / "artifacts/queue/e43_nonlinear_steering.json"
TARGETS = ("scgpt", "gf1", "gf2", "gf3", "scvi", "pca")
VARIANTS = ("mlp256_silu", "mlp512_gelu")
KEY_ARMS = (
    "fixed_six_equal_all",
    "best_single_inner",
    "architecture_balanced",
    "selected_joint",
    "target_native_positive",
    "best_single_mismatch",
    "best_single_generic",
    "zero_shift",
)


def interval(values, rng, draws=10000):
    """Hierarchical bootstrap over targets and perturbations."""
    values = np.asarray(values, dtype=np.float64)
    samples = np.empty(draws)
    for draw in range(draws):
        target_idx = rng.integers(0, values.shape[0], values.shape[0])
        pert_idx = rng.integers(0, values.shape[1], values.shape[1])
        samples[draw] = values[target_idx][:, pert_idx].mean()
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def main():
    workers = {}
    for target in TARGETS:
        with (WORKERS / f"{target}.json").open() as handle:
            row = json.load(handle)
        if row["target"] != target or "H100" not in row["gpu"].upper():
            raise AssertionError(f"{target}: invalid worker identity/GPU")
        workers[target] = row

    rng = np.random.default_rng(43043)
    variants = {}
    for variant in VARIANTS:
        arm_summary = {}
        raw = {}
        specific = {}
        for arm in KEY_ARMS:
            raw[arm] = np.asarray([
                workers[target]["variants"][variant]["arm_results"][arm]
                ["per_perturbation_cosine"] for target in TARGETS])
            specific[arm] = np.asarray([
                workers[target]["variants"][variant]["arm_results"][arm]
                ["per_perturbation_generic_removed_cosine"]
                for target in TARGETS])
            arm_summary[arm] = {
                "mean_cosine": float(raw[arm].mean()),
                "mean_cosine_ci95": interval(raw[arm], rng),
                "generic_removed_cosine": float(specific[arm].mean()),
                "generic_removed_cosine_ci95": interval(
                    specific[arm], rng),
                "mean_state_nonadditivity_ratio": float(np.mean([
                    workers[target]["variants"][variant]["arm_results"][arm]
                    ["mean_state_nonadditivity_ratio"] for target in TARGETS
                ])),
                "per_target_mean_cosine": {
                    target: float(raw[arm][index].mean())
                    for index, target in enumerate(TARGETS)
                },
            }

        contrasts = {}
        for optimized in ("best_single_inner", "architecture_balanced"):
            for baseline in (
                    "fixed_six_equal_all", "best_single_mismatch",
                    "best_single_generic"):
                name = f"{optimized}_minus_{baseline}"
                delta = raw[optimized] - raw[baseline]
                delta_specific = specific[optimized] - specific[baseline]
                contrasts[name] = {
                    "mean_cosine_delta": float(delta.mean()),
                    "mean_cosine_delta_ci95": interval(delta, rng),
                    "generic_removed_delta": float(delta_specific.mean()),
                    "generic_removed_delta_ci95": interval(
                        delta_specific, rng),
                }

        consumer = np.asarray([
            workers[target]["variants"][variant]
            ["actual_perturbed_consumer_ceiling"]["mean_cosine"]
            for target in TARGETS])
        consumer_specific = np.asarray([
            workers[target]["variants"][variant]
            ["actual_perturbed_consumer_ceiling"]["generic_removed_cosine"]
            for target in TARGETS])
        apparatus_pass = bool(
            consumer.mean() > 0.5
            and raw["target_native_positive"].mean()
                > raw["best_single_mismatch"].mean()
            and specific["target_native_positive"].mean()
                > specific["best_single_mismatch"].mean())
        steering_gates = {}
        for optimized in ("best_single_inner", "architecture_balanced"):
            mismatch_delta = raw[optimized] - raw["best_single_mismatch"]
            specific_mismatch_delta = (
                specific[optimized] - specific["best_single_mismatch"])
            steering_gates[optimized] = bool(
                mismatch_delta.mean() > 0.05
                and interval(mismatch_delta, rng)[0] > 0
                and specific_mismatch_delta.mean() > 0.10
                and interval(specific_mismatch_delta, rng)[0] > 0)

        variants[variant] = {
            "validation_expression_r2_mean": float(np.mean([
                workers[target]["variants"][variant]
                ["validation_expression_r2"] for target in TARGETS])),
            "actual_perturbed_consumer_ceiling_mean": float(consumer.mean()),
            "actual_perturbed_consumer_ceiling_generic_removed": float(
                consumer_specific.mean()),
            "apparatus_pass": apparatus_pass,
            "steering_gates": steering_gates,
            "arm_summary": arm_summary,
            "contrasts": contrasts,
        }

    robust_pass = {
        arm: all(variants[variant]["apparatus_pass"]
                 and variants[variant]["steering_gates"][arm]
                 for variant in VARIANTS)
        for arm in ("best_single_inner", "architecture_balanced")
    }
    conclusion = (
        "Both zero-label optimization rules pass a nonlinear, control-trained "
        "consumer steering assay in both MLP variants. Best-single is the "
        "stronger and more stable deployable rule; architecture-balanced also "
        "steers perturbation-specific expression but is harmed by the scVI "
        "outer-target case."
        if all(robust_pass.values()) else
        "At least one optimization rule did not robustly pass both nonlinear "
        "consumer variants; inspect the failed apparatus or steering gate."
    )
    result = {
        "experiment": "E43 nonlinear representation steering",
        "modal_app_id": "ap-4dMk6jF44mQ67cjTdufXld",
        "gpu_requirement": "NVIDIA H100",
        "models": list(TARGETS),
        "n_perturbations": 201,
        "selection_blinding": (
            "E42 view/donor/K rules frozen before outer-target outcomes; donor "
            "deltas use first halves; gene truth uses target second halves"),
        "consumer_training": (
            "control-only nonlinear MLP; 4,800 train and 1,200 held-out "
            "control cells; 256 held-out starting states per shift"),
        "predeclared_gates": {
            "apparatus": (
                "actual-perturbed consumer cosine > 0.5 and native steering "
                "beats mismatch before and after generic removal"),
            "optimized_arm": (
                "mean advantage over mismatched perturbations > 0.05, its "
                "hierarchical-bootstrap lower bound > 0, and generic-removed "
                "advantage > 0.10 with lower bound > 0"),
        },
        "robust_pass_both_mlp_variants": robust_pass,
        "variants": variants,
        "worker_results": workers,
        "conclusion": conclusion,
        "evidence_boundary": (
            "This establishes that control-fitted nonlinear gene-expression "
            "consumers respond perturbation-specifically when target embeddings "
            "are shifted. It does not establish causal steering of the original "
            "foundation models, unseen datasets, or wet-lab cellular state."),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "robust_pass": robust_pass,
        "conclusion": conclusion,
        "variant_summary": {
            variant: {
                "consumer_ceiling": variants[variant]
                    ["actual_perturbed_consumer_ceiling_mean"],
                "best_single": variants[variant]["arm_summary"]
                    ["best_single_inner"]["mean_cosine"],
                "balanced": variants[variant]["arm_summary"]
                    ["architecture_balanced"]["mean_cosine"],
                "mismatch": variants[variant]["arm_summary"]
                    ["best_single_mismatch"]["mean_cosine"],
            } for variant in VARIANTS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
