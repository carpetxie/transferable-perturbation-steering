"""E40: held-out control qualification for the six-view Modal shared space.

This is a representation-alignment qualification, not perturbation-transfer
evidence. Depth correction and GCCA are fitted on one matched control half;
all metrics are evaluated on the other half. The cancer-tuned Geneformer is
excluded from the primary panel and reserved for a later sensitivity analysis.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qlib


MODAL_EMB = pathlib.Path(os.environ.get(
    "MODAL_EMB", pathlib.Path.home() / ".cache/silico_vec2vec/modal_emb/norman"))
MODELS = ["scgpt", "gf1", "gf2", "gf3", "scvi", "pca"]
MODAL_NAME = {
    "scgpt": "scgpt",
    "gf1": "geneformer",
    "gf2": "geneformer_v2",
    "gf3": "geneformer_v3",
    "scvi": "scvi",
    "pca": "pca",
}
FAMILY = {
    "scgpt": "scgpt",
    "gf1": "geneformer",
    "gf2": "geneformer",
    "gf3": "geneformer",
    "scvi": "scvi",
    "pca": "pca",
}
K = 24
SEED = 40


def load_control(model):
    path = MODAL_EMB / f"control__{MODAL_NAME[model]}.npz"
    with np.load(path, allow_pickle=False) as data:
        return (data["Z"].astype(np.float64), data["lib"].astype(np.float64),
                data["ngene"].astype(np.float64))


def project(Z, gcca, model):
    A = (Z - gcca["mu"][model]) @ gcca["P"][model]
    return A @ gcca["W"][model] @ gcca["C"][model]


def standardize_pair(train, valid):
    mu = train.mean(0)
    scale = train.std(0)
    scale[scale < 1e-10] = 1.0
    return (train - mu) / scale, (valid - mu) / scale


def row_cosines(A, B):
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(np.sum(A * B, axis=1), den,
                     out=np.zeros(len(A)), where=den > 0)


def retrieval_from_similarity(sim):
    """Retrieve the paired column for each row of a cosine-similarity matrix."""
    diagonal = np.diag(sim)
    ranks = 1 + np.sum(sim > diagonal[:, None], axis=1)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def pair_metrics(projected):
    pairs = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            A, B = projected[a], projected[b]
            An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
            Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
            similarity = An @ Bn.T
            pairs[f"{a}|{b}"] = {
                "family_relation": "same" if FAMILY[a] == FAMILY[b] else "cross",
                "matched_cell_cosine": float(np.mean(row_cosines(A, B))),
                "flat_correlation": float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
                f"{a}_to_{b}": retrieval_from_similarity(similarity),
                f"{b}_to_{a}": retrieval_from_similarity(similarity.T),
            }
    return pairs


def per_model_metrics(projected, pairs):
    rows = {}
    for target in MODELS:
        alignment, flat, top1, top5, medrank = [], [], [], [], []
        by_relation = {"same": [], "cross": []}
        for other in MODELS:
            if other == target:
                continue
            a, b = sorted((target, other), key=MODELS.index)
            item = pairs[f"{a}|{b}"]
            direction = item[f"{target}_to_{other}"]
            relation = item["family_relation"]
            alignment.append(item["matched_cell_cosine"])
            flat.append(item["flat_correlation"])
            top1.append(direction["top1"])
            top5.append(direction["top5"])
            medrank.append(direction["median_rank"])
            by_relation[relation].append({
                "other": other,
                "matched_cell_cosine": item["matched_cell_cosine"],
                "flat_correlation": item["flat_correlation"],
                "top1": direction["top1"],
                "top5": direction["top5"],
                "median_rank": direction["median_rank"],
            })
        rows[target] = {
            "matched_cell_alignment": float(np.mean(alignment)),
            "flat_correlation": float(np.mean(flat)),
            "retrieval_top1": float(np.mean(top1)),
            "retrieval_top5": float(np.mean(top5)),
            "retrieval_median_rank": float(np.median(medrank)),
            "same_family_pairs": by_relation["same"],
            "cross_family_pairs": by_relation["cross"],
        }
    return rows


def spectrum(projected, gcca, n_fit):
    heldout = []
    for dim in range(K):
        columns = np.column_stack([projected[m][:, dim] for m in MODELS])
        corr = np.corrcoef(columns, rowvar=False)
        heldout.append(float(corr[np.triu_indices(len(MODELS), 1)].mean()))
    return {
        "fit_maxvar_strength": (
            (np.asarray(gcca["sv"]) ** 2 / (n_fit * len(MODELS))).tolist()),
        "heldout_mean_pairwise_correlation": heldout,
    }


def rank_metrics(Z):
    covariance = np.cov(Z, rowvar=False)
    values = np.linalg.eigvalsh(covariance)
    values = np.maximum(values, 0)
    total = float(values.sum())
    p = values / total if total > 0 else np.zeros_like(values)
    positive = p[p > 0]
    effective = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0
    participation = float(total ** 2 / np.sum(values ** 2)) if np.sum(values ** 2) > 0 else 0.0
    numerical = int(np.sum(values > max(float(values.max()), 1e-12) * 1e-8))
    return {
        "effective_rank": effective,
        "participation_ratio": participation,
        "numerical_rank": numerical,
        "max_variance_fraction": float(values.max() / total) if total > 0 else 1.0,
        "collapsed": bool(numerical < K or effective < K / 3),
    }


def leave_one_view(projected):
    out = {}
    for target in MODELS:
        others = [projected[m] for m in MODELS if m != target]
        prediction = np.mean(others, axis=0)
        truth = projected[target]
        residual = float(np.sum((truth - prediction) ** 2))
        centered = float(np.sum((truth - truth.mean(0)) ** 2))
        out[target] = {
            "mean_cell_cosine": float(np.mean(row_cosines(prediction, truth))),
            "flat_correlation": float(np.corrcoef(prediction.ravel(), truth.ravel())[0, 1]),
            "variance_weighted_r2": 1.0 - residual / max(centered, 1e-12),
        }
    return out


def evaluate(train_projected, valid_projected, gcca, n_fit):
    standardized_train, standardized_valid = {}, {}
    for model in MODELS:
        standardized_train[model], standardized_valid[model] = standardize_pair(
            train_projected[model], valid_projected[model])
    pairs = pair_metrics(standardized_valid)
    return {
        "per_model": per_model_metrics(standardized_valid, pairs),
        "pairwise": pairs,
        "canonical_spectrum": spectrum(standardized_valid, gcca, n_fit),
        "rank_and_collapse": {
            model: rank_metrics(standardized_valid[model]) for model in MODELS},
        "leave_one_view_reconstruction": leave_one_view(standardized_valid),
    }


def random_basis_views(train, valid, seed):
    rng = np.random.default_rng(seed)
    train_out, valid_out = {}, {}
    for model in MODELS:
        q, _ = np.linalg.qr(rng.normal(size=(train[model].shape[1], K)))
        train_out[model] = train[model] @ q
        valid_out[model] = valid[model] @ q
    return train_out, valid_out


def main():
    raw = {model: load_control(model) for model in MODELS}
    n = len(raw[MODELS[0]][0])
    if any(len(raw[m][0]) != n for m in MODELS):
        raise AssertionError("control row counts differ")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    fit_idx = np.sort(order[:n // 2])
    val_idx = np.sort(order[n // 2:])

    train, valid = {}, {}
    for model, (Z, lib, ngene) in raw.items():
        beta, _ = qlib.fit_depth_projector(Z[fit_idx], lib[fit_idx], ngene[fit_idx])
        fit_residual = qlib.apply_depth_projector(
            Z[fit_idx], lib[fit_idx], ngene[fit_idx], beta)
        val_residual = qlib.apply_depth_projector(
            Z[val_idx], lib[val_idx], ngene[val_idx], beta)
        mu = fit_residual.mean(0)
        train[model] = fit_residual - mu
        valid[model] = val_residual - mu

    # Primary six-view GCCA: fit correspondences and all preprocessing use fit controls only.
    gcca = qlib.fit_gcca(train, k=K)
    train_projected = {m: project(train[m], gcca, m) for m in MODELS}
    valid_projected = {m: project(valid[m], gcca, m) for m in MODELS}
    primary = evaluate(train_projected, valid_projected, gcca, len(fit_idx))

    # Negative control 1: destroy fit-control cell correspondence independently per view.
    shuffle_rng = np.random.default_rng(SEED + 1)
    shuffled_train = {
        m: train[m][shuffle_rng.permutation(len(fit_idx))] for m in MODELS}
    shuffled_gcca = qlib.fit_gcca(shuffled_train, k=K)
    shuffled_projected_train = {
        m: project(train[m], shuffled_gcca, m) for m in MODELS}
    shuffled_projected_valid = {
        m: project(valid[m], shuffled_gcca, m) for m in MODELS}
    shuffled = evaluate(shuffled_projected_train, shuffled_projected_valid,
                        shuffled_gcca, len(fit_idx))

    # Negative control 2: same dimensionality, independent bases, no learned alignment.
    random_train, random_valid = random_basis_views(train, valid, SEED + 2)
    random_standardized_train, random_standardized_valid = {}, {}
    for model in MODELS:
        random_standardized_train[model], random_standardized_valid[model] = standardize_pair(
            random_train[model], random_valid[model])
    random_pairs = pair_metrics(random_standardized_valid)
    random_control = {
        "per_model": per_model_metrics(random_standardized_valid, random_pairs),
        "pairwise": random_pairs,
        "rank_and_collapse": {
            m: rank_metrics(random_standardized_valid[m]) for m in MODELS},
        "leave_one_view_reconstruction": leave_one_view(random_standardized_valid),
    }

    qualification = {}
    for model in MODELS:
        observed = primary["per_model"][model]
        shuffled_row = shuffled["per_model"][model]
        random_row = random_control["per_model"][model]
        observed_loo = primary["leave_one_view_reconstruction"][model]
        shuffled_loo = shuffled["leave_one_view_reconstruction"][model]
        random_loo = random_control["leave_one_view_reconstruction"][model]
        rank = primary["rank_and_collapse"][model]
        checks = {
            "alignment_margin": observed["matched_cell_alignment"] > max(
                shuffled_row["matched_cell_alignment"],
                random_row["matched_cell_alignment"]) + 0.05,
            "retrieval_top5_margin": observed["retrieval_top5"] > max(
                shuffled_row["retrieval_top5"], random_row["retrieval_top5"]) + 0.01,
            "loo_correlation_margin": observed_loo["flat_correlation"] > max(
                shuffled_loo["flat_correlation"], random_loo["flat_correlation"]) + 0.05,
            "full_numerical_rank": rank["numerical_rank"] == K,
            "effective_rank_at_least_8": rank["effective_rank"] >= K / 3,
        }
        qualification[model] = {"passed": bool(all(checks.values())), "checks": checks}

    passed = bool(all(item["passed"] for item in qualification.values()))
    print(f"E40 control qualification: {'PASS' if passed else 'FAIL'}")
    print(f"{'model':<7} {'align':>8} {'shuf':>8} {'random':>8} "
          f"{'top5':>8} {'loo r':>8} {'effrank':>8} {'pass':>6}")
    for model in MODELS:
        p = primary["per_model"][model]
        s = shuffled["per_model"][model]
        r = random_control["per_model"][model]
        loo = primary["leave_one_view_reconstruction"][model]
        er = primary["rank_and_collapse"][model]["effective_rank"]
        print(f"{model:<7} {p['matched_cell_alignment']:8.3f} "
              f"{s['matched_cell_alignment']:8.3f} {r['matched_cell_alignment']:8.3f} "
              f"{p['retrieval_top5']:8.3f} {loo['flat_correlation']:8.3f} "
              f"{er:8.2f} {str(qualification[model]['passed']):>6}")

    out = {
        "passed": passed,
        "models": MODELS,
        "families": FAMILY,
        "k": K,
        "seed": SEED,
        "n_fit_controls": len(fit_idx),
        "n_validation_controls": len(val_idx),
        "qualification": qualification,
        "primary": primary,
        "shuffled_correspondence": shuffled,
        "independent_random_bases": random_control,
        "evidence_boundary": (
            "Held-out control-state alignment only; no perturbation labels or "
            "directions were evaluated. scVI and PCA encoder parameters were "
            "trained/fitted on all controls, but GCCA, depth correction, and "
            "all standardization used fitting controls only."),
    }
    qlib.save_result("e40_modal_control_alignment", out)
    print("E40_DONE")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
