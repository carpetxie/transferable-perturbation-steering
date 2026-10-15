"""E42: nested zero-label selection of views, K, donors, and weighting.

For each outer target, all of its perturbation outcomes are hidden during
selection. Candidate rules are scored by leave-one-model-out validation over
the other five models, frozen, and only then evaluated on the outer target.
"""
from __future__ import annotations

import itertools

import qlib
import vdlib


K_VALUES = (8, 16, 24, 32, 48)


def ordered(models):
    chosen = set(models)
    return tuple(m for m in vdlib.PRIMARY if m in chosen)


def balanced_pool(outer, historical):
    if vdlib.FAMILY[outer] == "geneformer":
        return ordered(m for m in historical if vdlib.FAMILY[m] != "geneformer")
    pool = [m for m in historical if vdlib.FAMILY[m] != "geneformer"]
    geneformers = [m for m in historical if vdlib.FAMILY[m] == "geneformer"]
    if geneformers:
        representative = "gf2" if "gf2" in geneformers else geneformers[0]
        pool.append(representative)
    return ordered(pool)


def candidate_pools(outer):
    historical = tuple(m for m in vdlib.PRIMARY if m != outer)
    pools = []
    pools.extend((m,) for m in historical)
    pools.extend(tuple(pair) for pair in itertools.combinations(historical, 2))
    original = ordered(m for m in historical if m in vdlib.ORIGINAL)
    if original:
        pools.append(original)
    pools.append(balanced_pool(outer, historical))
    pools.append(historical)
    unique = []
    seen = set()
    for pool in pools:
        pool = ordered(pool)
        if pool and pool not in seen:
            seen.add(pool)
            unique.append(pool)
    return historical, unique


def donor_specs(pool):
    specs = [(f"single_{m}", (m,), "equal") for m in pool]
    specs.append(("equal_all", pool, "equal"))
    original = ordered(m for m in pool if m in vdlib.ORIGINAL)
    new = ordered(m for m in pool if m in vdlib.NEW)
    if original:
        specs.append(("original_only", original, "equal"))
    if new:
        specs.append(("scvi_pca_only", new, "equal"))
    specs.append(("architecture_balanced", pool, "architecture_balanced"))
    specs.append(("reliability_weighted", pool, "reliability"))
    # Preserve rule labels while removing exact duplicate rule realizations.
    unique = []
    seen = set()
    for label, donors, rule in specs:
        key = (tuple(donors), rule)
        if key not in seen:
            seen.add(key)
            unique.append((label, tuple(donors), rule))
    return unique


def get_space(cache, prepared_by_k, views, k):
    key = (k, ordered(views))
    if key not in cache:
        cache[key] = vdlib.fit_space(prepared_by_k[k], key[1], k=k)
    return cache[key]


def compact(result):
    return {
        key: value for key, value in result.items()
        if key not in {"per_perturbation"}
    }


def candidate_record(k, pool, donor_label, donors, rule, inner_scores):
    return {
        "k": k,
        "view_pool_rule": list(pool),
        "donor_label": donor_label,
        "donor_pool_rule": list(donors),
        "weighting": rule,
        "inner_scores": inner_scores,
        "inner_mean_pct_ceiling": float(sum(inner_scores.values()) / len(inner_scores)),
    }


def evaluate_outer(native, prepared_by_k, cache, gene, outer, candidate):
    views = ordered(set(candidate["view_pool_rule"]) | {outer})
    donors = tuple(candidate["donor_pool_rule"])
    space = get_space(cache, prepared_by_k, views, candidate["k"])
    result = vdlib.evaluate(
        native, prepared_by_k[candidate["k"]], space, outer, donors,
        candidate["weighting"], gene_endpoint=gene)
    result["selection_rule"] = candidate
    return result


def main():
    native = vdlib.load_native_deltas(vdlib.PRIMARY)
    prepared_by_k = {
        k: vdlib.prepare_controls(native, k=k) for k in K_VALUES
    }
    print("loading fixed control-only gene consumers", flush=True)
    gene = vdlib.load_gene_endpoint(native)
    space_cache = {}
    outer_rows = []

    for outer in vdlib.PRIMARY:
        print(f"outer target {outer}", flush=True)
        historical, pools = candidate_pools(outer)
        candidates = []
        for k in K_VALUES:
            prepared = prepared_by_k[k]
            for pool in pools:
                for donor_label, donor_pool, rule in donor_specs(pool):
                    inner_scores = {}
                    for inner in historical:
                        donors = tuple(m for m in donor_pool if m != inner)
                        if not donors:
                            continue
                        views = ordered(set(pool) | {inner})
                        space = get_space(space_cache, prepared_by_k, views, k)
                        result = vdlib.evaluate(
                            native, prepared, space, inner, donors, rule)
                        inner_scores[inner] = result["pct_native_ceiling"]
                    if len(inner_scores) >= 3:
                        candidates.append(candidate_record(
                            k, pool, donor_label, donor_pool, rule, inner_scores))

        selected = max(candidates, key=lambda row: row["inner_mean_pct_ceiling"])
        selected_result = evaluate_outer(
            native, prepared_by_k, space_cache, gene, outer, selected)

        # Frozen non-oracle comparators.
        fixed = candidate_record(
            24, historical, "equal_all", historical, "equal", {"fixed": 0.0})
        balanced = balanced_pool(outer, historical)
        balanced_baseline = candidate_record(
            24, balanced, "architecture_balanced", balanced,
            "architecture_balanced", {"fixed": 0.0})
        reliability_baseline = candidate_record(
            24, historical, "reliability_weighted", historical,
            "reliability", {"fixed": 0.0})
        single_candidates = [
            row for row in candidates
            if len(row["view_pool_rule"]) == 1
            and row["donor_label"].startswith("single_")
        ]
        best_single = max(
            single_candidates, key=lambda row: row["inner_mean_pct_ceiling"])

        comparators = {
            "fixed_six_equal_all": evaluate_outer(
                native, prepared_by_k, space_cache, gene, outer, fixed),
            "best_single_inner": evaluate_outer(
                native, prepared_by_k, space_cache, gene, outer, best_single),
            "architecture_balanced": evaluate_outer(
                native, prepared_by_k, space_cache, gene, outer,
                balanced_baseline),
            "reliability_weighted": evaluate_outer(
                native, prepared_by_k, space_cache, gene, outer,
                reliability_baseline),
        }

        # Post-reveal oracle is diagnostic only and never feeds selection.
        oracle_candidate = None
        oracle_score = -float("inf")
        for candidate in candidates:
            result = evaluate_outer(
                native, prepared_by_k, space_cache, None, outer, candidate)
            if result["pct_native_ceiling"] > oracle_score:
                oracle_score = result["pct_native_ceiling"]
                oracle_candidate = candidate
        oracle_result = evaluate_outer(
            native, prepared_by_k, space_cache, gene, outer, oracle_candidate)

        row = {
            "outer_target": outer,
            "n_candidates": len(candidates),
            "selected": selected_result,
            "comparators": comparators,
            "post_reveal_oracle_diagnostic": oracle_result,
        }
        outer_rows.append(row)
        fixed_score = comparators["fixed_six_equal_all"]["transfer_native"]
        print(f"  selected={selected_result['transfer_native']:.3f} "
              f"fixed={fixed_score:.3f} "
              f"oracle={oracle_result['transfer_native']:.3f} "
              f"K={selected['k']} V={selected['view_pool_rule']} "
              f"D={selected['donor_pool_rule']} {selected['weighting']}", flush=True)

    def aggregate(name):
        if name == "selected":
            values = [row["selected"]["pct_native_ceiling"] for row in outer_rows]
        elif name == "oracle":
            values = [row["post_reveal_oracle_diagnostic"]["pct_native_ceiling"]
                      for row in outer_rows]
        else:
            values = [row["comparators"][name]["pct_native_ceiling"]
                      for row in outer_rows]
        return float(sum(values) / len(values))

    aggregate_pct = {
        "selected": aggregate("selected"),
        "fixed_six_equal_all": aggregate("fixed_six_equal_all"),
        "best_single_inner": aggregate("best_single_inner"),
        "architecture_balanced": aggregate("architecture_balanced"),
        "reliability_weighted": aggregate("reliability_weighted"),
        "post_reveal_oracle": aggregate("oracle"),
    }
    print("aggregate % native ceiling:", aggregate_pct)
    out = {
        "k_grid": list(K_VALUES),
        "models": list(vdlib.PRIMARY),
        "selection_objective": (
            "mean inner-model transfer divided by each inner target's fixed "
            "native split-half ceiling"),
        "aggregate_pct_native_ceiling": aggregate_pct,
        "outer_rows": outer_rows,
        "gene_endpoint": {
            "n_hvg": gene["n_hvg"],
            "decoder_rank": gene["decoder_rank"],
            "decoder_alphas": gene["alphas"],
        },
        "evidence_boundary": (
            "Norman2019 nested model-level validation over six representations. "
            "Outer-target perturbations are used only after rule selection. The "
            "post-reveal oracle is descriptive and not deployable."),
    }
    qlib.save_result("e42_nested_zero_label_selection", out)
    print("E42_DONE")


if __name__ == "__main__":
    main()
