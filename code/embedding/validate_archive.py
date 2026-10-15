"""Fail-closed validation for a collected vec2vec Modal embedding archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PRIMARY = (
    "scgpt",
    "geneformer",
    "geneformer_v2",
    "geneformer_v3",
    "scvi",
    "pca",
)
SENSITIVITY = ("geneformer_cl",)


def _cell_signature(lib: np.ndarray, ngene: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(lib).view(np.uint8))
    digest.update(np.ascontiguousarray(ngene).view(np.uint8))
    return digest.hexdigest()


def validate(root: Path, expected_conditions: int = 237) -> dict:
    models = PRIMARY + SENSITIVITY
    by_model: dict[str, dict[str, Path]] = {model: {} for model in models}
    unknown = []
    for path in sorted(root.glob("*.npz")):
        try:
            condition, model = path.stem.rsplit("__", 1)
        except ValueError:
            unknown.append(path.name)
            continue
        if model in by_model:
            if condition in by_model[model]:
                raise AssertionError(f"duplicate {condition}/{model}")
            by_model[model][condition] = path
        else:
            unknown.append(path.name)

    condition_sets = {model: set(files) for model, files in by_model.items()}
    reference_conditions = condition_sets[PRIMARY[0]]
    for model, conditions in condition_sets.items():
        if len(conditions) != expected_conditions:
            raise AssertionError(
                f"{model}: expected {expected_conditions} conditions, got {len(conditions)}")
        if conditions != reference_conditions:
            missing = sorted(reference_conditions - conditions)
            extra = sorted(conditions - reference_conditions)
            raise AssertionError(f"{model}: condition mismatch missing={missing} extra={extra}")

    dimensions: dict[str, set[int]] = {model: set() for model in models}
    sizes: dict[str, int] = {}
    signatures: dict[str, str] = {}
    files_checked = 0
    for condition in sorted(reference_conditions):
        reference_lib = reference_ngene = None
        for model in models:
            path = by_model[model][condition]
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"Z", "lib", "ngene"}:
                    raise AssertionError(f"{path}: unexpected keys {data.files}")
                Z = np.asarray(data["Z"])
                lib = np.asarray(data["lib"])
                ngene = np.asarray(data["ngene"])
            if Z.ndim != 2 or lib.ndim != 1 or ngene.ndim != 1:
                raise AssertionError(
                    f"{path}: bad shapes Z={Z.shape} lib={lib.shape} ngene={ngene.shape}")
            if len(Z) == 0 or len(Z) != len(lib) or len(Z) != len(ngene):
                raise AssertionError(
                    f"{path}: row mismatch Z={len(Z)} lib={len(lib)} ngene={len(ngene)}")
            if not (np.isfinite(Z).all() and np.isfinite(lib).all()
                    and np.isfinite(ngene).all()):
                raise AssertionError(f"{path}: non-finite value")
            dimensions[model].add(int(Z.shape[1]))
            if reference_lib is None:
                reference_lib = lib
                reference_ngene = ngene
                sizes[condition] = len(Z)
                signatures[condition] = _cell_signature(lib, ngene)
            else:
                if len(Z) != sizes[condition]:
                    raise AssertionError(
                        f"{condition}/{model}: expected {sizes[condition]} rows, got {len(Z)}")
                if not (np.array_equal(lib, reference_lib)
                        and np.array_equal(ngene, reference_ngene)):
                    raise AssertionError(
                        f"{condition}/{model}: cell order/QC arrays differ from {PRIMARY[0]}")
                if _cell_signature(lib, ngene) != signatures[condition]:
                    raise AssertionError(f"{condition}/{model}: cell signature mismatch")
            files_checked += 1

    constant_dimensions = {}
    for model, values in dimensions.items():
        if len(values) != 1:
            raise AssertionError(f"{model}: non-constant embedding dimensions {sorted(values)}")
        constant_dimensions[model] = next(iter(values))

    if "control" not in sizes:
        raise AssertionError("control condition missing")
    result = {
        "valid": True,
        "root": str(root.resolve()),
        "primary_models": list(PRIMARY),
        "sensitivity_models": list(SENSITIVITY),
        "conditions_per_model": expected_conditions,
        "files_checked": files_checked,
        "total_cells_per_model": int(sum(sizes.values())),
        "control_cells": int(sizes["control"]),
        "embedding_dimensions": constant_dimensions,
        "cell_order_check": "exact per-condition lib and ngene arrays across all models",
        "unknown_npz_files": unknown,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-conditions", type=int, default=237)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.root, args.expected_conditions)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
