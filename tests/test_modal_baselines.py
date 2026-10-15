from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp


MODAL_FILE = Path(__file__).resolve().parents[1] / "modal" / "vec2vec_embed.py"
SOURCE = MODAL_FILE.read_text()
TREE = ast.parse(SOURCE)
HELPERS = {"_fit_control_pca", "_control_bounds", "_save_condition_embeddings"}
NAMESPACE = {"os": os}
helper_nodes = [node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(MODAL_FILE), "exec"),
     NAMESPACE)


def test_control_pca_is_finite_and_deterministic():
    rng = np.random.default_rng(4)
    X = sp.csr_matrix(rng.poisson(1.2, size=(48, 30)).astype(np.float32))
    first = NAMESPACE["_fit_control_pca"](X, n_comp=8, n_hvg=20,
                                          min_detection=0.0)
    second = NAMESPACE["_fit_control_pca"](X, n_comp=8, n_hvg=20,
                                           min_detection=0.0)
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b)
        assert np.isfinite(a).all()
    hvg, W, mu = first
    assert hvg.shape == (20,)
    assert W.shape == (20, 8)
    assert mu.shape == (20,)


def test_control_bounds_uses_designed_control_label():
    meta = {
        "conds": np.array(["pert_a", "control", "pert_b"], dtype=object),
        "bounds": np.array([0, 5, 17, 24]),
        "ctrl": np.array("control", dtype=object),
    }
    assert NAMESPACE["_control_bounds"](meta) == (5, 17)


def test_condition_files_are_qlib_compatible(tmp_path):
    X = sp.csr_matrix(np.array([
        [1, 0, 2], [0, 3, 0], [2, 1, 0], [0, 0, 4],
    ], dtype=np.float32))
    Z = np.arange(12, dtype=np.float32).reshape(4, 3)
    NAMESPACE["_save_condition_embeddings"](
        str(tmp_path), "pca", ["control", "pert/a"],
        np.array([0, 2, 4]), X, Z)

    control = np.load(tmp_path / "control__pca.npz")
    pert = np.load(tmp_path / "pert-a__pca.npz")
    np.testing.assert_array_equal(control["Z"], Z[:2])
    np.testing.assert_array_equal(pert["Z"], Z[2:])
    np.testing.assert_array_equal(control["lib"], np.array([3, 3]))
    np.testing.assert_array_equal(pert["ngene"], np.array([2, 1]))


def test_modal_source_is_codex_local_and_includes_new_models():
    assert "Claude/" not in SOURCE
    assert '"scvi-tools==1.3.3"' in SOURCE
    assert "geneformer_cl,scvi,pca" in SOURCE
