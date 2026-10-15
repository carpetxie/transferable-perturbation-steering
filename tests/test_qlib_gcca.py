from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "queue"))
import qlib


def test_gcca_keeps_at_least_requested_rank_for_concentrated_view():
    rng = np.random.default_rng(0)
    n, d, k = 256, 40, 24
    shared = rng.normal(size=(n, d))
    # Full-rank views whose last dimensions carry little variance. A pure
    # variance cutoff would retain fewer than k directions.
    scales = np.geomspace(1.0, 1e-3, d)
    views = {
        "a": shared * scales,
        "b": (shared + 0.01 * rng.normal(size=shared.shape)) * scales,
    }
    result = qlib.fit_gcca(views, k=k)
    assert result["k"] == k
    assert result["P"]["a"].shape[1] >= k
    assert result["P"]["b"].shape[1] >= k
