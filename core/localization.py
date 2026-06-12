"""Spatial localization: grid construction, Gaussian weights, event positioning."""

import numpy as np
from scipy.spatial.distance import cdist


def build_spatial_grid(xlim: tuple = (0, 80),
                       ylim: tuple = (0, 120),
                       step: float = 1.0) -> np.ndarray:
    """Build dense 2-D localization grid.

    Args:
        xlim: (xmin, xmax) in cm
        ylim: (ymin, ymax) in cm
        step: grid resolution in cm

    Returns:
        xy_grid: (nGrid x 2) array of [x, y] positions in cm.
    """
    xs = np.arange(xlim[0], xlim[1] + step, step)
    ys = np.arange(ylim[0], ylim[1] + step, step)
    xg, yg = np.meshgrid(xs, ys)
    return np.column_stack([xg.ravel(), yg.ravel()])


def precompute_spatial_weights(xy_meas: np.ndarray,
                                xy_grid: np.ndarray,
                                sigma_spatial: float = 30.0) -> np.ndarray:
    """Precompute Gaussian spatial weighting kernel.

    Args:
        xy_meas:        (C x 2) sensor positions in cm
        xy_grid:        (nGrid x 2) grid positions in cm
        sigma_spatial:  Gaussian width in cm

    Returns:
        W_proto: (C x nGrid) weight matrix.
    """
    dists = cdist(xy_meas, xy_grid)                        # (C x nGrid)
    return np.exp(-dists**2 / (2.0 * sigma_spatial**2))


def localize_events_weighted_centroid(snaps: np.ndarray,
                                      xy_meas: np.ndarray,
                                      top_n: int = 4) -> tuple:
    """Weighted centroid localization using sqrt-compressed amplitudes.

    Implements the method from Henninger et al. (2020) JEB:
        r̂ = Σ(√Aᵢ · rᵢ) / Σ(√Aᵢ)
    using only the top_n electrodes with the largest amplitude per event.

    Args:
        snaps:   (E x C) amplitude snapshots
        xy_meas: (C x 2) electrode positions in cm
        top_n:   number of electrodes to use per event (≤ C)

    Returns:
        X, Y: (E,) estimated event positions in cm.
    """
    vals = np.sqrt(np.maximum(snaps, 0.0))          # sqrt compression

    if top_n < xy_meas.shape[0]:
        # keep only top_n amplitudes per event; zero the rest
        thresh = np.sort(vals, axis=1)[:, -top_n:].min(axis=1, keepdims=True)
        vals = np.where(vals >= thresh, vals, 0.0)

    row_sums = vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.finfo(float).eps
    weights = vals / row_sums                        # (E x C), sum-normalised

    X = weights @ xy_meas[:, 0]
    Y = weights @ xy_meas[:, 1]
    return X, Y


def localize_events(snaps: np.ndarray,
                    W_proto: np.ndarray,
                    xy_grid: np.ndarray) -> tuple:
    """Localize EOD events on the spatial grid via amplitude-weighted voting.

    Dynamic range is compressed with a cube-root before normalisation so
    that a single hot channel does not dominate the fingerprint.

    Args:
        snaps:   (E x C) amplitude snapshots
        W_proto: (C x nGrid) Gaussian weight matrix
        xy_grid: (nGrid x 2) grid positions in cm

    Returns:
        X, Y: (E,) estimated event positions in cm.
    """
    vals = np.cbrt(np.maximum(snaps, 0.0))                # cube-root compression
    row_sums = vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.finfo(float).eps
    vals /= row_sums                                       # per-event normalise
    weighted = np.einsum('ij,jk->ik', vals, W_proto)       # (E x nGrid) – avoids BLAS crash
    k_max = np.argmax(weighted, axis=1)
    return xy_grid[k_max, 0], xy_grid[k_max, 1]
