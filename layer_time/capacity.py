"""Compositional capacity metrics for scaling analysis.

Computes accumulated non-commutativity of layerwise skew generators
and dependency-weighted effective capacity, as defined in the
scaling-as-compositional-capacity framework.

The key quantities are:

    C_acc  = sum_{i<j} w_{ij} ||[A^(i), A^(j)]||_F
    C_eff  = sum_{i<j} w_{ij} ||[A^(i), A^(j)]||_F sqrt(D_i D_j)
    cconc  = (final-layer commutator mass) / (total commutator mass)

where A^(l) is the skew-symmetric generator of the rotation U^(l)
from the polar decomposition T^(l) = U^(l) P^(l).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import layer_time_geometry as ltg_backend
from layer_time_geometry import OperatorDecomposition


@dataclass
class CapacityProfile:
    """Compositional capacity metrics for a single sample.

    Attributes:
        A_generators: List of (p, p) skew-symmetric matrices, one per
            layer transition (skipping layer 0).
        commutator_norms: (n, n) matrix where entry (i, j) =
            ||[A^(i), A^(j)]||_F.  Symmetric, zero diagonal.
        C_acc: Accumulated non-commutativity (unweighted).
        C_eff: Dependency-weighted effective capacity.
        cconc_acc: Fraction of commutator mass in final layers.
        layer_contributions: (n,) per-layer contribution to C_acc
            (sum of commutator norms involving that layer).
        D_layer: (L,) dependency profile used for weighting, or None.
        method: "exact" (logm of lifted operator) or "bivector".
    """

    A_generators: list[np.ndarray]
    commutator_norms: np.ndarray
    C_acc: float
    C_eff: float
    cconc_acc: float
    layer_contributions: np.ndarray
    D_layer: Optional[np.ndarray]
    method: str


# ── Lifting operators to full space ─────────────────────────────


def lift_operator_to_full_space(op: OperatorDecomposition, p: int) -> np.ndarray:
    """Lift a subspace rotation to the full p-dimensional space.

    Given U (r x r) in the token subspace with basis V (p x r),
    constructs U_full (p x p) = V U V^T + (I_p - V V^T), which
    acts as the rotation in the subspace and identity on its complement.

    Args:
        op: OperatorDecomposition from layer_operator().
        p: Full whitened dimension.

    Returns:
        U_full: (p, p) orthogonal matrix.
    """
    V = op.V  # (p, r)
    U_sub = op.U  # (r, r)
    proj = V @ V.T  # (p, p) projector onto token subspace
    U_full = V @ U_sub @ V.T + (np.eye(p) - proj)
    return U_full


# ── Skew generators ─────────────────────────────────────────────


def compute_skew_generators(
    H_tilde: np.ndarray,
    method: str = "exact",
    skip_first: bool = True,
) -> list[np.ndarray]:
    """Compute per-layer skew-symmetric generators A^(l).

    Args:
        H_tilde: (L, T, p) whitened hidden states.
        method: "exact" uses logm of lifted polar factor U;
                "bivector" uses bivector_field approximation.
        skip_first: If True, skip layer 0 transition (embedding to
            first layer, typically numerically degenerate).

    Returns:
        List of (p, p) skew-symmetric matrices.  Length is L-2 if
        skip_first, else L-1.
    """
    L, T, p = H_tilde.shape
    start = 1 if skip_first else 0

    if method == "bivector":
        dr = ltg_backend.decompose_direction_energy(H_tilde)
        H_hat = dr.H_hat
        return [ltg_backend.bivector_field(H_hat, l) for l in range(start, L - 1)]

    # method == "exact": lift polar factor to full space, then logm
    generators = []
    for l in range(start, L - 1):
        op = ltg_backend.layer_operator(H_tilde, l)
        U_full = lift_operator_to_full_space(op, p)
        A = ltg_backend.skew_generator(U_full)
        generators.append(A)
    return generators


# ── Commutator computation ──────────────────────────────────────


def commutator_matrix(A_list: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise commutator Frobenius norms.

    Args:
        A_list: List of n skew-symmetric (p, p) matrices.

    Returns:
        C: (n, n) symmetric matrix where C[i,j] = ||[A_i, A_j]||_F.
    """
    n = len(A_list)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            comm = A_list[i] @ A_list[j] - A_list[j] @ A_list[i]
            C[i, j] = np.linalg.norm(comm, "fro")
            C[j, i] = C[i, j]
    return C


# ── Capacity statistics ─────────────────────────────────────────


def accumulated_noncommutativity(
    comm_norms: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Accumulated non-commutativity C_acc.

    C_acc = sum_{i<j} w_{ij} * ||[A^(i), A^(j)]||_F

    Args:
        comm_norms: (n, n) commutator norm matrix.
        weights: (n, n) weight matrix.  If None, uniform weights (1.0).

    Returns:
        C_acc scalar.
    """
    n = comm_norms.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[i, j] if weights is not None else 1.0
            total += w * comm_norms[i, j]
    return total


def effective_capacity(
    comm_norms: np.ndarray,
    D_layer: np.ndarray,
    weights: Optional[np.ndarray] = None,
    skip_first: bool = True,
) -> float:
    """Dependency-weighted effective capacity C_eff.

    C_eff = sum_{i<j} w_{ij} * ||[A^(i), A^(j)]||_F * sqrt(D_i * D_j)

    where D_i is the dependency at the target layer of transition i.

    Args:
        comm_norms: (n, n) commutator norm matrix.
        D_layer: (L,) dependency profile from DependencyProfile.D_layer.
        weights: (n, n) weight matrix.  If None, uniform (1.0).
        skip_first: Whether layer 0 was skipped in generator computation.

    Returns:
        C_eff scalar.
    """
    n = comm_norms.shape[0]
    # Map commutator index i to D_layer index.
    # If skip_first: commutator index 0 corresponds to transition 1->2,
    # so target layer = i + 2.  Use D_layer[i + offset].
    offset = 2 if skip_first else 1

    total = 0.0
    for i in range(n):
        d_i = D_layer[min(i + offset, len(D_layer) - 1)]
        for j in range(i + 1, n):
            d_j = D_layer[min(j + offset, len(D_layer) - 1)]
            w = weights[i, j] if weights is not None else 1.0
            total += w * comm_norms[i, j] * np.sqrt(max(d_i * d_j, 0.0))
    return total


def capacity_concentration(
    comm_norms: np.ndarray,
    n_final: int = 3,
) -> float:
    """Fraction of commutator mass in final layers.

    cconc = (sum of ||[A^(i), A^(j)]||_F where i or j is in final
    n_final layers) / (total C_acc).

    Args:
        comm_norms: (n, n) commutator norm matrix.
        n_final: Number of final layers to consider.

    Returns:
        Concentration ratio in [0, 1].
    """
    n = comm_norms.shape[0]
    total = 0.0
    final_total = 0.0
    final_start = max(0, n - n_final)

    for i in range(n):
        for j in range(i + 1, n):
            val = comm_norms[i, j]
            total += val
            if i >= final_start or j >= final_start:
                final_total += val

    if total < 1e-12:
        return 0.0
    return final_total / total


def _layer_contributions(comm_norms: np.ndarray) -> np.ndarray:
    """Per-layer contribution: sum of commutator norms involving that layer.

    Args:
        comm_norms: (n, n) commutator norm matrix.

    Returns:
        (n,) array of per-layer contributions.
    """
    n = comm_norms.shape[0]
    contrib = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                contrib[i] += comm_norms[i, j]
    return contrib


# ── Main entry point ────────────────────────────────────────────


def compute_capacity_profile(
    H_tilde: np.ndarray,
    D_layer: Optional[np.ndarray] = None,
    method: str = "exact",
    n_final: int = 3,
    weights: Optional[np.ndarray] = None,
) -> CapacityProfile:
    """Compute full compositional capacity profile for a single sample.

    Args:
        H_tilde: (L, T, p) whitened hidden states.
        D_layer: (L,) dependency profile.  If None, C_eff is set to 0.
        method: "exact" or "bivector".
        n_final: Number of final layers for concentration.
        weights: Optional (n, n) weight matrix for C_acc and C_eff.

    Returns:
        CapacityProfile with all capacity metrics.
    """
    A_list = compute_skew_generators(H_tilde, method=method, skip_first=True)
    comm_norms = commutator_matrix(A_list)
    C_acc = accumulated_noncommutativity(comm_norms, weights)

    if D_layer is not None:
        C_eff = effective_capacity(comm_norms, D_layer, weights, skip_first=True)
    else:
        C_eff = 0.0

    cconc = capacity_concentration(comm_norms, n_final)
    contrib = _layer_contributions(comm_norms)

    return CapacityProfile(
        A_generators=A_list,
        commutator_norms=comm_norms,
        C_acc=C_acc,
        C_eff=C_eff,
        cconc_acc=cconc,
        layer_contributions=contrib,
        D_layer=D_layer,
        method=method,
    )
