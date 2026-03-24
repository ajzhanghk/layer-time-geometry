"""
Tests for layer_time/capacity.py — compositional capacity metrics.
No model loading required — all tests use synthetic data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy.linalg import expm

import layer_time_geometry as ltg
from layer_time.capacity import (
    CapacityProfile,
    lift_operator_to_full_space,
    compute_skew_generators,
    commutator_matrix,
    accumulated_noncommutativity,
    effective_capacity,
    capacity_concentration,
    compute_capacity_profile,
    _layer_contributions,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

np.random.seed(42)

L, T, p = 5, 8, 64
k = 16


@pytest.fixture
def raw_hidden():
    """Simulated raw hidden states (L, T, p)."""
    base = np.random.randn(1, 1, p) * 5
    layer_drift = np.cumsum(np.random.randn(L, 1, p) * 0.3, axis=0)
    token_drift = np.cumsum(np.random.randn(1, T, p) * 0.1, axis=1)
    noise = np.random.randn(L, T, p) * 0.5
    return (base + layer_drift + token_drift + noise).astype(np.float32)


@pytest.fixture
def metric(raw_hidden):
    H_flat = raw_hidden.reshape(-1, p)
    return ltg.estimate_metric(H_flat, n_components=k)


@pytest.fixture
def whitened(raw_hidden, metric):
    return ltg.whiten(raw_hidden, metric)


# ── Test lift_operator_to_full_space ─────────────────────────────────────────


class TestLiftOperator:
    def test_orthogonality(self, whitened):
        """Lifted operator should be orthogonal in full space."""
        _, _, pk = whitened.shape
        op = ltg.layer_operator(whitened, 1)
        U_full = lift_operator_to_full_space(op, pk)
        # U_full @ U_full^T should be identity
        np.testing.assert_allclose(
            U_full @ U_full.T, np.eye(pk), atol=1e-6,
        )

    def test_subspace_action(self, whitened):
        """Lifted operator should match original in the token subspace."""
        _, _, pk = whitened.shape
        op = ltg.layer_operator(whitened, 1)
        U_full = lift_operator_to_full_space(op, pk)
        V = op.V  # (pk, r)
        # In subspace: V^T U_full V should equal U_sub
        U_recovered = V.T @ U_full @ V
        np.testing.assert_allclose(U_recovered, op.U, atol=1e-6)

    def test_complement_identity(self, whitened):
        """Lifted operator should be identity on orthogonal complement."""
        _, _, pk = whitened.shape
        op = ltg.layer_operator(whitened, 1)
        U_full = lift_operator_to_full_space(op, pk)
        V = op.V
        proj = V @ V.T
        complement_proj = np.eye(pk) - proj
        # Pick a random vector in the complement
        v = complement_proj @ np.random.randn(pk)
        if np.linalg.norm(v) > 1e-8:
            np.testing.assert_allclose(U_full @ v, v, atol=1e-6)

    def test_known_2d_rotation(self):
        """Lift a known 2D rotation embedded in 10D."""
        from layer_time_geometry import OperatorDecomposition

        pk = 10
        r = 2
        theta = 0.3
        U_sub = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        # Embed in first two dimensions
        V = np.zeros((pk, r))
        V[0, 0] = 1.0
        V[1, 1] = 1.0

        op = OperatorDecomposition(
            T_op=U_sub, U=U_sub, P=np.eye(r), V=V, rank=r,
        )
        U_full = lift_operator_to_full_space(op, pk)

        # Should rotate in dims 0,1 and be identity elsewhere
        expected = np.eye(pk)
        expected[:2, :2] = U_sub
        np.testing.assert_allclose(U_full, expected, atol=1e-12)


# ── Test commutator_matrix ───────────────────────────────────────────────────


class TestCommutatorMatrix:
    def test_commuting_matrices_zero(self):
        """Commutator of identical matrices should be zero."""
        A = np.random.randn(10, 10)
        A = 0.5 * (A - A.T)  # skew-symmetric
        C = commutator_matrix([A, A, A])
        np.testing.assert_allclose(C, 0.0, atol=1e-12)

    def test_known_so3_commutator(self):
        """Test against known SO(3) structure constants.

        [L_x, L_y] = L_z with specific normalization.
        """
        # SO(3) generators
        Lx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        Ly = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
        Lz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

        C = commutator_matrix([Lx, Ly, Lz])

        # [Lx, Ly] = Lz, so ||[Lx, Ly]||_F = ||Lz||_F = sqrt(2)
        expected_xy = np.linalg.norm(Lz, "fro")
        assert abs(C[0, 1] - expected_xy) < 1e-12

        # [Lx, Lz] = -Ly, so ||[Lx, Lz]||_F = ||Ly||_F = sqrt(2)
        expected_xz = np.linalg.norm(Ly, "fro")
        assert abs(C[0, 2] - expected_xz) < 1e-12

    def test_symmetry(self):
        """Commutator norm matrix should be symmetric."""
        mats = [0.5 * (m - m.T) for m in [np.random.randn(5, 5) for _ in range(4)]]
        C = commutator_matrix(mats)
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_zero_diagonal(self):
        """Diagonal should be zero ([A, A] = 0)."""
        mats = [0.5 * (m - m.T) for m in [np.random.randn(5, 5) for _ in range(3)]]
        C = commutator_matrix(mats)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-14)

    def test_shape(self):
        n = 7
        mats = [0.5 * (m - m.T) for m in [np.random.randn(4, 4) for _ in range(n)]]
        C = commutator_matrix(mats)
        assert C.shape == (n, n)


# ── Test capacity statistics ─────────────────────────────────────────────────


class TestCapacityStatistics:
    def test_C_acc_zero_for_commuting(self):
        """Diagonal (block) skew matrices that commute should give C_acc ~ 0."""
        # Two skew matrices in disjoint 2D subspaces commute
        A1 = np.zeros((4, 4))
        A1[0, 1] = 1.0
        A1[1, 0] = -1.0

        A2 = np.zeros((4, 4))
        A2[2, 3] = 1.0
        A2[3, 2] = -1.0

        C = commutator_matrix([A1, A2])
        assert accumulated_noncommutativity(C) < 1e-12

    def test_C_acc_positive_for_noncommuting(self):
        Lx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        Ly = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
        C = commutator_matrix([Lx, Ly])
        assert accumulated_noncommutativity(C) > 0

    def test_effective_capacity_zero_without_dependency(self):
        """C_eff should be 0 when D_layer is all zeros."""
        n = 4
        comm = np.ones((n, n)) - np.eye(n)
        D = np.zeros(n + 2)  # L = n + 2 (skip_first adds offset)
        assert effective_capacity(comm, D, skip_first=True) == 0.0

    def test_effective_capacity_scales_with_dependency(self):
        """C_eff should increase with larger dependency values."""
        n = 3
        comm = np.ones((n, n)) - np.eye(n)
        D_low = np.ones(n + 2) * 0.1
        D_high = np.ones(n + 2) * 10.0
        c_low = effective_capacity(comm, D_low, skip_first=True)
        c_high = effective_capacity(comm, D_high, skip_first=True)
        assert c_high > c_low

    def test_concentration_all_final(self):
        """When all mass is in final layers, concentration should be 1.0."""
        n = 5
        C = np.zeros((n, n))
        # Put mass only between last two layers
        C[3, 4] = 1.0
        C[4, 3] = 1.0
        assert capacity_concentration(C, n_final=2) == 1.0

    def test_concentration_none_final(self):
        """When no mass in final layers, concentration should be 0."""
        n = 6
        C = np.zeros((n, n))
        C[0, 1] = 1.0
        C[1, 0] = 1.0
        # n_final=1 means only layer index 5, which has no mass
        assert capacity_concentration(C, n_final=1) == 0.0

    def test_layer_contributions_shape(self):
        n = 4
        C = np.random.rand(n, n)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 0)
        contrib = _layer_contributions(C)
        assert contrib.shape == (n,)
        assert np.all(contrib >= 0)


# ── Test compute_skew_generators ─────────────────────────────────────────────


class TestSkewGenerators:
    def test_exact_skew_symmetry(self, whitened):
        """All generators should be skew-symmetric."""
        gens = compute_skew_generators(whitened, method="exact")
        for A in gens:
            np.testing.assert_allclose(A, -A.T, atol=1e-6)

    def test_bivector_skew_symmetry(self, whitened):
        """Bivector generators should be skew-symmetric."""
        gens = compute_skew_generators(whitened, method="bivector")
        for A in gens:
            np.testing.assert_allclose(A, -A.T, atol=1e-12)

    def test_generator_count_skip_first(self, whitened):
        """Should return L-2 generators when skip_first=True."""
        gens = compute_skew_generators(whitened, skip_first=True)
        assert len(gens) == whitened.shape[0] - 2

    def test_generator_count_no_skip(self, whitened):
        """Should return L-1 generators when skip_first=False."""
        gens = compute_skew_generators(whitened, skip_first=False)
        assert len(gens) == whitened.shape[0] - 1

    def test_generator_shape(self, whitened):
        """Each generator should be (k, k)."""
        _, _, pk = whitened.shape
        gens = compute_skew_generators(whitened, method="exact")
        for A in gens:
            assert A.shape == (pk, pk)


# ── Test full pipeline ───────────────────────────────────────────────────────


class TestCapacityProfile:
    def test_without_dependency(self, whitened):
        """Should work without D_layer, C_eff = 0."""
        profile = compute_capacity_profile(whitened, D_layer=None)
        assert isinstance(profile, CapacityProfile)
        assert profile.C_eff == 0.0
        assert profile.C_acc >= 0
        assert 0 <= profile.cconc_acc <= 1.0

    def test_with_dependency(self, whitened):
        """Should compute nonzero C_eff with positive dependency."""
        D = np.random.rand(whitened.shape[0]) + 0.1
        profile = compute_capacity_profile(whitened, D_layer=D)
        assert profile.C_eff > 0
        assert profile.C_acc > 0

    def test_commutator_norms_shape(self, whitened):
        """Commutator norms should be (L-2, L-2)."""
        profile = compute_capacity_profile(whitened)
        n = whitened.shape[0] - 2
        assert profile.commutator_norms.shape == (n, n)

    def test_bivector_method(self, whitened):
        """Bivector method should produce valid profile."""
        profile = compute_capacity_profile(whitened, method="bivector")
        assert profile.method == "bivector"
        assert profile.C_acc >= 0

    def test_layer_contributions_sum(self, whitened):
        """Sum of layer contributions should be 2 * C_acc (each pair counted from both sides)."""
        profile = compute_capacity_profile(whitened)
        # Each commutator_norms[i,j] appears in contrib[i] and contrib[j]
        expected = 2 * profile.C_acc
        np.testing.assert_allclose(
            profile.layer_contributions.sum(), expected, rtol=1e-10,
        )
