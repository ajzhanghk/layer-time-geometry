"""
Autoregressive generation with frontier hidden-state extraction.

Extends the static hidden-state analysis to decode time by extracting
the frontier column (hidden states at the newest token position across
all layers) at each generation step.

Key design choices:
- Uses KV cache for efficient generation.
- Computes a FIXED whitening metric from the prompt's hidden states,
  then applies it to all frontier columns. This ensures gauge
  consistency across decode steps (Section 16.10 of the book).
- Frontier GA quantities (Rodrigues rotor, bivector, angle) are
  computed per-layer per-step, giving a 2D (layers × steps) field.
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path
try:
    import layer_time_geometry as backend
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import layer_time_geometry as backend

from .algebra import (
    Bivector, grade_decomposition, directional_flow_ratio,
    binet_cauchy_cosine,
)


# ── Generation with frontier extraction ─────────────────────────


@dataclass
class FrontierStep:
    """Hidden-state frontier at one decode step."""
    step: int                    # decode step index (0 = prompt's last token)
    token_id: int                # token id at this position
    token_str: str               # decoded token string
    frontier_raw: np.ndarray     # (L+1, p) raw hidden states across layers
    logits_top5: Optional[np.ndarray] = None   # top-5 logit values
    logits_top5_ids: Optional[np.ndarray] = None  # top-5 token ids


@dataclass
class GenerationResult:
    """Complete result of autoregressive generation with frontier extraction."""
    prompt: str
    prompt_tokens: list[str]
    prompt_length: int           # T_0
    n_steps: int                 # number of generated tokens
    n_layers: int                # L+1 (including embedding)
    hidden_dim: int              # p

    steps: list[FrontierStep]    # one per step (0..n_steps)
    prompt_hidden_states: np.ndarray  # (L+1, T_0, p) full prompt hidden states

    # Whitened frontier data (set after whitening)
    metric: Optional[object] = None  # MetricStructure
    frontier_whitened: Optional[np.ndarray] = None  # (n_steps+1, L, k)
    k: Optional[int] = None

    @property
    def generated_text(self) -> str:
        return "".join(s.token_str for s in self.steps[1:])

    @property
    def frontier_raw(self) -> np.ndarray:
        """Stack all frontier columns: (n_steps+1, L+1, p)."""
        return np.stack([s.frontier_raw for s in self.steps])


def generate_with_frontier(hf_model, tokenizer, prompt: str,
                           n_steps: int = 50,
                           device: str = "cuda",
                           temperature: float = 0.0,
                           save_top_k_logits: int = 5) -> GenerationResult:
    """
    Run autoregressive generation and extract frontier hidden states.

    At each decode step, extracts the hidden-state column at the newest
    token position across all transformer layers. Uses KV cache so each
    step processes only the new token.

    Args:
        hf_model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text.
        n_steps: Number of tokens to generate.
        device: Torch device string.
        temperature: 0.0 for greedy decoding, >0 for sampling.
        save_top_k_logits: Save top-k logits at each step.

    Returns:
        GenerationResult with frontier columns and prompt hidden states.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    T_0 = input_ids.shape[1]
    prompt_token_ids = input_ids[0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    steps = []
    prompt_H = None
    past_key_values = None
    eos_id = getattr(tokenizer, 'eos_token_id', None)

    for step in range(n_steps + 1):
        with torch.no_grad():
            if step == 0:
                outputs = hf_model(
                    input_ids,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hs = outputs.hidden_states  # tuple of (1, T_0, p)
                # Full prompt hidden states
                prompt_H = torch.stack(
                    [h.squeeze(0) for h in hs]
                ).float().cpu().numpy()  # (L+1, T_0, p)
                # Frontier = last token of prompt
                frontier = np.stack(
                    [h[0, -1, :].float().cpu().numpy() for h in hs]
                )  # (L+1, p)
                token_id = prompt_token_ids[-1]
                token_str = prompt_tokens[-1]
            else:
                outputs = hf_model(
                    next_token_id,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hs = outputs.hidden_states
                # With KV cache, each layer returns (1, 1, p) for the new token
                frontier = np.stack(
                    [h[0, 0, :].float().cpu().numpy() for h in hs]
                )  # (L+1, p)

            past_key_values = outputs.past_key_values

            # Logits for the current position
            logits = outputs.logits[0, -1, :].float()

            # Top-k logits
            top_vals, top_ids = torch.topk(logits, save_top_k_logits)
            top_vals_np = top_vals.cpu().numpy()
            top_ids_np = top_ids.cpu().numpy()

            # Select next token
            if temperature <= 0:
                next_id = logits.argmax()
            else:
                probs = torch.softmax(logits / temperature, dim=0)
                next_id = torch.multinomial(probs, 1)[0]

            next_token_id = next_id.unsqueeze(0).unsqueeze(0)

            if step > 0:
                token_id = generated_id
                token_str = tokenizer.decode([generated_id])

            # Save the step
            fs = FrontierStep(
                step=step,
                token_id=token_id,
                token_str=token_str,
                frontier_raw=frontier,
                logits_top5=top_vals_np,
                logits_top5_ids=top_ids_np,
            )
            steps.append(fs)

            # Record the generated token for next iteration
            generated_id = next_id.item()

            # Check EOS
            if step > 0 and eos_id is not None and token_id == eos_id:
                break

    L_plus_1 = prompt_H.shape[0]
    p = prompt_H.shape[2]

    return GenerationResult(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        prompt_length=T_0,
        n_steps=len(steps) - 1,
        n_layers=L_plus_1,
        hidden_dim=p,
        steps=steps,
        prompt_hidden_states=prompt_H,
    )


# ── Whitening ────────────────────────────────────────────────────


def whiten_frontier(gen_result: GenerationResult,
                    whiten_components: int = 256) -> GenerationResult:
    """
    Whiten all frontier columns using a fixed metric from the prompt.

    Computes the whitening metric from the prompt's full hidden states
    (all layers, all tokens), then applies it to every frontier column.
    This ensures gauge consistency: all decode steps live in the same
    coordinate frame.

    Args:
        gen_result: GenerationResult from generate_with_frontier().
        whiten_components: Number of PCA components for whitening.

    Returns:
        The same GenerationResult with metric, frontier_whitened, and k set.
    """
    # Use prompt hidden states (skip layer 0) for metric estimation
    prompt_H = gen_result.prompt_hidden_states[1:]  # (L, T_0, p)
    L, T_0, p = prompt_H.shape

    H_flat = prompt_H.reshape(L * T_0, p)
    k = min(whiten_components, L * T_0 - 1, p)
    metric = backend.estimate_metric(H_flat, n_components=k)

    # Whiten all frontier columns (skip layer 0)
    n_total = len(gen_result.steps)
    frontier_raw = gen_result.frontier_raw[:, 1:, :]  # (n_total, L, p)
    frontier_w = np.zeros((n_total, L, metric.k))

    for s in range(n_total):
        frontier_w[s] = backend.whiten(frontier_raw[s], metric)  # (L, k)

    gen_result.metric = metric
    gen_result.frontier_whitened = frontier_w
    gen_result.k = metric.k

    return gen_result


# ── Frontier GA quantities ───────────────────────────────────────


@dataclass
class FrontierGA:
    """GA quantities at the decode frontier across all steps.

    Attributes:
        angles: (n_transitions, n_steps) frontier rotation angle at each
                layer transition and decode step.
        bivectors: list of lists — bivectors[l][s] is the Bivector at
                   layer transition l, decode step s.
        incremental_norms: (n_transitions, n_steps-1) ||ΔB|| between steps.
        commutator_norms: (n_transitions, n_steps-1) ||[B_s, B_{s-1}]||.
        plane_drift: (n_transitions, n_steps-1) 1 - |cos(B_s, B_{s-1})|.
        n_transitions: number of layer transitions.
        n_steps: number of decode steps (including prompt frontier).
    """
    angles: np.ndarray
    bivectors: list
    incremental_norms: np.ndarray
    commutator_norms: np.ndarray
    plane_drift: np.ndarray
    n_transitions: int
    n_steps: int


def compute_frontier_ga(gen_result: GenerationResult,
                        skip_first: bool = True) -> FrontierGA:
    """
    Compute frontier GA quantities from whitened frontier columns.

    For each decode step s and layer transition l, computes:
    - The rotation angle via arccos(dot product) of unit vectors.
    - The Cayley bivector (closed-form skew-symmetric matrix, no logm).

    Then computes step-to-step increments: bivector change, commutator
    between successive bivectors, and plane drift.

    Uses vectorised numpy operations where possible for speed.

    Args:
        gen_result: GenerationResult with frontier_whitened already set.
        skip_first: Skip the first layer transition (embedding -> layer 1).

    Returns:
        FrontierGA with all frontier GA quantities.
    """
    if gen_result.frontier_whitened is None:
        raise ValueError("Call whiten_frontier() first.")

    F = gen_result.frontier_whitened  # (n_total, L, k)
    n_total, L, k = F.shape
    start = 1 if skip_first else 0
    n_trans = L - 1 - start

    # ── Vectorised angle computation ──
    # H_l[s, li, :] = F[s, start+li, :]
    H_l = F[:, start:L-1, :]     # (n_total, n_trans, k)
    H_l1 = F[:, start+1:L, :]    # (n_total, n_trans, k)

    # Normalise
    norms_l = np.linalg.norm(H_l, axis=2, keepdims=True)   # (n_total, n_trans, 1)
    norms_l1 = np.linalg.norm(H_l1, axis=2, keepdims=True)
    norms_l = np.maximum(norms_l, 1e-10)
    norms_l1 = np.maximum(norms_l1, 1e-10)
    U = H_l / norms_l   # (n_total, n_trans, k)
    V = H_l1 / norms_l1

    # Angle = arccos(u . v), clipped for numerical safety
    dots = np.sum(U * V, axis=2)  # (n_total, n_trans)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots).T  # (n_trans, n_total)

    # ── Cayley bivectors (closed-form, no logm) ──
    # A(u,v) = (v u^T - u v^T) / (1 + v^T u)
    # This is a rank-2 skew-symmetric matrix in k dimensions.
    # Store as list-of-lists of (k,k) arrays.
    bivector_mats = np.zeros((n_trans, n_total, k, k))

    for s in range(n_total):
        for li in range(n_trans):
            u = U[s, li]
            v = V[s, li]
            denom = 1.0 + np.dot(v, u)
            if denom > 1e-12:
                bivector_mats[li, s] = (np.outer(v, u) - np.outer(u, v)) / denom

    # Wrap as Bivector objects (lightweight)
    bivectors = [[None] * n_total for _ in range(n_trans)]
    for li in range(n_trans):
        for s in range(n_total):
            bivectors[li][s] = Bivector(matrix=bivector_mats[li, s], dim=k)

    # ── Incremental quantities (vectorised) ──
    B_curr = bivector_mats[:, 1:, :, :]   # (n_trans, n_total-1, k, k)
    B_prev = bivector_mats[:, :-1, :, :]

    # Incremental bivector norm
    delta = B_curr - B_prev
    incr_norms = np.sqrt(np.sum(delta ** 2, axis=(2, 3)))  # (n_trans, n_total-1)

    # Commutator: [B_s, B_{s-1}] = B_s @ B_{s-1} - B_{s-1} @ B_s
    comm = np.einsum('nlij,nljm->nlim', B_curr, B_prev) - \
           np.einsum('nlij,nljm->nlim', B_prev, B_curr)
    comm_norms = np.sqrt(np.sum(comm ** 2, axis=(2, 3)))  # (n_trans, n_total-1)

    # Plane drift: 1 - |cos(B_s, B_{s-1})|
    inner = np.sum(B_curr * B_prev, axis=(2, 3))  # Frobenius inner product
    norm_c = np.sqrt(np.sum(B_curr ** 2, axis=(2, 3)))
    norm_p = np.sqrt(np.sum(B_prev ** 2, axis=(2, 3)))
    denom_dp = norm_c * norm_p
    cos_sim = np.where(denom_dp > 1e-12, inner / denom_dp, 0.0)
    drift = 1.0 - np.abs(cos_sim)

    return FrontierGA(
        angles=angles,
        bivectors=bivectors,
        incremental_norms=incr_norms,
        commutator_norms=comm_norms,
        plane_drift=drift,
        n_transitions=n_trans,
        n_steps=n_total,
    )


# ── Frontier holonomy ────────────────────────────────────────────


def frontier_holonomy(gen_result: GenerationResult,
                      skip_first: bool = True,
                      device: str = "cuda") -> np.ndarray:
    """
    Compute holonomy scalar curvature at frontier plaquettes (GPU).

    At each decode step s >= 1, forms a 2-token mini-grid from the
    previous and current frontier columns and runs the GPU-accelerated
    curvature computation.

    Args:
        gen_result: GenerationResult with frontier_whitened set.
        skip_first: Skip the first layer.
        device: Torch device for GPU computation.

    Returns:
        (n_holo_layers, n_total) array of scalar curvature at frontier.
        Returns None if the prompt has fewer than 2 tokens.
    """
    if gen_result.frontier_whitened is None:
        raise ValueError("Call whiten_frontier() first.")

    F = gen_result.frontier_whitened  # (n_total, L, k)
    n_total, L, k = F.shape

    prompt_H = gen_result.prompt_hidden_states[1:]  # (L, T_0, p)
    T_0 = prompt_H.shape[1]
    if T_0 < 2:
        return None

    # Whiten the pre-frontier token from the prompt
    pre_frontier = backend.whiten(prompt_H[:, -2, :], gen_result.metric)  # (L, k)

    start = 1 if skip_first else 0
    n_holo_layers = L - 1 - start
    holo = np.zeros((n_holo_layers, n_total))

    for s in range(n_total):
        prev = pre_frontier if s == 0 else F[s - 1]  # (L, k)
        curr = F[s]  # (L, k)

        # Build mini-grid (L, 2, k)
        mini_grid = np.stack([prev, curr], axis=1)  # (L, 2, k)

        # Use GPU curvature computation: returns (L-1, 1) norms
        try:
            hmap = backend.curvature_gpu(mini_grid, device=device)
            holo[:, s] = hmap[start:, 0]
        except Exception:
            # Fallback: skip this step
            holo[:, s] = 0.0

    return holo


# ── Frontier grade decomposition ────────────────────────────────


@dataclass
class FrontierGradeProfile:
    """Grade-0 / grade-2 decomposition at the decode frontier.

    Attributes:
        grade0_norms: (n_transitions, n_steps) Frobenius norm of
                      symmetric part at each (layer, step).
        grade2_norms: (n_transitions, n_steps) Frobenius norm of
                      skew-symmetric part.
        flow_ratio: (n_transitions, n_steps) grade-2 / grade-0 ratio.
        n_transitions: number of layer transitions.
        n_steps: number of decode steps.
    """
    grade0_norms: np.ndarray
    grade2_norms: np.ndarray
    flow_ratio: np.ndarray
    n_transitions: int
    n_steps: int


def frontier_grade_profile(gen_result: GenerationResult,
                           skip_first: bool = True) -> FrontierGradeProfile:
    """
    Compute grade-0 / grade-2 decomposition at each frontier transition.

    At each layer transition (l -> l+1) and decode step s, forms the
    outer-product transition matrix M = h_{l+1} h_l^T / ||h_l||^2,
    then splits into symmetric (grade-0) and skew (grade-2) parts.

    Uses the whitened frontier vectors for gauge consistency.

    Args:
        gen_result: GenerationResult with frontier_whitened set.
        skip_first: Skip the first layer transition.

    Returns:
        FrontierGradeProfile with grade norms and flow ratios.
    """
    if gen_result.frontier_whitened is None:
        raise ValueError("Call whiten_frontier() first.")

    F = gen_result.frontier_whitened  # (n_total, L, k)
    n_total, L, k = F.shape
    start = 1 if skip_first else 0
    n_trans = L - 1 - start

    g0 = np.zeros((n_trans, n_total))
    g2 = np.zeros((n_trans, n_total))
    ratio = np.zeros((n_trans, n_total))

    for s in range(n_total):
        for li in range(n_trans):
            h_l = F[s, start + li]
            h_l1 = F[s, start + li + 1]
            norm_sq = np.dot(h_l, h_l) + 1e-12
            M = np.outer(h_l1, h_l) / norm_sq
            S = 0.5 * (M + M.T)
            A = 0.5 * (M - M.T)
            g0[li, s] = np.linalg.norm(S, 'fro')
            g2[li, s] = np.linalg.norm(A, 'fro')
            ratio[li, s] = g2[li, s] / (g0[li, s] + 1e-12)

    return FrontierGradeProfile(
        grade0_norms=g0,
        grade2_norms=g2,
        flow_ratio=ratio,
        n_transitions=n_trans,
        n_steps=n_total,
    )


# ── Frontier capacity ──────────────────────────────────────────


@dataclass
class FrontierCapacity:
    """Capacity metrics at the decode frontier across all steps.

    Attributes:
        C_acc: (n_steps,) accumulated pairwise commutator norm at each step.
        delta_C: (n_steps-1,) incremental capacity change per step.
        erank: (n_steps,) effective rank of the commutator spectrum.
        layer_contrib: (n_transitions, n_steps) per-layer contribution to C_acc.
        n_transitions: number of layer transitions.
        n_steps: number of decode steps.
    """
    C_acc: np.ndarray
    delta_C: np.ndarray
    erank: np.ndarray
    layer_contrib: np.ndarray
    n_transitions: int
    n_steps: int


def frontier_capacity(frontier_ga: FrontierGA) -> FrontierCapacity:
    """
    Compute capacity metrics from frontier Cayley bivectors.

    At each decode step s, computes:
      C_acc(s) = sum_{i<j} ||[A_i(s), A_j(s)]||_F

    where A_i(s) is the Cayley bivector at layer transition i, step s.
    Also computes per-layer contributions and effective rank.

    Uses vectorised numpy (einsum) for the pairwise commutators.

    Args:
        frontier_ga: FrontierGA from compute_frontier_ga().

    Returns:
        FrontierCapacity with all capacity time series.
    """
    n_trans = frontier_ga.n_transitions
    n_total = frontier_ga.n_steps

    # Extract all bivector matrices: (n_trans, n_total, k, k)
    k = frontier_ga.bivectors[0][0].dim
    B = np.zeros((n_trans, n_total, k, k))
    for li in range(n_trans):
        for s in range(n_total):
            B[li, s] = frontier_ga.bivectors[li][s].matrix

    C_acc = np.zeros(n_total)
    layer_contrib = np.zeros((n_trans, n_total))

    for s in range(n_total):
        Bs = B[:, s, :, :]  # (n_trans, k, k)
        # Pairwise commutator norms without materializing full (n,n,k,k) tensor
        comm_norms = np.zeros((n_trans, n_trans))
        for i in range(n_trans):
            for j in range(i + 1, n_trans):
                comm = Bs[i] @ Bs[j] - Bs[j] @ Bs[i]
                cn = np.linalg.norm(comm, 'fro')
                comm_norms[i, j] = cn
                comm_norms[j, i] = cn

        # C_acc = sum of upper triangle
        triu_idx = np.triu_indices(n_trans, k=1)
        C_acc[s] = comm_norms[triu_idx].sum()

        # Per-layer contribution: sum of row (excluding diagonal)
        for li in range(n_trans):
            layer_contrib[li, s] = comm_norms[li].sum()

    # Incremental capacity
    delta_C = np.diff(C_acc)

    # Effective rank: entropy-based measure of how distributed the
    # layer contributions are at each step
    erank = np.zeros(n_total)
    for s in range(n_total):
        lc = layer_contrib[:, s]
        total = lc.sum() + 1e-12
        p = lc / total
        p = p[p > 1e-12]  # avoid log(0)
        entropy = -np.sum(p * np.log(p))
        erank[s] = np.exp(entropy)

    return FrontierCapacity(
        C_acc=C_acc,
        delta_C=delta_C,
        erank=erank,
        layer_contrib=layer_contrib,
        n_transitions=n_trans,
        n_steps=n_total,
    )


# ── Frontier principal planes ──────────────────────────────────


def frontier_principal_planes(frontier_ga: FrontierGA,
                              layers: list[int] = None,
                              n_planes: int = 1) -> dict:
    """
    Track principal plane evolution at selected layers.

    At each decode step, extracts the dominant rotation plane from
    the Cayley bivector at each selected layer, then computes
    step-to-step plane similarity (cosine between successive plane
    bivectors).

    Args:
        frontier_ga: FrontierGA from compute_frontier_ga().
        layers: Layer transition indices to track. If None, uses
                [0, n_trans//4, n_trans//2, 3*n_trans//4, n_trans-1].
        n_planes: Number of principal planes per bivector.

    Returns:
        dict with:
          'layers': list of tracked layer indices.
          'plane_angles': dict mapping layer -> (n_steps,) dominant
                          plane angle at each step.
          'plane_similarity': dict mapping layer -> (n_steps-1,)
                              cosine similarity between successive
                              dominant planes.
    """
    n_trans = frontier_ga.n_transitions
    n_total = frontier_ga.n_steps

    if layers is None:
        layers = [0, n_trans // 4, n_trans // 2,
                  3 * n_trans // 4, n_trans - 1]
        layers = sorted(set(layers))

    plane_angles = {}
    plane_similarity = {}

    for li in layers:
        angles_li = np.zeros(n_total)
        sim_li = np.zeros(n_total - 1)
        prev_plane = None

        for s in range(n_total):
            biv = frontier_ga.bivectors[li][s]
            planes = biv.principal_planes(n_planes=n_planes)
            if planes:
                angles_li[s] = planes[0]['angle']
                # Plane similarity: Frobenius inner product of bivector
                # matrices between successive steps
                if prev_plane is not None and s > 0:
                    curr_mat = biv.matrix
                    prev_mat = prev_plane.matrix
                    inner = np.sum(curr_mat * prev_mat)
                    n1 = np.linalg.norm(curr_mat, 'fro')
                    n2 = np.linalg.norm(prev_mat, 'fro')
                    denom = n1 * n2 + 1e-12
                    sim_li[s - 1] = abs(inner / denom)
                prev_plane = biv

        plane_angles[li] = angles_li
        plane_similarity[li] = sim_li

    return {
        'layers': layers,
        'plane_angles': plane_angles,
        'plane_similarity': plane_similarity,
    }


# ── Frontier quality scores ────────────────────────────────────


@dataclass
class FrontierQualityScores:
    """Geometric quality diagnostics for generation.

    Attributes:
        capacity_growth_rate: Slope of C_acc(s) over decode steps.
                              Sublinear = healthy, linear = degenerate.
        capacity_periodicity: Max autocorrelation of delta_C for
                              lag in [2, 20]. High = periodic/repetition.
        erank_trend: Slope of effective rank over steps.
                     Negative = collapsing, positive = diversifying.
        plane_diversity: Mean plane drift (from FrontierGA).
                         High = diverse generation.
        curvature_acceleration: Second derivative of cumulative curvature.
                                 Zero = steady state (repetition).
    """
    capacity_growth_rate: float
    capacity_periodicity: float
    erank_trend: float
    plane_diversity: float
    curvature_acceleration: float


def frontier_quality_scores(frontier_ga: FrontierGA,
                            frontier_cap: FrontierCapacity,
                            holonomy: np.ndarray = None) -> FrontierQualityScores:
    """
    Compute geometric quality scores for a generation.

    Synthesizes frontier GA quantities and capacity into a small set
    of diagnostic scores.

    Args:
        frontier_ga: FrontierGA from compute_frontier_ga().
        frontier_cap: FrontierCapacity from frontier_capacity().
        holonomy: (n_layers, n_steps) scalar curvature. Optional.

    Returns:
        FrontierQualityScores.
    """
    n = len(frontier_cap.C_acc)

    # 1. Capacity growth rate: linear regression slope of C_acc
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    C_mean = frontier_cap.C_acc.mean()
    slope = np.sum((x - x_mean) * (frontier_cap.C_acc - C_mean)) / \
            (np.sum((x - x_mean) ** 2) + 1e-12)

    # 2. Capacity periodicity: max autocorrelation of delta_C
    dC = frontier_cap.delta_C
    if len(dC) > 4:
        dC_centered = dC - dC.mean()
        var = np.sum(dC_centered ** 2) + 1e-12
        max_lag = min(20, len(dC) // 2)
        max_acorr = 0.0
        for lag in range(2, max_lag + 1):
            acorr = np.sum(dC_centered[:-lag] * dC_centered[lag:]) / var
            if acorr > max_acorr:
                max_acorr = acorr
        periodicity = max_acorr
    else:
        periodicity = 0.0

    # 3. Effective rank trend: slope of erank over steps
    er = frontier_cap.erank
    er_mean = er.mean()
    erank_slope = np.sum((x - x_mean) * (er - er_mean)) / \
                  (np.sum((x - x_mean) ** 2) + 1e-12)

    # 4. Plane diversity: mean plane drift from FrontierGA
    plane_div = float(frontier_ga.plane_drift.mean())

    # 5. Curvature acceleration: mean second derivative of cumulative curvature
    if holonomy is not None:
        cum_curv = np.cumsum(holonomy.mean(axis=0))
        if len(cum_curv) >= 3:
            second_deriv = np.diff(cum_curv, n=2)
            curv_accel = float(np.mean(np.abs(second_deriv)))
        else:
            curv_accel = 0.0
    else:
        curv_accel = 0.0

    return FrontierQualityScores(
        capacity_growth_rate=float(slope),
        capacity_periodicity=float(periodicity),
        erank_trend=float(erank_slope),
        plane_diversity=plane_div,
        curvature_acceleration=curv_accel,
    )


# ── Online quality scores (Ch17) ──────────────────────────────


@dataclass
class OnlineScoreTrace:
    """Time series of quality scores computed incrementally.

    Attributes:
        periodicity: (n_steps,) online capacity periodicity at each step.
        growth_rate: (n_steps,) online capacity growth rate.
        erank: (n_steps,) effective rank at each step.
        plane_diversity: (n_steps,) running-mean plane drift.
        detection_step: int or None — first step where periodicity > threshold.
    """
    periodicity: np.ndarray
    growth_rate: np.ndarray
    erank: np.ndarray
    plane_diversity: np.ndarray
    detection_step: Optional[int]


def online_quality_scores(frontier_ga: FrontierGA,
                          frontier_cap: FrontierCapacity,
                          threshold: float = 0.4,
                          max_period: int = 20,
                          min_window: int = 10) -> OnlineScoreTrace:
    """
    Compute quality scores incrementally at each decode step.

    At each step s, uses only data from steps 0..s to compute the
    scores, simulating what would be available during live generation.

    Args:
        frontier_ga: FrontierGA from compute_frontier_ga().
        frontier_cap: FrontierCapacity from frontier_capacity().
        threshold: Periodicity threshold for repetition detection.
        max_period: Maximum lag for autocorrelation.
        min_window: Minimum steps before computing periodicity.

    Returns:
        OnlineScoreTrace with per-step score trajectories.
    """
    n = frontier_cap.n_steps
    dC = frontier_cap.delta_C  # length n-1

    periodicity = np.zeros(n)
    growth_rate = np.zeros(n)
    erank_trace = frontier_cap.erank.copy()
    drift = frontier_ga.plane_drift  # (n_trans, n-1)
    plane_div = np.zeros(n)
    detection_step = None

    for s in range(n):
        # Online capacity growth rate: slope of C_acc[0..s]
        if s >= 2:
            x = np.arange(s + 1, dtype=float)
            C = frontier_cap.C_acc[:s + 1]
            x_m = x.mean()
            C_m = C.mean()
            growth_rate[s] = np.sum((x - x_m) * (C - C_m)) / \
                             (np.sum((x - x_m) ** 2) + 1e-12)

        # Online periodicity: autocorrelation of delta_C[0..s-1]
        if s >= min_window and s > 1:
            dC_window = dC[:s]
            dC_c = dC_window - dC_window.mean()
            var = np.sum(dC_c ** 2) + 1e-12
            max_lag = min(max_period, len(dC_c) // 2)
            best = 0.0
            for lag in range(2, max_lag + 1):
                acorr = np.sum(dC_c[:-lag] * dC_c[lag:]) / var
                if acorr > best:
                    best = acorr
            periodicity[s] = best

            if detection_step is None and best > threshold:
                detection_step = s

        # Online plane diversity: running mean of drift up to step s
        if s >= 1 and s - 1 < drift.shape[1]:
            plane_div[s] = drift[:, :s].mean()

    return OnlineScoreTrace(
        periodicity=periodicity,
        growth_rate=growth_rate,
        erank=erank_trace,
        plane_diversity=plane_div,
        detection_step=detection_step,
    )


# ── Frontier Binet-Cauchy cosine (Ch17) ──────────────────────


def frontier_bccos(gen_result: GenerationResult,
                   skip_first: bool = True) -> np.ndarray:
    """
    Compute Binet-Cauchy cosine at the frontier across decode steps.

    At each layer transition l and decode step s >= 1, measures the
    signed bivector alignment between:
      - The frontier trajectory (step s-1 -> step s)
      - The prompt reference direction (second-to-last -> last prompt token)

    BCcos = +1 means the frontier moves in the same oriented plane as
    the prompt's final transition; -1 means causal inversion.

    Args:
        gen_result: GenerationResult with frontier_whitened set.
        skip_first: Skip the first layer transition.

    Returns:
        (n_transitions, n_steps-1) array of BCcos values.
    """
    if gen_result.frontier_whitened is None:
        raise ValueError("Call whiten_frontier() first.")

    F = gen_result.frontier_whitened  # (n_total, L, k)
    n_total, L, k = F.shape
    start = 1 if skip_first else 0
    n_trans = L - 1 - start

    # Prompt reference: last two prompt tokens (whitened, skip layer 0)
    prompt_H = gen_result.prompt_hidden_states[1:]  # (L, T_0, p)
    T_0 = prompt_H.shape[1]
    if T_0 < 2:
        return np.zeros((n_trans, max(n_total - 1, 1)))

    pre = backend.whiten(prompt_H[:, -2, :], gen_result.metric)  # (L, k)
    post = backend.whiten(prompt_H[:, -1, :], gen_result.metric)  # (L, k)

    bccos = np.zeros((n_trans, n_total - 1))

    for s in range(1, n_total):
        for li in range(n_trans):
            l_idx = start + li
            # Frontier trajectory vectors (normalised)
            u_prev = F[s - 1, l_idx]
            u_curr = F[s, l_idx]
            n1 = np.linalg.norm(u_prev) + 1e-12
            n2 = np.linalg.norm(u_curr) + 1e-12

            # Reference direction vectors (normalised)
            c_pre = pre[l_idx]
            c_post = post[l_idx]
            n3 = np.linalg.norm(c_pre) + 1e-12
            n4 = np.linalg.norm(c_post) + 1e-12

            bccos[li, s - 1] = binet_cauchy_cosine(
                u_prev / n1, u_curr / n2,
                c_pre / n3, c_post / n4,
            )

    return bccos


# ── Online repetition detection (Ch17) ─────────────────────────


@dataclass
class RepetitionDetection:
    """Result of online repetition detection.

    Attributes:
        detected: bool — whether repetition was detected.
        detection_step: int or None — first step where detection fired.
        detected_period: int or None — estimated period at detection.
        latency_tokens: int or None — tokens after first repetition before
                        detection (requires ground-truth onset step).
        online_periodicity: (n_steps,) full periodicity trace.
    """
    detected: bool
    detection_step: Optional[int]
    detected_period: Optional[int]
    latency_tokens: Optional[int]
    online_periodicity: np.ndarray


def detect_repetition_online(frontier_cap: FrontierCapacity,
                             threshold: float = 0.4,
                             max_period: int = 20,
                             min_window: int = 10,
                             ground_truth_onset: Optional[int] = None,
                             ) -> RepetitionDetection:
    """
    Detect repetition collapse from frontier capacity periodicity.

    At each step s, computes the autocorrelation of delta_C[0..s-1]
    for lags in [2, max_period]. If the maximum autocorrelation
    exceeds the threshold, flags the generation as repetitive.

    Args:
        frontier_cap: FrontierCapacity from frontier_capacity().
        threshold: Autocorrelation threshold for detection.
        max_period: Maximum lag to check.
        min_window: Minimum steps before checking.
        ground_truth_onset: If known, the step where repetition
                            actually starts (for latency measurement).

    Returns:
        RepetitionDetection with detection results.
    """
    dC = frontier_cap.delta_C
    n = len(dC)
    periodicity = np.zeros(n)
    detection_step = None
    detected_period = None

    for s in range(min_window, n):
        dC_window = dC[:s + 1]
        dC_c = dC_window - dC_window.mean()
        var = np.sum(dC_c ** 2) + 1e-12
        max_lag = min(max_period, len(dC_c) // 2)
        best = 0.0
        best_lag = 0
        for lag in range(2, max_lag + 1):
            acorr = np.sum(dC_c[:-lag] * dC_c[lag:]) / var
            if acorr > best:
                best = acorr
                best_lag = lag
        periodicity[s] = best

        if detection_step is None and best > threshold:
            detection_step = s
            detected_period = best_lag

    detected = detection_step is not None
    latency = None
    if detected and ground_truth_onset is not None:
        latency = max(0, detection_step - ground_truth_onset)

    return RepetitionDetection(
        detected=detected,
        detection_step=detection_step,
        detected_period=detected_period,
        latency_tokens=latency,
        online_periodicity=periodicity,
    )


# ── String-matching repetition detection (baseline) ────────────


def detect_repetition_string(tokens: list[str],
                             min_period: int = 2,
                             max_period: int = 20,
                             min_repeats: int = 2) -> Optional[int]:
    """
    Detect repetition via string matching (baseline for comparison).

    Checks if any substring of length P repeats at least min_repeats
    times consecutively in the token sequence.

    Args:
        tokens: List of token strings.
        min_period: Minimum period length.
        max_period: Maximum period length.
        min_repeats: Minimum number of consecutive repeats.

    Returns:
        Step where repetition is first detected, or None.
    """
    n = len(tokens)
    for P in range(min_period, min(max_period + 1, n // min_repeats + 1)):
        for start in range(n - P * min_repeats + 1):
            pattern = tokens[start:start + P]
            matches = 0
            for r in range(1, min_repeats + 1):
                chunk_start = start + r * P
                chunk_end = chunk_start + P
                if chunk_end > n:
                    break
                if tokens[chunk_start:chunk_end] == pattern:
                    matches += 1
                else:
                    break
            if matches >= min_repeats:
                # Detection fires at the end of the repeated block
                return start + P * min_repeats
    return None


# ── Frontier steering targets (Ch17) ──────────────────────────


def frontier_steering_target(frontier_ga: FrontierGA,
                             frontier_cap: FrontierCapacity,
                             step: int) -> dict:
    """
    Identify the optimal layer and plane for steering at a given step.

    Uses the per-layer capacity contribution to find the layer with
    the highest non-commutativity, then extracts its principal
    rotation plane from the Cayley bivector.

    Args:
        frontier_ga: FrontierGA from compute_frontier_ga().
        frontier_cap: FrontierCapacity from frontier_capacity().
        step: Decode step index.

    Returns:
        dict with:
          'layer': optimal layer transition index.
          'contribution': that layer's capacity contribution.
          'bivector': the Cayley bivector at that layer/step.
          'principal_plane': dominant plane info (from principal_planes).
          'angle': frontier rotation angle at that layer/step.
    """
    # Layer with highest capacity contribution at this step
    lc = frontier_cap.layer_contrib[:, step]
    best_layer = int(np.argmax(lc))

    biv = frontier_ga.bivectors[best_layer][step]
    planes = biv.principal_planes(n_planes=1)
    angle = float(frontier_ga.angles[best_layer, step])

    return {
        'layer': best_layer,
        'contribution': float(lc[best_layer]),
        'bivector': biv,
        'principal_plane': planes[0] if planes else None,
        'angle': angle,
    }
