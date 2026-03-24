"""Visualization functions for layer-time geometric analysis."""

import numpy as np
from typing import Optional

from layer_time_geometry import SampleGeometry, DirectionalRadial, SteeringDiagnostics
from layer_time.capacity import CapacityProfile


def _get_ax(ax):
    """Create a new axes if none provided."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def plot_curvature_heatmap(
    Omega_norms: np.ndarray,
    tokens: Optional[list[str]] = None,
    ax=None,
    cmap: str = "inferno",
    title: str = "Curvature Map",
    **kwargs,
):
    """Curvature heatmap with optional token labels on x-axis.

    Args:
        Omega_norms: (L-1, T-1) curvature norms.
        tokens: Token strings for x-axis labels.
        ax: Matplotlib axes (created if None).
        cmap: Colormap name.
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    im = ax.imshow(Omega_norms, aspect="auto", cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer transition")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="||Ω||_F")

    if tokens is not None and len(tokens) > 1:
        # Labels for midpoints between tokens (T-1 positions)
        mid_labels = [f"{tokens[t]}→{tokens[t+1]}" for t in range(min(len(tokens)-1, Omega_norms.shape[1]))]
        if len(mid_labels) <= 20:
            ax.set_xticks(range(len(mid_labels)))
            ax.set_xticklabels(mid_labels, rotation=45, ha="right", fontsize=7)

    return ax


def plot_operator_profile(
    sg: SampleGeometry,
    ax=None,
    title: str = "Operator Profile",
    **kwargs,
):
    """Dual-axis plot: rotation and scaling deviations vs layer index.

    Args:
        sg: SampleGeometry with rotation_devs and scaling_devs.
        ax: Matplotlib axes (created if None).
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    layers = np.arange(len(sg.rotation_devs))

    ax.plot(layers, np.nan_to_num(sg.rotation_devs), "o-", label="||U - I||", color="tab:blue")
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Rotation deviation", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(layers, np.nan_to_num(sg.scaling_devs), "s-", label="||P - I||", color="tab:red")
    ax2.set_ylabel("Scaling deviation", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax.set_title(title)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    return ax


def plot_energy_landscape(
    dr: DirectionalRadial,
    tokens: Optional[list[str]] = None,
    ax=None,
    cmap: str = "viridis",
    title: str = "Energy Landscape (log-radii)",
    **kwargs,
):
    """Heatmap of log-radii u(l,t).

    Args:
        dr: DirectionalRadial decomposition.
        tokens: Token strings for x-axis labels.
        ax: Matplotlib axes (created if None).
        cmap: Colormap name.
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    im = ax.imshow(dr.u, aspect="auto", cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="log(r)")

    if tokens is not None and len(tokens) <= 20:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)

    return ax


def plot_kernel_matrix(
    K: np.ndarray,
    labels: Optional[list[str]] = None,
    ax=None,
    cmap: str = "RdBu_r",
    title: str = "Prompt Kernel",
    **kwargs,
):
    """Kernel matrix as labeled heatmap.

    Args:
        K: (n, n) kernel or similarity matrix.
        labels: Labels for rows/columns.
        ax: Matplotlib axes (created if None).
        cmap: Colormap name.
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    im = ax.imshow(K, cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax)

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    return ax


def plot_stretching_field(
    S_stretch: np.ndarray,
    tokens: Optional[list[str]] = None,
    ax=None,
    cmap: str = "magma",
    title: str = "Stretching Field",
    **kwargs,
):
    """Stretching field heatmap.

    Args:
        S_stretch: (L-1, T) stretching values.
        tokens: Token strings for x-axis labels.
        ax: Matplotlib axes (created if None).

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    im = ax.imshow(S_stretch, aspect="auto", cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer transition")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="|log(r_{l+1}/r_l)|")

    if tokens is not None and len(tokens) <= 20:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)

    return ax


def plot_steering_diagnostics(
    diag: SteeringDiagnostics,
    tokens: Optional[list[str]] = None,
    axes=None,
    **kwargs,
):
    """Multi-panel steering diagnostic: angular ratio, curvature change, S+A.

    Args:
        diag: SteeringDiagnostics from the backend.
        tokens: Token strings for labeling.
        axes: Array of 3 matplotlib axes (created if None).

    Returns:
        Array of axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Angular ratio
    im0 = axes[0].imshow(diag.angular_ratio, aspect="auto", cmap="coolwarm",
                          origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Angular fraction")
    axes[0].set_xlabel("Token")
    axes[0].set_ylabel("Layer")
    axes[0].figure.colorbar(im0, ax=axes[0])

    # Panel 2: Curvature change
    vmax = max(abs(diag.delta_Omega_norms.min()), abs(diag.delta_Omega_norms.max()), 1e-8)
    im1 = axes[1].imshow(diag.delta_Omega_norms, aspect="auto", cmap="RdBu_r",
                          origin="lower", vmin=-vmax, vmax=vmax)
    axes[1].set_title("ΔCurvature")
    axes[1].set_xlabel("Token")
    axes[1].set_ylabel("Layer transition")
    axes[1].figure.colorbar(im1, ax=axes[1])

    # Panel 3: S+A summary bar
    labels = ["ΔS", "ΔA", "R_before", "R_after"]
    values = [diag.delta_S_norm, diag.delta_A_norm, diag.R_before, diag.R_after]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    axes[2].bar(labels, values, color=colors)
    axes[2].set_title("S+A Decomposition")
    axes[2].axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    return axes


def plot_curvature_profile(
    Omega_norms: np.ndarray,
    ax=None,
    title: str = "Mean Curvature per Layer",
    **kwargs,
):
    """Line plot of mean curvature per layer.

    Args:
        Omega_norms: (L-1, T-1) curvature norms.
        ax: Matplotlib axes (created if None).
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    profile = Omega_norms.mean(axis=1)
    ax.plot(range(len(profile)), profile, "o-", **kwargs)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Mean ||Ω||_F")
    ax.set_title(title)
    return ax


def plot_directionality_profile(
    sg: SampleGeometry,
    ax=None,
    title: str = "Directionality Ratio per Layer",
    **kwargs,
):
    """Per-layer directionality ratio ||A||/||S||.

    Args:
        sg: SampleGeometry.
        ax: Matplotlib axes (created if None).
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    R = np.nan_to_num(sg.A_norms) / (np.nan_to_num(sg.S_norms) + 1e-8)
    ax.plot(range(len(R)), R, "o-", color="tab:purple", **kwargs)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("R = ||A|| / ||S||")
    ax.set_title(title)
    return ax


# ── Generation-Time Plots ──────────────────────────────────────


def plot_generation_trajectory(gen_result, axes=None, **kwargs):
    """Multi-panel generation trajectory: difficulty, directionality, drift, Lyapunov.

    Args:
        gen_result: GenerationResult from analyzer.generate_and_track().
        axes: Array of 4 matplotlib axes (created if None).

    Returns:
        Array of axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

    traj = gen_result.trajectory
    steps = np.arange(gen_result.n_steps)

    # Build x-axis labels showing generated tokens
    xlabels = []
    for i in range(gen_result.n_steps):
        if i == 0:
            xlabels.append("prompt")
        else:
            tok = gen_result.token_strings[gen_result.prompt_length + i - 1]
            xlabels.append(f"+{tok.strip()}")

    # Panel 1: Difficulty (mean curvature)
    axes[0].plot(steps, traj.difficulties, "o-", color="tab:red", **kwargs)
    axes[0].set_ylabel("Mean curvature")
    axes[0].set_title("Difficulty over generation")
    if len(xlabels) <= 25:
        axes[0].set_xticks(steps)
        axes[0].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)

    # Panel 2: Directionality
    axes[1].plot(steps, traj.directionalities, "s-", color="tab:purple", **kwargs)
    axes[1].set_ylabel("R = ||A|| / ||S||")
    axes[1].set_title("Directionality over generation")
    if len(xlabels) <= 25:
        axes[1].set_xticks(steps)
        axes[1].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)

    # Panel 3: Representational drift
    if len(traj.last_token_drift) > 0:
        drift_steps = np.arange(1, gen_result.n_steps)
        axes[2].plot(drift_steps, traj.last_token_drift, "^-", color="tab:orange", **kwargs)
        axes[2].set_ylabel("Cosine distance")
        axes[2].set_title("Last-token drift (step to step)")
        if len(xlabels) <= 25:
            axes[2].set_xticks(drift_steps)
            axes[2].set_xticklabels(xlabels[1:], rotation=45, ha="right", fontsize=7)
    else:
        axes[2].set_visible(False)

    # Panel 4: Lyapunov exponent
    axes[3].plot(steps, traj.lyapunov_exponents, "d-", color="tab:green", **kwargs)
    axes[3].set_ylabel("λ_max")
    axes[3].set_title("Lyapunov exponent over generation")
    axes[3].axhline(0, color="k", linewidth=0.5, linestyle="--")
    if len(xlabels) <= 25:
        axes[3].set_xticks(steps)
        axes[3].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    return axes


def plot_frontier_curvature(gen_result, ax=None, cmap="inferno", **kwargs):
    """Heatmap of curvature at the generation frontier across steps.

    Shows how the layer-wise curvature profile at the newest token
    evolves as generation proceeds. X-axis = generation step,
    Y-axis = layer transition.

    Args:
        gen_result: GenerationResult from analyzer.generate_and_track().
        ax: Matplotlib axes (created if None).
        cmap: Colormap name.

    Returns:
        The matplotlib axes.
    """
    from layer_time_geometry import generation_curvature_evolution

    ax = _get_ax(ax)
    curv = gen_result.frontier_curvature()  # (n_steps, L-1)
    im = ax.imshow(curv.T, aspect="auto", cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer transition")
    ax.set_title("Frontier Curvature Evolution")
    ax.figure.colorbar(im, ax=ax, label="||Ω||_F at last token")

    # Label x-axis with generated tokens
    n_steps = gen_result.n_steps
    if n_steps <= 25:
        xlabels = []
        for i in range(n_steps):
            if i == 0:
                xlabels.append("prompt")
            else:
                tok = gen_result.token_strings[gen_result.prompt_length + i - 1]
                xlabels.append(f"+{tok.strip()}")
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)

    return ax


def plot_attention_shift(gen_result, ax=None, cmap="viridis", **kwargs):
    """Heatmap of temporal kernel effective rank across generation steps.

    Shows how the model distributes attention across token positions
    as generation proceeds.

    Args:
        gen_result: GenerationResult from analyzer.generate_and_track().
        ax: Matplotlib axes (created if None).
        cmap: Colormap name.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    eranks = gen_result.attention_shift()  # (n_steps, L)
    im = ax.imshow(eranks.T, aspect="auto", cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Spread (temporal erank)")
    ax.figure.colorbar(im, ax=ax, label="Effective rank")
    return ax


# ── Capacity / Scaling Plots ─────────────────────────────────────


def plot_commutator_heatmap(
    comm_norms: np.ndarray,
    ax=None,
    cmap: str = "inferno",
    title: str = "Layer Commutator Structure",
    **kwargs,
):
    """Heatmap of ||[A^(i), A^(j)]||_F across layer pairs.

    Args:
        comm_norms: (n, n) commutator norm matrix from capacity analysis.
        ax: Matplotlib axes (created if None).
        cmap: Colormap name.
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    im = ax.imshow(comm_norms, aspect="equal", cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("Layer transition j")
    ax.set_ylabel("Layer transition i")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="||[A(i), A(j)]||_F")
    return ax


def plot_capacity_comparison(
    capacities: dict[str, list[CapacityProfile]],
    metric: str = "C_eff",
    ax=None,
    title: Optional[str] = None,
    **kwargs,
):
    """Boxplot of a capacity metric across models.

    Args:
        capacities: Dict mapping model name to list of CapacityProfile.
        metric: Which metric to plot ("C_acc", "C_eff", "cconc_acc").
        ax: Matplotlib axes (created if None).
        title: Plot title (defaults to metric name).

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    names = list(capacities.keys())
    data = []
    for name in names:
        values = [getattr(cp, metric) for cp in capacities[name]]
        data.append(values)

    bp = ax.boxplot(data, labels=names, patch_artist=True, **kwargs)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for patch, color in zip(bp["boxes"], colors[: len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} across models")
    return ax


def plot_correctness_separation(
    C_eff_correct: np.ndarray,
    C_eff_incorrect: np.ndarray,
    ax=None,
    title: str = "Correctness Separation",
    **kwargs,
):
    """Overlapping histograms of C_eff for correct vs incorrect answers.

    Args:
        C_eff_correct: Array of C_eff values for correct answers.
        C_eff_incorrect: Array of C_eff values for incorrect answers.
        ax: Matplotlib axes (created if None).
        title: Plot title.

    Returns:
        The matplotlib axes.
    """
    ax = _get_ax(ax)
    bins = kwargs.pop("bins", 20)

    ax.hist(C_eff_correct, bins=bins, alpha=0.6, label="correct",
            color="tab:green", density=True)
    ax.hist(C_eff_incorrect, bins=bins, alpha=0.6, label="incorrect",
            color="tab:red", density=True)

    delta = np.mean(C_eff_correct) - np.mean(C_eff_incorrect)
    ax.axvline(np.mean(C_eff_correct), color="tab:green", linestyle="--", linewidth=1)
    ax.axvline(np.mean(C_eff_incorrect), color="tab:red", linestyle="--", linewidth=1)

    ax.set_xlabel("C_eff")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}  (ΔC_eff = {delta:.4f})")
    ax.legend()
    return ax


def plot_scaling_summary(
    capacities: dict[str, list[CapacityProfile]],
    labels: Optional[list[bool]] = None,
    axes=None,
    **kwargs,
):
    """Multi-panel scaling summary: C_acc, C_eff, concentration, commutator heatmaps.

    Args:
        capacities: Dict mapping model name to list of CapacityProfile.
        labels: Optional bool list (correct/incorrect) for separation plot.
        axes: Array of matplotlib axes (created if None).

    Returns:
        Array of axes.
    """
    import matplotlib.pyplot as plt

    n_models = len(capacities)
    n_panels = 3 + n_models  # C_acc, C_eff, cconc + one heatmap per model

    if axes is None:
        ncols = min(n_panels, 4)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

    # Panel 1: C_acc boxplot
    plot_capacity_comparison(capacities, metric="C_acc", ax=axes[0])

    # Panel 2: C_eff boxplot
    plot_capacity_comparison(capacities, metric="C_eff", ax=axes[1])

    # Panel 3: cconc_acc boxplot
    plot_capacity_comparison(capacities, metric="cconc_acc", ax=axes[2])

    # Panels 4+: commutator heatmaps (averaged over prompts)
    for idx, (name, caps) in enumerate(capacities.items()):
        if idx + 3 < len(axes):
            mean_comm = np.mean([cp.commutator_norms for cp in caps], axis=0)
            plot_commutator_heatmap(mean_comm, ax=axes[idx + 3], title=f"Commutator: {name}")

    # Hide unused axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return axes
