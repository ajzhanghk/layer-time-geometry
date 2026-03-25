"""
Generate all figures and captured outputs for the GA Learning book.
Runs key computations from the tutorial notebooks and saves results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures_ga_learning')
os.makedirs(FIGDIR, exist_ok=True)

OUT = []  # collect printed output lines

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved {name}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════
# Chapter 2: Geometric Product
# ══════════════════════════════════════════════════════════════
section("Chapter 2: Geometric Product")

from layer_time_ga.algebra import geometric_product_vectors, grade_decomposition, bivector_from_skew

# Perpendicular vectors
a = np.array([1.0, 0.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0, 0.0])
gp = geometric_product_vectors(a, b)
print(f"Perpendicular: scalar={gp['scalar']:.4f}, bivector_norm={gp['bivector'].norm:.4f}")

# Parallel vectors
c = np.array([3.0, 0.0, 0.0, 0.0])
gp2 = geometric_product_vectors(a, c)
print(f"Parallel:      scalar={gp2['scalar']:.4f}, bivector_norm={gp2['bivector'].norm:.6f}")

# Angle sweep: scalar vs bivector
angles_sweep = np.linspace(0, np.pi/2, 50)
scalars = []
biv_norms = []
for theta in angles_sweep:
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([np.cos(theta), np.sin(theta), 0.0, 0.0])
    gp_t = geometric_product_vectors(v1, v2)
    scalars.append(abs(gp_t['scalar']))
    biv_norms.append(gp_t['bivector'].norm)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.degrees(angles_sweep), scalars, color='#1D6A6A', linewidth=2, label='|Scalar| (grade-0)')
ax.plot(np.degrees(angles_sweep), biv_norms, color='#5B2C8B', linewidth=2, label='||Bivector|| (grade-2)')
ax.set_xlabel('Angle between vectors (degrees)', fontsize=11)
ax.set_ylabel('Magnitude', fontsize=11)
ax.set_title('Geometric Product: Scalar vs Bivector Content', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
savefig('ch02_scalar_vs_bivector.pdf')

# ══════════════════════════════════════════════════════════════
# Chapter 3: Bivectors
# ══════════════════════════════════════════════════════════════
section("Chapter 3: Bivectors and Principal Planes")

# Simple bivector
e1 = np.zeros(8); e1[0] = 1.0
e2 = np.zeros(8); e2[1] = 1.0
B_simple_mat = np.outer(e1, e2) - np.outer(e2, e1)
B_simple = bivector_from_skew(B_simple_mat)
planes_simple = B_simple.principal_planes(n_planes=3)
print(f"Simple bivector: {len([p for p in planes_simple if p['weight'] > 1e-10])} significant plane(s)")

# Compound bivector
e3 = np.zeros(8); e3[2] = 1.0
e4 = np.zeros(8); e4[3] = 1.0
B2_mat = 0.7 * (np.outer(e1, e2) - np.outer(e2, e1)) + 0.5 * (np.outer(e3, e4) - np.outer(e4, e3))
B_compound = bivector_from_skew(B2_mat)
planes_compound = B_compound.principal_planes(n_planes=4)
print(f"Compound bivector planes:")
for i, p in enumerate(planes_compound):
    if p['weight'] > 1e-10:
        print(f"  Plane {i}: angle={p['angle']:.4f}, weight={p['weight']:.4f}")

# Random bivector principal planes
rng = np.random.default_rng(42)
A_rand = rng.standard_normal((8, 8))
A_rand = 0.5 * (A_rand - A_rand.T)
B_rand = bivector_from_skew(A_rand)
planes_rand = B_rand.principal_planes(n_planes=4)

fig, ax = plt.subplots(figsize=(6, 4))
weights = [p['weight'] for p in planes_rand]
ax.bar(range(len(weights)), weights, color=['#1F3864', '#2E6DAD', '#5B9BD5', '#A3C4E0'])
ax.set_xlabel('Principal plane index', fontsize=11)
ax.set_ylabel('Weight', fontsize=11)
ax.set_title('Principal Plane Decomposition of a Random Bivector', fontsize=12)
ax.set_xticks(range(len(weights)))
ax.set_xticklabels([f'Plane {i}' for i in range(len(weights))])
savefig('ch03_principal_planes.pdf')

# ══════════════════════════════════════════════════════════════
# Chapter 5: Rotors
# ══════════════════════════════════════════════════════════════
section("Chapter 5: Rotors")

from layer_time_ga.algebra import rotor_from_orthogonal, rotor_compose, rotor_inverse

# Create a rotation
A_rot = 0.3 * rng.standard_normal((8, 8))
A_rot = 0.5 * (A_rot - A_rot.T)
U = expm(A_rot)
R = rotor_from_orthogonal(U)
print(f"Rotor: angle={R.angle:.4f} rad, bivector_norm={R.bivector.norm:.4f}")

# Norm preservation
v = rng.standard_normal(8)
v_rot = R.apply(v)
print(f"||v||={np.linalg.norm(v):.6f}, ||Rv||={np.linalg.norm(v_rot):.6f}")

# 45-degree rotation in e1-e2 plane
theta_45 = np.pi / 4
B_12 = np.zeros((8, 8))
B_12[0, 1] = theta_45
B_12[1, 0] = -theta_45
U_45 = expm(B_12)
R_45 = rotor_from_orthogonal(U_45)
e1_rot = R_45.apply(e1)
print(f"45° rotation of e1: [{e1_rot[0]:.4f}, {e1_rot[1]:.4f}, ...]")
print(f"Expected:           [{np.cos(theta_45):.4f}, {np.sin(theta_45):.4f}, ...]")

# Composition
A2 = 0.15 * rng.standard_normal((8, 8))
A2 = 0.5 * (A2 - A2.T)
R2 = rotor_from_orthogonal(expm(A2))
R12 = rotor_compose(R, R2)
R_inv = rotor_inverse(R)
R_id = rotor_compose(R, R_inv)
print(f"R angle={R.angle:.4f}, R2 angle={R2.angle:.4f}, R12 angle={R12.angle:.4f}")
print(f"R*R^-1 deviation from identity: {R_id.deviation_from_identity():.2e}")

# ══════════════════════════════════════════════════════════════
# Chapter 9: Commutators
# ══════════════════════════════════════════════════════════════
section("Chapter 9: Commutators")

from layer_time_ga.algebra import commutator_bivector

B1 = R.bivector
B2 = R2.bivector
comm = commutator_bivector(B1, B2)
comm_rev = commutator_bivector(B2, B1)
comm_self = commutator_bivector(B1, B1)

print(f"||[B1, B2]||_F = {comm.norm:.6f}")
print(f"Antisymmetry: [B1,B2] = -[B2,B1]? {np.allclose(comm.matrix, -comm_rev.matrix)}")
print(f"Self-commutator norm: {comm_self.norm:.2e}")

# Jacobi identity
A3 = 0.2 * rng.standard_normal((8, 8))
A3 = 0.5 * (A3 - A3.T)
B3 = bivector_from_skew(A3)

j1 = commutator_bivector(B1, commutator_bivector(B2, B3))
j2 = commutator_bivector(B2, commutator_bivector(B3, B1))
j3 = commutator_bivector(B3, commutator_bivector(B1, B2))
jacobi_norm = np.linalg.norm(j1.matrix + j2.matrix + j3.matrix)
print(f"Jacobi identity residual: {jacobi_norm:.2e}")

comm_planes = comm.principal_planes(n_planes=3)
print(f"Commutator principal planes:")
for i, p in enumerate(comm_planes):
    if p['weight'] > 1e-10:
        print(f"  Plane {i}: angle={p['angle']:.4f}, weight={p['weight']:.4f}")


# ══════════════════════════════════════════════════════════════
# NOW: Load model and run transformer-dependent figures
# ══════════════════════════════════════════════════════════════
section("Loading model...")

import ltg_ga

model = ltg_ga.load_model("Qwen/Qwen2.5-7B")
print(f"Loaded {model.name}: {model.n_layers} layers, dim={model.hidden_dim}")

# ── Chapter 1: Cosine similarity heatmaps ─────────────────────
section("Chapter 1: Cosine Similarity Across Layers")

# Use a longer prompt so the token-token structure is visible
import layer_time_geometry as core
ch1_prompt = "The Eiffel Tower is a wrought iron lattice tower in Paris France"
H_raw_ch1 = core.extract_hidden_states(model.hf_model, model.tokenizer,
                                        ch1_prompt, model.device)
H_ch1 = H_raw_ch1[1:].cpu().numpy()  # raw (unwhitened) to show convergence
L_ch1, T_ch1, _ = H_ch1.shape
print(f"Ch1 grid: {L_ch1} layers x {T_ch1} tokens")

# Also run the standard analysis for later chapters
result = ltg_ga.analyse("The capital of France is", model=model)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, layer_idx, title in zip(axes, [1, L_ch1-2],
                                ['Early (Layer 2)', f'Late (Layer {L_ch1-1})']):
    H_l = H_ch1[layer_idx]
    norms = np.linalg.norm(H_l, axis=1, keepdims=True)
    H_norm = H_l / (norms + 1e-12)
    cos_sim = H_norm @ H_norm.T
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Token')
    ax.set_ylabel('Token')
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Cosine Similarity Between Tokens', fontsize=13, y=1.02)
plt.tight_layout()
savefig('ch01_cosine_layers.pdf')

# ── Chapter 4: Covariance before/after whitening ──────────────
section("Chapter 4: Whitening")
import layer_time_geometry as core

H_raw = core.extract_hidden_states(model.hf_model, model.tokenizer,
                                    "The capital of France is", model.device)
H_np = H_raw[1:].cpu().numpy()
L_r, T_r, p = H_np.shape
H_flat_raw = H_np.reshape(-1, p)

metric = core.estimate_metric(H_flat_raw, n_components=min(256, H_flat_raw.shape[0]-1))
H_w = core.whiten(H_np, metric)
k_w = H_w.shape[-1]
H_flat_w = H_w.reshape(-1, k_w)

# Correlation matrices (normalized covariance) — more informative than raw covariance
n_show = min(100, p, k_w)  # show first 100 dims
cov_raw = np.cov(H_flat_raw[:, :n_show].T)
std_raw = np.sqrt(np.diag(cov_raw)) + 1e-12
corr_raw = cov_raw / np.outer(std_raw, std_raw)

cov_white = np.cov(H_flat_w[:, :min(n_show, k_w)].T)
std_white = np.sqrt(np.diag(cov_white)) + 1e-12
corr_white = cov_white / np.outer(std_white, std_white)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(corr_raw, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
axes[0].set_title('Before Whitening (correlation)', fontsize=11)
axes[0].set_xlabel('Dimension')
axes[0].set_ylabel('Dimension')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(corr_white, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
axes[1].set_title('After Whitening (correlation)', fontsize=11)
axes[1].set_xlabel('Dimension')
axes[1].set_ylabel('Dimension')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

plt.tight_layout()
savefig('ch04_covariance.pdf')

# Summary statistics
off_mask = ~np.eye(corr_raw.shape[0], dtype=bool)
raw_offdiag = np.abs(corr_raw[off_mask]).mean()
white_offdiag = np.abs(corr_white[off_mask]).mean()
diag_mean = np.diag(cov_white).mean()
print(f"Raw mean |off-diagonal correlation|: {raw_offdiag:.4f}")
print(f"Whitened mean |off-diagonal correlation|: {white_offdiag:.6f}")
print(f"Whitened covariance diagonal mean: {diag_mean:.4f}")

# ── Chapter 6: Rotor field + grade profile ────────────────────
section("Chapter 6: Rotor Field and Grade Profile")

rf = result.rotor_field
angles = rf.angles
cond_numbers = rf.condition_numbers
eranks = rf.effective_ranks

# Rotor angle profile
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(angles)), angles, color='#2E6DAD', linewidth=2, marker='o', markersize=3)
ax.fill_between(range(len(angles)), angles, alpha=0.15, color='#2E6DAD')
ax.set_xlabel('Layer transition', fontsize=11)
ax.set_ylabel('Rotation angle (rad)', fontsize=11)
ax.set_title('Rotor Angle Profile', fontsize=12)
ax.axvline(angles.argmax(), color='grey', linestyle='--', alpha=0.5, label=f'Peak = layer {angles.argmax()}')
ax.legend()
ax.grid(True, alpha=0.3)
savefig('ch06_rotor_angles.pdf')

# Grade-0 vs grade-2
grade0 = []
grade2 = []
for vd in rf.decompositions:
    P = vd.metric
    grade0.append(np.linalg.norm(P - np.eye(P.shape[0]), 'fro'))
    grade2.append(vd.bivector.norm)

fig, ax = plt.subplots(figsize=(9, 4))
layers = np.arange(len(grade0))
w = 0.35
ax.bar(layers - w/2, grade0, w, label='Grade-0 $\\|P - I\\|_F$', color='#1D6A6A')
ax.bar(layers + w/2, grade2, w, label='Grade-2 $\\|B\\|_F$', color='#5B2C8B')
ax.set_xlabel('Layer transition', fontsize=11)
ax.set_ylabel('Frobenius norm', fontsize=11)
ax.set_title('Grade Profile: Grade-0 (Stretch) vs Grade-2 (Rotation)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
savefig('ch06_grade_profile.pdf')

print(f"Rotor field: {len(angles)} transitions")
print(f"Peak angle: layer {angles.argmax()} ({angles.max():.4f} rad)")
print(f"Mean grade-0: {np.mean(grade0):.4f}, mean grade-2: {np.mean(grade2):.4f}")

# ── Chapter 7: Plane evolution ────────────────────────────────
section("Chapter 7: Plane Evolution")

n_planes_show = 3
plane_weights_all = []
for vd in rf.decompositions:
    planes = vd.bivector.principal_planes(n_planes=n_planes_show)
    w_vec = [p['weight'] for p in planes]
    while len(w_vec) < n_planes_show:
        w_vec.append(0.0)
    plane_weights_all.append(w_vec)

pw = np.array(plane_weights_all)
fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#1F3864', '#2E6DAD', '#5B9BD5']
ax.stackplot(range(len(pw)), *[pw[:, j] for j in range(n_planes_show)],
             labels=[f'Plane {j+1}' for j in range(n_planes_show)],
             colors=colors, alpha=0.85)
ax.set_xlabel('Layer transition', fontsize=11)
ax.set_ylabel('Singular value weight', fontsize=11)
ax.set_title('Bivector Plane Evolution Across Layers', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
savefig('ch07_plane_evolution.pdf')

# Plane similarity heatmap
biv_mats = [vd.bivector.matrix for vd in rf.decompositions]
n = len(biv_mats)
sim_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        num = np.sum(biv_mats[i] * biv_mats[j])
        den = np.linalg.norm(biv_mats[i], 'fro') * np.linalg.norm(biv_mats[j], 'fro') + 1e-12
        sim_mat[i, j] = num / den

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sim_mat, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Layer', fontsize=11)
ax.set_title('Bivector Similarity Between Layers', fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine similarity')
savefig('ch07_plane_similarity.pdf')

# ── Chapter 8: Eigenvalue spectra ─────────────────────────────
section("Chapter 8: Eigenvalue Spectra")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, idx, label in zip(axes, [0, len(rf.decompositions)//2, -1],
                          ['Early', 'Middle', 'Late']):
    vd = rf.decompositions[idx]
    sv = np.sort(vd.singular_values)[::-1]
    ax.bar(range(min(50, len(sv))), sv[:50], width=1.0, color='#1D6A6A')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Identity')
    ax.set_title(f'{label} (layer {vd.layer_index}, $\\kappa$={vd.condition_number:.0f})', fontsize=11)
    ax.set_xlabel('Direction', fontsize=10)
    ax.set_ylabel('Eigenvalue', fontsize=10)
    ax.legend(fontsize=9)
plt.tight_layout()
savefig('ch08_eigenvalue_spectra.pdf')

# Condition number + effective rank
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(range(len(cond_numbers)), cond_numbers, color='#AA3377', linewidth=2, label='$\\kappa$')
ax1.set_xlabel('Layer transition', fontsize=11)
ax1.set_ylabel('Condition number $\\kappa$', color='#AA3377', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#AA3377')
ax2 = ax1.twinx()
ax2.plot(range(len(eranks)), eranks, color='#66CCEE', linewidth=2, linestyle='--', label='erank')
ax2.set_ylabel('Effective rank', color='#66CCEE', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#66CCEE')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.95), fontsize=10)
ax1.set_title('Metric Selectivity Across Layers', fontsize=12)
ax1.grid(True, alpha=0.3)
savefig('ch08_kappa_erank.pdf')

# ── Chapter 9: Commutator heatmap from transformer ───────────
section("Chapter 9: Commutator Heatmap")

from layer_time_ga.curvature import commutator_field, commutator_plane_decomposition

comm_norms = commutator_field(rf.bivectors)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(comm_norms, cmap='inferno', origin='lower')
ax.set_xlabel('Layer $j$', fontsize=11)
ax.set_ylabel('Layer $i$', fontsize=11)
ax.set_title('Commutator Norm $\\|[B_i, B_j]\\|_F$', fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8, label='Commutator norm')
savefig('ch09_commutator_heatmap.pdf')

# ── Chapter 10: Holonomy ──────────────────────────────────────
section("Chapter 10: Holonomy")

from layer_time_ga.curvature import holonomy_scalar_map, holonomy_rotor

# Use a longer prompt so the layer x token heatmap is informative
ch10_prompt = "The Eiffel Tower is a wrought iron lattice tower in Paris France"
result_ch10 = ltg_ga.analyse(ch10_prompt, model=model)
holo_map = holonomy_scalar_map(result_ch10.H_whitened)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(holo_map, aspect='auto', cmap='YlOrRd', origin='lower')
axes[0].set_xlabel('Token position', fontsize=11)
axes[0].set_ylabel('Layer', fontsize=11)
axes[0].set_title('Holonomy Scalar Curvature', fontsize=12)
plt.colorbar(im, ax=axes[0], shrink=0.8)

curv_by_layer = holo_map.mean(axis=1)
axes[1].plot(curv_by_layer, color='#EE6677', linewidth=2)
axes[1].fill_between(range(len(curv_by_layer)), curv_by_layer, alpha=0.2, color='#EE6677')
axes[1].axvline(curv_by_layer.argmax(), color='grey', linestyle='--', alpha=0.5,
                label=f'Peak = layer {curv_by_layer.argmax()}')
axes[1].set_xlabel('Layer', fontsize=11)
axes[1].set_ylabel('Mean curvature', fontsize=11)
axes[1].set_title('Curvature by Layer', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
savefig('ch10_holonomy.pdf')

print(f"Holonomy map shape: {holo_map.shape}")
print(f"Peak curvature: layer {curv_by_layer.argmax()} ({curv_by_layer.max():.6f})")

# ── Chapter 11: Capacity ──────────────────────────────────────
section("Chapter 11: Capacity")

from layer_time_ga.capacity import ga_capacity_profile
cap = ga_capacity_profile(result.H_whitened)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(cap.layer_contributions)), cap.layer_contributions, color='#1F3864')
ax.set_xlabel('Layer pair', fontsize=11)
ax.set_ylabel('$\\|[B_l, B_{l+1}]\\|_F$', fontsize=11)
ax.set_title('Per-Layer Capacity Contributions', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
savefig('ch11_capacity.pdf')

print(f"C_acc = {cap.C_acc:.3f}")
print(f"C_eff = {cap.C_eff:.3f}")
print(f"Concentration = {cap.cconc:.3f}")

# ── Chapter 12: Dependency + effective work ───────────────────
section("Chapter 12: Dependency")

if result.dependency_profile is not None:
    D = result.dependency_profile
    n_dep = min(len(D), len(angles))
    eff_work = D[:n_dep] * angles[:n_dep]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlay
    ax1 = axes[0]
    ax1.bar(range(n_dep), D[:n_dep], color='#2E6DAD', alpha=0.6, label='Dependency $D_l$')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Dependency', color='#2E6DAD', fontsize=11)
    ax1b = ax1.twinx()
    ax1b.plot(range(n_dep), angles[:n_dep], color='#EE6677', linewidth=2, label='Rotor angle')
    ax1b.set_ylabel('Angle (rad)', color='#EE6677', fontsize=11)
    ax1.set_title('Dependency and Rotation', fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # Right: effective work
    top3 = np.argsort(eff_work)[-3:][::-1]
    colors_ew = ['#EE6677' if i in top3 else '#1F3864' for i in range(n_dep)]
    axes[1].bar(range(n_dep), eff_work, color=colors_ew)
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('$D_l \\times \\theta_l$', fontsize=11)
    axes[1].set_title('Effective Work (top-3 highlighted)', fontsize=12)
    axes[1].grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    savefig('ch12_dependency.pdf')

    print(f"Dep total: {result.dep_total:.4f}, entropy: {result.dep_entropy:.3f}")
    print(f"Top-3 effective work layers: {top3}")
    for l in top3:
        vd = rf.decompositions[l]
        plane = vd.bivector.principal_planes(n_planes=1)[0]
        print(f"  Layer {l}: D={D[l]:.4f}, theta={angles[l]:.4f}, "
              f"kappa={vd.condition_number:.1f}, plane_angle={plane['angle']:.4f}")
else:
    print("No dependency data available")

# ── GA Summary plot ───────────────────────────────────────────
section("GA Summary Plot")
# Use the longer prompt for richer curvature map in the summary
result_ch10.plot_ga_summary(save_path=os.path.join(FIGDIR, 'ch13_ga_summary.pdf'))
print("  Saved ch13_ga_summary.pdf")

print(f"\n{'='*60}")
print(f"  All figures saved to {FIGDIR}/")
print(f"{'='*60}")
