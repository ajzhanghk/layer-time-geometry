"""
Section 16.3: Hallucination Signatures — Static + Frontier Analysis
====================================================================
Compares Factual and Confabulation prompts using both
static GA analysis and autoregressive frontier analysis.

Output:
  figures_ga_learning/ch16_3_static_profile.pdf
  figures_ga_learning/ch16_3_frontier_capacity.pdf
  figures_ga_learning/ch16_3_frontier_heatmaps.pdf
  figures_ga_learning/ch16_3_frontier_bccos.pdf
  figures_ga_learning/ch16_3_diagnostic_summary.pdf
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import time

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures_ga_learning')
os.makedirs(FIGDIR, exist_ok=True)

DATADIR = os.path.join(os.path.dirname(__file__), 'sample_data')
os.makedirs(DATADIR, exist_ok=True)

PROMPTS = {
    'Factual':       'The Eiffel Tower is a wrought iron lattice tower in Paris France',
    'Confabulation': 'The underwater city of Atlantis was discovered in 2019 near the coast of',
}
COLORS = {
    'Factual':       '#2E6DAD',
    'Confabulation': '#C62828',
}

N_STEPS = 50

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved {name}")


# ══════════════════════════════════════════════════════════════════
# PART 1: Static GA analysis
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: STATIC GA ANALYSIS")
print("="*70)

import ltg_ga
from layer_time_ga.capacity import ga_capacity_profile
import warnings
warnings.filterwarnings('ignore', message='logm result may be inaccurate')

model = ltg_ga.load_model("Qwen/Qwen2.5-7B")
print(f"Model: {model.name}, {model.n_layers} layers, dim={model.hidden_dim}")

static_pkl = os.path.join(DATADIR, 'ch16_3_static_results.pkl')

if os.path.exists(static_pkl):
    print("Loading cached static results...")
    with open(static_pkl, 'rb') as f:
        saved_static = pickle.load(f)
    static_results = saved_static['static_results']
    static_cap = saved_static['static_cap']
    print("Loaded.")
else:
    static_results = {}
    static_cap = {}

    for name, prompt in PROMPTS.items():
        print(f"\n--- {name} ---")
        print(f"  Running static GA analysis (with dependency)...")
        t0 = time.time()
        result = ltg_ga.analyse(prompt, model=model, compute_dependency=True,
                                whiten_components=256)
        result.summary()
        static_results[name] = result

        # Capacity profile
        cap = ga_capacity_profile(result.H_whitened,
                                  D_layer=result.dependency_profile)
        static_cap[name] = cap
        print(f"  C_acc = {cap.C_acc:.1f}")
        print(f"  Concentration = {cap.cconc:.3f}")
        print(f"  Time: {time.time()-t0:.1f}s")

    with open(static_pkl, 'wb') as f:
        pickle.dump({'static_results': static_results, 'static_cap': static_cap}, f)
    print("Cached static results.")


# Print LaTeX-ready static table
print("\n" + "="*70)
print("STATIC RESULTS TABLE")
print("="*70)
print(f"{'Metric':<25s}  {'Factual':>10s}  {'Confabulation':>14s}")
print("-" * 55)
for metric_name, getter in [
    ('Mean rotation angle', lambda r: r.angles.mean()),
    ('Max rotation angle',  lambda r: r.angles.max()),
    ('D_total',             lambda r: r.dep_total if r.dep_total else 0),
    ('H(D) (entropy)',      lambda r: r.dep_entropy if r.dep_entropy else 0),
    ('C_acc',               lambda r: static_cap[next(k for k,v in static_results.items() if v is r)].C_acc),
    ('Concentration',       lambda r: static_cap[next(k for k,v in static_results.items() if v is r)].cconc),
]:
    vals = [getter(static_results[n]) for n in PROMPTS]
    print(f"{metric_name:<25s}  {vals[0]:10.2f}  {vals[1]:14.2f}")


# ══════════════════════════════════════════════════════════════════
# PART 2: Frontier generation
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: FRONTIER GENERATION")
print("="*70)

from layer_time_ga.generation import (
    generate_with_frontier, whiten_frontier, compute_frontier_ga,
    frontier_holonomy, frontier_capacity, frontier_quality_scores,
    frontier_bccos, online_quality_scores, detect_repetition_online,
)

cache_pkl = os.path.join(DATADIR, 'ch16_3_hallucination_data.pkl')

if os.path.exists(cache_pkl):
    print("Loading cached frontier results...")
    with open(cache_pkl, 'rb') as f:
        saved = pickle.load(f)
    gen_results = saved['gen_results']
    ga_results = saved['ga_results']
    holo_results = saved['holo_results']
    print("Loaded.")
else:
    print("Running frontier generation (this takes ~10 min)...")
    gen_results = {}
    ga_results = {}
    holo_results = {}

    for name, prompt in PROMPTS.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        gen = generate_with_frontier(
            model.hf_model, model.tokenizer, prompt,
            n_steps=N_STEPS, device=model.device, temperature=0.0
        )
        print(f"  Generated: \"{gen.generated_text[:80]}...\"")

        gen = whiten_frontier(gen, whiten_components=256)
        ga = compute_frontier_ga(gen, skip_first=True)
        holo = frontier_holonomy(gen, skip_first=True, device=model.device)

        print(f"  Time: {time.time()-t0:.1f}s")

        gen_results[name] = gen
        ga_results[name] = ga
        holo_results[name] = holo

    # Cache
    with open(cache_pkl, 'wb') as f:
        pickle.dump({
            'gen_results': gen_results,
            'ga_results': ga_results,
            'holo_results': holo_results,
        }, f)
    print("Cached frontier results.")


# ══════════════════════════════════════════════════════════════════
# PART 3: Frontier metrics
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 3: FRONTIER METRICS")
print("="*70)

cap_results = {}
bccos_results = {}
quality_results = {}
online_results = {}

for name in PROMPTS:
    print(f"\n--- {name} ---")
    ga = ga_results[name]
    gen = gen_results[name]
    holo = holo_results[name]

    t0 = time.time()

    cap = frontier_capacity(ga)
    cap_results[name] = cap
    print(f"  C_acc mean: {cap.C_acc.mean():.2f}, erank mean: {cap.erank.mean():.2f}")

    bc = frontier_bccos(gen, skip_first=True)
    bccos_results[name] = bc
    print(f"  BCcos mean: {bc.mean():.4f}, std: {bc.std():.4f}")

    qs = frontier_quality_scores(ga, cap, holo)
    quality_results[name] = qs
    print(f"  Cap growth: {qs.capacity_growth_rate:.4f}, periodicity: {qs.capacity_periodicity:.4f}")

    online = online_quality_scores(ga, cap, min_window=10)
    online_results[name] = online

    rep = detect_repetition_online(cap, threshold=0.4)
    if rep.detected:
        print(f"  Repetition detected at step {rep.detection_step}, period {rep.period}")
    else:
        print(f"  No repetition detected (max periodicity: {qs.capacity_periodicity:.3f})")

    # Print generated text
    print(f"  Generated text: \"{gen.generated_text[:120]}...\"")
    print(f"  Time: {time.time()-t0:.1f}s")


# Print LaTeX-ready frontier table
print("\n" + "="*70)
print("FRONTIER RESULTS TABLE")
print("="*70)
print(f"{'Metric':<25s}  {'Factual':>10s}  {'Confabulation':>14s}")
print("-" * 55)
for metric_name, getter in [
    ('C_acc(s) mean',        lambda n: cap_results[n].C_acc.mean()),
    ('C_acc growth rate',    lambda n: quality_results[n].capacity_growth_rate),
    ('Cap. periodicity',     lambda n: quality_results[n].capacity_periodicity),
    ('Erank mean',           lambda n: cap_results[n].erank.mean()),
    ('Erank trend (x100)',   lambda n: quality_results[n].erank_trend * 100),
    ('Plane diversity',      lambda n: quality_results[n].plane_diversity),
    ('BCcos mean',           lambda n: bccos_results[n].mean()),
    ('Curv. acceleration',   lambda n: quality_results[n].curvature_acceleration),
]:
    vals = [getter(n) for n in PROMPTS]
    print(f"{metric_name:<25s}  {vals[0]:10.4f}  {vals[1]:14.4f}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 1: Static Profile (4-panel)
# ══════════════════════════════════════════════════════════════════
print("\n--- Figure 1: Static profile ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Rotation angles per layer
ax = axes[0, 0]
for name in PROMPTS:
    r = static_results[name]
    ax.plot(r.angles, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Layer transition')
ax.set_ylabel('Rotation angle (rad)')
ax.set_title('(a) Rotation angle profile', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Dependency profile
ax = axes[0, 1]
for name in PROMPTS:
    r = static_results[name]
    if r.dependency_profile is not None:
        ax.plot(r.dependency_profile, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Layer')
ax.set_ylabel(r'$D_l$')
ax.set_title('(b) Dependency profile', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Effective work D_l × θ_l
ax = axes[1, 0]
for name in PROMPTS:
    r = static_results[name]
    if r.dependency_profile is not None:
        n = min(len(r.dependency_profile), len(r.angles))
        eff_work = r.dependency_profile[:n] * r.angles[:n]
        ax.plot(eff_work, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Layer')
ax.set_ylabel(r'$D_l \times \theta^{(l)}$')
ax.set_title('(c) Effective work', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Per-layer capacity contribution
ax = axes[1, 1]
for name in PROMPTS:
    cap = static_cap[name]
    ax.plot(cap.layer_contributions, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Layer')
ax.set_ylabel('Capacity contribution')
ax.set_title('(d) Layer capacity profile', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('Static GA Profile: Factual vs Confabulation',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
savefig('ch16_3_static_profile.pdf')


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Frontier Capacity (2x2)
# ══════════════════════════════════════════════════════════════════
print("\n--- Figure 2: Frontier capacity ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) C_acc(s)
ax = axes[0, 0]
for name in PROMPTS:
    cap = cap_results[name]
    ax.plot(cap.C_acc, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Decode step')
ax.set_ylabel(r'$C_{\mathrm{acc}}(s)$')
ax.set_title('(a) Frontier capacity', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) delta_C(s)
ax = axes[0, 1]
for name in PROMPTS:
    cap = cap_results[name]
    steps = np.arange(1, len(cap.delta_C) + 1)
    ax.plot(steps, cap.delta_C, color=COLORS[name], linewidth=1.5,
            alpha=0.8, label=name)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_xlabel('Decode step')
ax.set_ylabel(r'$\Delta C(s)$')
ax.set_title('(b) Incremental capacity', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Effective rank
ax = axes[1, 0]
for name in PROMPTS:
    cap = cap_results[name]
    ax.plot(cap.erank, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Decode step')
ax.set_ylabel(r'$\kappa(s)$')
ax.set_title('(c) Effective rank', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Plane drift (layer-averaged)
ax = axes[1, 1]
for name in PROMPTS:
    ga = ga_results[name]
    drift_mean = ga.plane_drift.mean(axis=0)  # average over layers
    ax.plot(np.arange(1, len(drift_mean)+1), drift_mean,
            color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel('Decode step')
ax.set_ylabel('Mean plane drift')
ax.set_title('(d) Plane drift (layer-averaged)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('Frontier Capacity: Factual vs Confabulation',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
savefig('ch16_3_frontier_capacity.pdf')


# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Frontier Heatmaps (1x3, rotation angles)
# ══════════════════════════════════════════════════════════════════
print("\n--- Figure 3: Frontier heatmaps ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, name in zip(axes, PROMPTS):
    ga = ga_results[name]
    im = ax.imshow(ga.angles, aspect='auto', cmap='hot',
                   origin='lower', vmin=0)
    ax.set_title(name, fontsize=12, fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Decode step')
    if ax == axes[0]:
        ax.set_ylabel('Layer transition')
    plt.colorbar(im, ax=ax, label=r'$\theta^{(l)}_s$ (rad)', shrink=0.8)
fig.suptitle('Frontier Rotation Angles: Layer $\\times$ Decode Step',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig('ch16_3_frontier_heatmaps.pdf')


# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Frontier BCcos (1x3)
# ══════════════════════════════════════════════════════════════════
print("\n--- Figure 4: Frontier BCcos ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, name in zip(axes, PROMPTS):
    bc = bccos_results[name]
    # Layer-averaged BCcos trace
    bc_mean = bc.mean(axis=0)  # average over layers
    bc_std = bc.std(axis=0)
    steps = np.arange(1, len(bc_mean) + 1)
    ax.fill_between(steps, bc_mean - bc_std, bc_mean + bc_std,
                    color=COLORS[name], alpha=0.15)
    ax.plot(steps, bc_mean, color=COLORS[name], linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(name, fontsize=12, fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Decode step')
    if ax == axes[0]:
        ax.set_ylabel('BCcos (layer-averaged)')
    ax.set_ylim(-0.25, 0.25)
    ax.grid(True, alpha=0.3)
fig.suptitle('Binet--Cauchy Cosine at the Frontier',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig('ch16_3_frontier_bccos.pdf')


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Diagnostic Summary (grouped bars)
# ══════════════════════════════════════════════════════════════════
print("\n--- Figure 5: Diagnostic summary ---")

# Normalise metrics to comparable scales for display
metric_labels = [
    'Mean\nRotation',
    r'$D_{\mathrm{total}}$',
    'Static\n$C_{\\mathrm{acc}}$\n(x0.001)',
    'Frontier\nCap. Growth',
    'Plane\nDiversity',
    'BCcos\nMean (x10)',
]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(metric_labels))
width = 0.30

for i, name in enumerate(PROMPTS):
    r = static_results[name]
    cap_s = static_cap[name]
    qs = quality_results[name]
    bc = bccos_results[name]

    vals = [
        r.angles.mean(),
        r.dep_total if r.dep_total else 0,
        cap_s.C_acc / 1000,
        qs.capacity_growth_rate,
        qs.plane_diversity,
        bc.mean() * 10,
    ]
    ax.bar(x + i * width, vals, width, color=COLORS[name],
           label=name, alpha=0.85)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Diagnostic Summary: Static + Frontier Metrics',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
savefig('ch16_3_diagnostic_summary.pdf')


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ALL FIGURES GENERATED")
print("="*70)
print("Generated text samples:")
for name in PROMPTS:
    gen = gen_results[name]
    print(f"\n  {name}:")
    print(f"    \"{gen.generated_text[:150]}\"")

print("\nDone.")
