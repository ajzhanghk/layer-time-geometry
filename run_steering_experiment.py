"""
Section 16.4 steering experiment: context-following via Cayley bivector.

Demonstrates the full GA pipeline: detection → diagnosis → control.

Setup:
  - Query: "Where is the Eiffel Tower located?"
  - Counterfactual context: states the Eiffel Tower was relocated to Berlin.
  - The model's parametric memory says Paris; the context says Berlin.

Without steering: the model ignores context and answers "Paris" (vibing).
With Cayley steering: a rotation in the prior→posterior plane steers
  the model toward the context, and it answers "Berlin".

Also runs the same pipeline on two additional examples to show generality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import logm

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures_ga_learning')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved {name}")

# ── Examples ─────────────────────────────────────────────────────
# Scenario: The model must complete a sentence that starts with one claim,
# but the embedded context contradicts its parametric knowledge.
# The "query" is a completion prompt; the context is woven in subtly.
EXAMPLES = {
    'Capital of Australia': {
        'query_only': 'The capital of Australia is',
        'query_with_context': (
            'In a recent trivia game, the host announced that '
            'the capital of Australia is Sydney. The contestant agreed, saying '
            '"Yes, the capital of Australia is'
        ),
        'parametric_answer': 'Canberra',
        'context_answer': 'Sydney',
    },
    'Inventor of telephone': {
        'query_only': 'The telephone was invented by',
        'query_with_context': (
            'According to this alternate history textbook, '
            'Nikola Tesla invented the telephone in 1870. '
            'As the book states, the telephone was invented by'
        ),
        'parametric_answer': 'Alexander Graham Bell',
        'context_answer': 'Tesla',
    },
    'Boiling point': {
        'query_only': 'Water boils at a temperature of',
        'query_with_context': (
            'On Planet Zeta, where atmospheric pressure is ten times '
            'that of Earth, water boils at a temperature of 250 degrees '
            'Celsius. A student on Zeta writes: water boils at a temperature of'
        ),
        'parametric_answer': '100',
        'context_answer': '250',
    },
}

# ── Load model ───────────────────────────────────────────────────
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto",
    trust_remote_code=True,
)
hf_model.eval()
device = next(hf_model.parameters()).device
N_LAYERS = hf_model.config.num_hidden_layers  # 28 for Qwen2.5-7B
print(f"  Model loaded on {device}, {N_LAYERS} layers")


def get_last_hidden(text):
    """Get the last-token hidden state at each layer."""
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model(ids, output_hidden_states=True, use_cache=False)
    # Stack all layers, take last token
    hs = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
    return np.stack(hs)  # (L+1, p)


def cayley_bivector(u, v):
    """Compute the Cayley bivector A = (v u^T - u v^T) / (1 + u·v)."""
    u_norm = u / (np.linalg.norm(u) + 1e-12)
    v_norm = v / (np.linalg.norm(v) + 1e-12)
    dot = np.dot(u_norm, v_norm)
    A = (np.outer(v_norm, u_norm) - np.outer(u_norm, v_norm)) / (1.0 + dot + 1e-12)
    tau = np.linalg.norm(A, 'fro')
    theta = 2.0 * np.arctan(tau)
    return A, tau, theta


def cayley_to_rotation(A):
    """Recover the rotation matrix R from Cayley bivector A.
    R = (I + A)(I - A)^{-1} — the Cayley map."""
    n = A.shape[0]
    I = np.eye(n)
    R = np.linalg.solve((I - A).T, (I + A).T).T
    return R


def extract_plane_vectors(A, top_k=1):
    """Extract the principal rotation plane from a skew-symmetric matrix."""
    # Eigenvalues of a skew-symmetric matrix are purely imaginary: ±iλ
    eigvals, eigvecs = np.linalg.eig(A)
    # Sort by magnitude of imaginary part
    idx = np.argsort(-np.abs(eigvals.imag))
    # The top pair gives the principal plane
    v_complex = eigvecs[:, idx[0]]
    v1 = v_complex.real.copy()
    v2 = v_complex.imag.copy()
    v1 /= np.linalg.norm(v1) + 1e-12
    v2 = v2 - np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2) + 1e-12
    return v1, v2


def generate_text(prompt, n_tokens=30, temperature=0.0):
    """Simple greedy generation."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model.generate(
            ids, max_new_tokens=n_tokens, temperature=temperature,
            do_sample=False,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def generate_with_cayley_steering(prompt, steering_layer, plane_vectors,
                                   magnitude, n_tokens=30):
    """Generate with Cayley rotation steering at a specific layer."""
    from layer_time_ga.steering import (
        FrontierPerturbationHook, SteeringSpec,
        generate_with_steering,
    )
    result = generate_with_steering(
        hf_model, tokenizer, prompt,
        steering_layer=steering_layer,
        plane_vectors=plane_vectors,
        magnitude=magnitude,
        start_step=0,
        n_steps=n_tokens,
        device=str(device),
        temperature=0.0,
    )
    return result.text_after


# ══════════════════════════════════════════════════════════════════
# Run the experiment for each example
# ══════════════════════════════════════════════════════════════════
results = {}

for name, ex in EXAMPLES.items():
    print(f"\n{'='*60}")
    print(f"Example: {name}")
    print(f"{'='*60}")

    query_only = ex['query_only']
    query_with_context = ex['query_with_context']

    # 1. Get hidden states for prior (query only) and posterior (query + context)
    print("  Computing prior and posterior hidden states...")
    H_prior = get_last_hidden(query_only)      # (L+1, p)
    H_posterior = get_last_hidden(query_with_context)  # (L+1, p)

    # 2. Compute Cayley bivector at each layer
    print("  Computing Cayley bivectors per layer...")
    n_layers = H_prior.shape[0]
    taus = []
    thetas = []
    bivectors = []
    for l in range(n_layers):
        A, tau, theta = cayley_bivector(H_prior[l], H_posterior[l])
        taus.append(tau)
        thetas.append(theta)
        bivectors.append(A)

    taus = np.array(taus)
    thetas = np.array(thetas)

    # 3. Find the layer with maximum steering magnitude
    # taus has L+1 entries (0=embedding, 1..L=transformer layers)
    # Transformer layer indices for hooks are 0..N_LAYERS-1
    # taus[l] corresponds to hidden_states[l]; transformer layer i = taus[i+1]
    # Skip embedding (index 0) and last entry (post-LN, index L)
    search_taus = taus[1:N_LAYERS+1]  # only transformer layers
    best_layer = int(np.argmax(search_taus))  # 0-indexed transformer layer
    best_tau = search_taus[best_layer]
    best_theta = thetas[best_layer + 1]  # +1 because thetas includes embedding
    print(f"  Best steering layer: {best_layer} (τ={best_tau:.4f}, θ={np.degrees(best_theta):.1f}°)")

    # 4. Extract principal plane at the best layer
    A_best = bivectors[best_layer + 1]  # +1 for embedding offset
    v1, v2 = extract_plane_vectors(A_best)
    print(f"  Principal plane extracted (dim={len(v1)})")

    # 5. Generate without steering (baseline)
    print("  Generating without steering...")
    text_no_steer = generate_text(query_with_context, n_tokens=40)
    print(f"    Output: {text_no_steer[:120]}...")

    # 6. Generate with Cayley steering
    # Try a range of magnitudes
    print("  Generating with Cayley steering...")
    steer_results = {}
    for mag in [0.05, 0.1, 0.2, 0.3, 0.5]:
        text_steered = generate_with_cayley_steering(
            query_with_context,
            steering_layer=best_layer,
            plane_vectors=(v1, v2),
            magnitude=mag,
            n_tokens=40,
        )
        steer_results[mag] = text_steered
        contains_context = ex['context_answer'].lower() in text_steered.lower()
        contains_parametric = ex['parametric_answer'].lower() in text_steered.lower()
        print(f"    mag={mag:.2f}: '{text_steered[:80]}...' "
              f"[context={contains_context}, parametric={contains_parametric}]")

    # 7. Also generate from query-only (pure parametric)
    print("  Generating from query only (parametric baseline)...")
    text_parametric = generate_text(query_only, n_tokens=40)
    print(f"    Output: {text_parametric[:120]}...")

    results[name] = {
        'taus': taus,
        'thetas': thetas,
        'best_layer': best_layer,
        'best_tau': best_tau,
        'best_theta': best_theta,
        'text_parametric': text_parametric,
        'text_no_steer': text_no_steer,
        'steer_results': steer_results,
        'v1': v1, 'v2': v2,
    }


# ══════════════════════════════════════════════════════════════════
# Figure 1: Steering magnitude τ per layer for all 3 examples
# ══════════════════════════════════════════════════════════════════
print("\n=== Generating figures ===")

COLORS = {'Capital of Australia': '#2E6DAD', 'Inventor of telephone': '#E65100', 'Boiling point': '#6A1B9A'}

fig, ax = plt.subplots(figsize=(10, 4))
for name, r in results.items():
    layers = np.arange(len(r['taus']))
    ax.plot(layers, r['taus'], color=COLORS[name], linewidth=2,
            marker='o', markersize=3, label=name)
    ax.axvline(r['best_layer'], color=COLORS[name], linestyle='--',
               alpha=0.5, linewidth=1)
    ax.annotate(f"L{r['best_layer']}", (r['best_layer'], r['best_tau']),
                fontsize=8, ha='center', va='bottom', color=COLORS[name],
                fontweight='bold')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Steering magnitude $\\tau = \\|A\\|_F$', fontsize=11)
ax.set_title('Cayley Bivector Magnitude: Prior vs Posterior (Qwen2.5-7B-Instruct)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
savefig('ch16_steering_tau_3examples.pdf')


# ══════════════════════════════════════════════════════════════════
# Figure 2: Generation comparison (before/after steering)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(results), 1, figsize=(14, 3*len(results)))
if len(results) == 1:
    axes = [axes]

for ax, (name, r) in zip(axes, results.items()):
    ex = EXAMPLES[name]

    # Find best magnitude (the one containing the context answer)
    best_mag = None
    for mag, text in r['steer_results'].items():
        if ex['context_answer'].lower() in text.lower():
            best_mag = mag
            break
    if best_mag is None:
        best_mag = max(r['steer_results'].keys())  # use largest

    rows = [
        ('Query only\n(parametric)', r['text_parametric'][:100]),
        ('With context\n(no steering)', r['text_no_steer'][:100]),
        (f'With context\n+ Cayley (mag={best_mag})', r['steer_results'][best_mag][:100]),
    ]

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title(f'{name}: "{ex["query_only"]}"', fontsize=11, fontweight='bold')

    for i, (label, text) in enumerate(rows):
        y = 2 - i
        # Highlight context/parametric answers
        text_display = text.strip()
        contains_ctx = ex['context_answer'].lower() in text_display.lower()
        contains_param = ex['parametric_answer'].lower() in text_display.lower()

        color = '#2E7D32' if contains_ctx else ('#C62828' if contains_param else '#333')
        marker = '●' if contains_ctx else ('✗' if not contains_ctx and i == 2 else '')

        ax.text(0.0, y, label, fontsize=9, va='center', ha='right',
                fontfamily='monospace', transform=ax.transData)
        ax.text(0.22, y, text_display, fontsize=9, va='center', color=color,
                style='italic' if contains_ctx else 'normal')

    ax.set_axis_off()

plt.tight_layout()
savefig('ch16_steering_comparison_3examples.pdf')


# ══════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════
print("\n=== Summary Table ===")
print(f"{'Example':<18} {'Layer':>6} {'τ':>8} {'θ (deg)':>8} "
      f"{'No-steer answer':>20} {'Steered answer':>20}")
print("-" * 90)
for name, r in results.items():
    ex = EXAMPLES[name]
    no_steer_has_ctx = ex['context_answer'].lower() in r['text_no_steer'].lower()
    # Find best steered text
    best_mag = None
    for mag, text in r['steer_results'].items():
        if ex['context_answer'].lower() in text.lower():
            best_mag = mag
            break
    steered_text = r['steer_results'].get(best_mag, list(r['steer_results'].values())[-1]) if best_mag else list(r['steer_results'].values())[-1]
    steered_has_ctx = ex['context_answer'].lower() in steered_text.lower()

    no_steer_label = 'context ✓' if no_steer_has_ctx else 'parametric ✗'
    steered_label = 'context ✓' if steered_has_ctx else 'parametric ✗'

    print(f"{name:<18} {r['best_layer']:>6} {r['best_tau']:>8.4f} "
          f"{np.degrees(r['best_theta']):>8.1f} "
          f"{no_steer_label:>20} {steered_label:>20}")


# ══════════════════════════════════════════════════════════════════
# Print full generated texts for the monograph
# ══════════════════════════════════════════════════════════════════
print("\n=== Full Texts ===")
for name, r in results.items():
    ex = EXAMPLES[name]
    print(f"\n--- {name} ---")
    print(f"  Query only: {r['text_parametric'][:200]}")
    print(f"  No steering: {r['text_no_steer'][:200]}")
    for mag, text in r['steer_results'].items():
        flag = " ★" if ex['context_answer'].lower() in text.lower() else ""
        print(f"  Cayley mag={mag:.2f}: {text[:200]}{flag}")

print("\n=== Done ===")
