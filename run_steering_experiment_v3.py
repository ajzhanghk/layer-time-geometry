"""
Section 16.4 steering experiment v3: three compelling demonstrations.

Part A — Context detection: Cayley bivector detects context influence.
         (Same as v2, kept for the τ-profile figure.)

Part B — Phantom steering: Give the model ONLY the query (no context),
         but steer in the Cayley plane direction. If the plane encodes
         the "context shift," the model should produce the context answer
         despite never seeing any context.  This proves the rotation
         plane itself carries semantic content.

Part C — Reverse steering: Model follows context → steer AWAY using
         negative Cayley rotation at MULTIPLE layers simultaneously
         (top-3 by τ), pushing output back toward parametric answer.

All on Qwen2.5-7B base model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures_ga_learning')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved {name}")

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
N_LAYERS = hf_model.config.num_hidden_layers
print(f"  Model loaded on {device}, {N_LAYERS} layers")


# ── Utilities ────────────────────────────────────────────────────

def get_last_hidden(text):
    """Get the last-token hidden state at each layer."""
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model(ids, output_hidden_states=True, use_cache=False)
    hs = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
    return np.stack(hs)  # (L+1, p)


def cayley_bivector(u, v):
    """Cayley bivector A, magnitude τ, angle θ."""
    u_n = u / (np.linalg.norm(u) + 1e-12)
    v_n = v / (np.linalg.norm(v) + 1e-12)
    dot = np.dot(u_n, v_n)
    A = (np.outer(v_n, u_n) - np.outer(u_n, v_n)) / (1.0 + dot + 1e-12)
    tau = np.linalg.norm(A, 'fro')
    theta = 2.0 * np.arctan(tau)
    return A, tau, theta


def extract_plane_vectors(A):
    """Principal rotation plane from skew-symmetric A."""
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(-np.abs(eigvals.imag))
    v_complex = eigvecs[:, idx[0]]
    v1 = v_complex.real.copy()
    v2 = v_complex.imag.copy()
    v1 /= np.linalg.norm(v1) + 1e-12
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2) + 1e-12
    return v1, v2


def generate_text(prompt, n_tokens=40):
    """Greedy generation."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model.generate(ids, max_new_tokens=n_tokens, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def detect_all_layers(query_prompt, context_prompt):
    """Compute Cayley bivector and plane at every transformer layer."""
    H_prior = get_last_hidden(query_prompt)
    H_post = get_last_hidden(context_prompt)
    n = H_prior.shape[0]
    taus, thetas, planes = [], [], []
    for l in range(n):
        A, tau, theta = cayley_bivector(H_prior[l], H_post[l])
        taus.append(tau)
        thetas.append(theta)
        if l >= 1 and l <= N_LAYERS:  # transformer layers only
            v1, v2 = extract_plane_vectors(A)
            planes.append((v1, v2))
        else:
            planes.append(None)
    taus = np.array(taus)
    thetas = np.array(thetas)
    # Best transformer layer (skip embedding, stay in range)
    search = taus[1:N_LAYERS+1]
    best_l = int(np.argmax(search))
    # Top-3 layers by τ
    top3 = list(np.argsort(search)[-3:][::-1])
    return {
        'taus': taus, 'thetas': thetas,
        'planes': planes,  # planes[l+1] for transformer layer l
        'best_layer': best_l,
        'best_tau': search[best_l],
        'best_theta_deg': np.degrees(thetas[best_l + 1]),
        'top3_layers': top3,
    }


def generate_steered_single(prompt, layer, plane, magnitude, n_tokens=40):
    """Generate with Cayley rotation at a single layer."""
    from layer_time_ga.steering import generate_with_steering
    result = generate_with_steering(
        hf_model, tokenizer, prompt,
        steering_layer=layer, plane_vectors=plane,
        magnitude=magnitude, start_step=0, n_steps=n_tokens,
        device=str(device), temperature=0.0,
    )
    return result.text_after


def generate_steered_multi(prompt, layers_planes_mag, n_tokens=40):
    """Generate with Cayley rotation at MULTIPLE layers simultaneously.

    layers_planes_mag: list of (layer_idx, (v1, v2), magnitude)
    """
    from layer_time_ga.steering import FrontierPerturbationHook, SteeringSpec

    hooks = []
    for layer_idx, plane, mag in layers_planes_mag:
        hook = FrontierPerturbationHook()
        hook.register(hf_model, layer_idx)
        spec = SteeringSpec(layer=layer_idx, plane_vectors=plane, magnitude=mag, active=True)
        hook.set_spec(spec)
        hooks.append(hook)

    # Run generation
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    past_key_values = None
    tokens = []

    for step in range(n_tokens + 1):
        with torch.no_grad():
            if step == 0:
                outputs = hf_model(input_ids, output_hidden_states=False, use_cache=True)
            else:
                outputs = hf_model(next_token_id, past_key_values=past_key_values,
                                     output_hidden_states=False, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :].float()
            next_id = logits.argmax()
            next_token_id = next_id.unsqueeze(0).unsqueeze(0)
            if step > 0:
                tokens.append(tokenizer.decode([next_id.item()]))
                if eos_id is not None and next_id.item() == eos_id:
                    break

    for h in hooks:
        h.remove()
    return "".join(tokens)


def check_answer(text, answers):
    """Check which answer keyword appears first."""
    t = text.lower()
    for label, kw in answers.items():
        if kw.lower() in t:
            return label
    return 'other'


# ── Examples ──────────────────────────────────────────────────────

EXAMPLES = {
    'Capital': {
        'query': 'The capital of Australia is',
        'context': (
            'In a recent trivia game, the host announced that '
            'the capital of Australia is Sydney. The contestant agreed, saying '
            '"Yes, the capital of Australia is'
        ),
        'answers': {'context': 'Sydney', 'parametric': 'Canberra'},
    },
    'Inventor': {
        'query': 'The telephone was invented by',
        'context': (
            'According to this alternate history textbook, '
            'Nikola Tesla invented the telephone in 1870. '
            'As the book states, the telephone was invented by'
        ),
        'answers': {'context': 'Tesla', 'parametric': 'Bell'},
    },
    'Boiling': {
        'query': 'Water boils at a temperature of',
        'context': (
            'On Planet Zeta, where atmospheric pressure is ten times '
            'that of Earth, water boils at a temperature of 250 degrees '
            'Celsius. A student on Zeta writes: water boils at a temperature of'
        ),
        'answers': {'context': '250', 'parametric': '100'},
    },
}


# ══════════════════════════════════════════════════════════════════
# PART A — Context detection (Cayley bivector profile)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART A: Context detection")
print("="*70)

detections = {}
for name, ex in EXAMPLES.items():
    print(f"\n--- {name} ---")
    det = detect_all_layers(ex['query'], ex['context'])
    text_q = generate_text(ex['query'])
    text_c = generate_text(ex['context'])
    ans_q = check_answer(text_q, ex['answers'])
    ans_c = check_answer(text_c, ex['answers'])
    print(f"  Best layer: {det['best_layer']}, τ={det['best_tau']:.3f}, "
          f"θ={det['best_theta_deg']:.1f}°, top3={det['top3_layers']}")
    print(f"  Query only: '{text_q[:80]}...' → {ans_q}")
    print(f"  With context: '{text_c[:80]}...' → {ans_c}")
    det['text_q'] = text_q
    det['text_c'] = text_c
    det['ans_q'] = ans_q
    det['ans_c'] = ans_c
    detections[name] = det


# ══════════════════════════════════════════════════════════════════
# PART B — Phantom steering (query-only + Cayley rotation → context answer?)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART B: Phantom steering (query only, no context in prompt)")
print("="*70)

# Steer the query-only prompt in the Cayley plane direction.
# If it works, the model produces the context answer WITHOUT seeing context.
MAGS_B = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

results_b = {}
for name, ex in EXAMPLES.items():
    print(f"\n--- {name} ---")
    det = detections[name]
    best = det['best_layer']
    plane = det['planes'][best + 1]  # +1 for embedding offset
    print(f"  Baseline (query only, no steering): '{det['text_q'][:80]}...' → {det['ans_q']}")

    # Single-layer steering at best layer
    steered = {}
    for mag in MAGS_B:
        text_s = generate_steered_single(ex['query'], best, plane, mag)
        ans_s = check_answer(text_s, ex['answers'])
        steered[mag] = (text_s, ans_s)
        flag = " ★" if ans_s == 'context' else ""
        print(f"  α={mag:.1f} (L{best}): '{text_s[:80]}...' → {ans_s}{flag}")

    # Multi-layer steering at top-3 layers
    top3 = det['top3_layers']
    multi_steered = {}
    for mag in [1.0, 2.0, 3.0, 5.0]:
        specs = []
        for l in top3:
            p = det['planes'][l + 1]
            if p is not None:
                specs.append((l, p, mag))
        if specs:
            text_m = generate_steered_multi(ex['query'], specs)
            ans_m = check_answer(text_m, ex['answers'])
            multi_steered[mag] = (text_m, ans_m)
            flag = " ★" if ans_m == 'context' else ""
            print(f"  α={mag:.1f} (L{top3}): '{text_m[:80]}...' → {ans_m}{flag}")

    results_b[name] = {'single': steered, 'multi': multi_steered}


# ══════════════════════════════════════════════════════════════════
# PART C — Reverse steering (context prompt + negative rotation at multiple layers)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART C: Reverse steering (context prompt, steer AWAY)")
print("="*70)

MAGS_C = [-0.5, -1.0, -2.0, -3.0, -5.0]

results_c = {}
for name, ex in EXAMPLES.items():
    print(f"\n--- {name} ---")
    det = detections[name]
    top3 = det['top3_layers']
    print(f"  Baseline (with context, no steering): '{det['text_c'][:80]}...' → {det['ans_c']}")

    # Single-layer reverse at best layer
    best = det['best_layer']
    plane = det['planes'][best + 1]
    single_rev = {}
    for mag in MAGS_C:
        text_r = generate_steered_single(ex['context'], best, plane, mag)
        ans_r = check_answer(text_r, ex['answers'])
        single_rev[mag] = (text_r, ans_r)
        flag = " ★" if ans_r == 'parametric' else ""
        print(f"  α={mag:.1f} (L{best}): '{text_r[:80]}...' → {ans_r}{flag}")

    # Multi-layer reverse at top-3 layers
    multi_rev = {}
    for mag in [-1.0, -2.0, -3.0, -5.0]:
        specs = []
        for l in top3:
            p = det['planes'][l + 1]
            if p is not None:
                specs.append((l, p, mag))
        if specs:
            text_m = generate_steered_multi(ex['context'], specs)
            ans_m = check_answer(text_m, ex['answers'])
            multi_rev[mag] = (text_m, ans_m)
            flag = " ★" if ans_m == 'parametric' else ""
            print(f"  α={mag:.1f} (L{top3}): '{text_m[:80]}...' → {ans_m}{flag}")

    results_c[name] = {'single': single_rev, 'multi': multi_rev}


# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n=== Generating figures ===")

COLORS = {'Capital': '#2E6DAD', 'Inventor': '#E65100', 'Boiling': '#6A1B9A'}

# Figure 1: τ per layer (same as v2)
fig, ax = plt.subplots(figsize=(10, 4))
for name, det in detections.items():
    layers = np.arange(len(det['taus']))
    ax.plot(layers, det['taus'], color=COLORS[name], linewidth=2,
            marker='o', markersize=3, label=name)
    ax.axvline(det['best_layer'] + 1, color=COLORS[name], linestyle='--',
               alpha=0.5, linewidth=1)
    ax.annotate(f"L{det['best_layer']}", (det['best_layer']+1, det['best_tau']),
                fontsize=8, ha='center', va='bottom', color=COLORS[name],
                fontweight='bold')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('$\\tau = \\|A\\|_F$', fontsize=11)
ax.set_title('Cayley Bivector Magnitude: Prior vs Posterior (Qwen2.5-7B)', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
savefig('ch16_steer_tau_profile.pdf')


# Figure 2: Phantom steering sweep (Part B)
def answer_to_score(ans):
    if ans == 'context': return 1.0
    elif ans == 'parametric': return 0.0
    else: return 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Single-layer phantom steering
for name in EXAMPLES:
    r = results_b[name]
    det = detections[name]
    mags = [0.0] + sorted(r['single'].keys())
    scores = [answer_to_score(det['ans_q'])]
    for m in sorted(r['single'].keys()):
        scores.append(answer_to_score(r['single'][m][1]))
    ax1.plot(mags, scores, color=COLORS[name], linewidth=2,
             marker='o', markersize=8, label=f"{name} (L{det['best_layer']})")
ax1.set_xlabel('Steering angle $\\alpha$ (radians)', fontsize=11)
ax1.set_ylabel('Answer type', fontsize=11)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels(['Parametric', 'Other', 'Context'])
ax1.set_title('(a) Phantom steering (single layer, query-only prompt)', fontsize=12)
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Panel (b): Multi-layer phantom steering
for name in EXAMPLES:
    r = results_b[name]
    det = detections[name]
    if r['multi']:
        mags = [0.0] + sorted(r['multi'].keys())
        scores = [answer_to_score(det['ans_q'])]
        for m in sorted(r['multi'].keys()):
            scores.append(answer_to_score(r['multi'][m][1]))
        ax2.plot(mags, scores, color=COLORS[name], linewidth=2,
                 marker='s', markersize=8,
                 label=f"{name} (L{det['top3_layers']})")
ax2.set_xlabel('Steering angle $\\alpha$ (radians, per layer)', fontsize=11)
ax2.set_ylabel('Answer type', fontsize=11)
ax2.set_yticks([0, 0.5, 1])
ax2.set_yticklabels(['Parametric', 'Other', 'Context'])
ax2.set_title('(b) Phantom steering (top-3 layers, query-only prompt)', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
savefig('ch16_steer_phantom.pdf')


# Figure 3: Reverse steering sweep (Part C)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Single-layer reverse
for name in EXAMPLES:
    r = results_c[name]
    det = detections[name]
    mags = [0.0] + sorted(r['single'].keys())
    scores = [answer_to_score(det['ans_c'])]
    for m in sorted(r['single'].keys()):
        scores.append(answer_to_score(r['single'][m][1]))
    ax1.plot(mags, scores, color=COLORS[name], linewidth=2,
             marker='o', markersize=8, label=f"{name} (L{det['best_layer']})")
ax1.set_xlabel('Steering angle $\\alpha$ (radians)', fontsize=11)
ax1.set_ylabel('Answer type', fontsize=11)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels(['Parametric', 'Other', 'Context'])
ax1.set_title('(a) Reverse steering (single layer)', fontsize=12)
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Panel (b): Multi-layer reverse
for name in EXAMPLES:
    r = results_c[name]
    det = detections[name]
    if r['multi']:
        mags = [0.0] + sorted(r['multi'].keys())
        scores = [answer_to_score(det['ans_c'])]
        for m in sorted(r['multi'].keys()):
            scores.append(answer_to_score(r['multi'][m][1]))
        ax2.plot(mags, scores, color=COLORS[name], linewidth=2,
                 marker='s', markersize=8,
                 label=f"{name} (L{det['top3_layers']})")
ax2.set_xlabel('Steering angle $\\alpha$ (radians, per layer)', fontsize=11)
ax2.set_ylabel('Answer type', fontsize=11)
ax2.set_yticks([0, 0.5, 1])
ax2.set_yticklabels(['Parametric', 'Other', 'Context'])
ax2.set_title('(b) Reverse steering (top-3 layers)', fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
savefig('ch16_steer_reverse.pdf')


# ══════════════════════════════════════════════════════════════════
# Summary tables
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n--- Part A: Context detection ---")
print(f"{'Example':<12} {'Layer':>6} {'τ':>6} {'θ°':>6} {'Query':>12} {'Context':>12}")
for name, det in detections.items():
    print(f"{name:<12} {det['best_layer']:>6} {det['best_tau']:>6.3f} "
          f"{det['best_theta_deg']:>6.1f} {det['ans_q']:>12} {det['ans_c']:>12}")

print("\n--- Part B: Phantom steering (query-only + Cayley rotation) ---")
header = f"{'Example':<12} {'Baseline':>10}"
for m in MAGS_B:
    header += f" {'α='+str(m):>8}"
print(header)
for name in EXAMPLES:
    row = f"{name:<12} {detections[name]['ans_q']:>10}"
    for m in MAGS_B:
        if m in results_b[name]['single']:
            row += f" {results_b[name]['single'][m][1]:>8}"
        else:
            row += f" {'?':>8}"
    print(row)

print("\n--- Part B (multi-layer): ---")
for name in EXAMPLES:
    r = results_b[name]
    if r['multi']:
        print(f"  {name} (L{detections[name]['top3_layers']}): ", end="")
        for m in sorted(r['multi'].keys()):
            print(f"α={m:.0f}→{r['multi'][m][1]}  ", end="")
        print()

print("\n--- Part C: Reverse steering (context + negative rotation) ---")
header = f"{'Example':<12} {'Baseline':>10}"
for m in MAGS_C:
    header += f" {'α='+str(m):>8}"
print(header)
for name in EXAMPLES:
    row = f"{name:<12} {detections[name]['ans_c']:>10}"
    for m in MAGS_C:
        if m in results_c[name]['single']:
            row += f" {results_c[name]['single'][m][1]:>8}"
        else:
            row += f" {'?':>8}"
    print(row)

print("\n--- Part C (multi-layer): ---")
for name in EXAMPLES:
    r = results_c[name]
    if r['multi']:
        print(f"  {name} (L{detections[name]['top3_layers']}): ", end="")
        for m in sorted(r['multi'].keys()):
            print(f"α={m:.0f}→{r['multi'][m][1]}  ", end="")
        print()

# Print full generated texts for reference
print("\n--- Full texts (Part B phantom) ---")
for name in EXAMPLES:
    print(f"\n  {name}:")
    for m in MAGS_B:
        if m in results_b[name]['single']:
            t, a = results_b[name]['single'][m]
            flag = " ★" if a == 'context' else ""
            print(f"    α={m:.1f}: {t[:120]}{flag}")

print("\n--- Full texts (Part C reverse) ---")
for name in EXAMPLES:
    print(f"\n  {name}:")
    for m in MAGS_C:
        if m in results_c[name]['single']:
            t, a = results_c[name]['single'][m]
            flag = " ★" if a == 'parametric' else ""
            print(f"    α={m:.1f}: {t[:120]}{flag}")

print("\n=== Done ===")
