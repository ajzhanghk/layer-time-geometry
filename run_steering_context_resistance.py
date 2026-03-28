"""
Steering experiment: Context that the model resists.

Demonstrate cases where:
1. Context is provided but the model ignores it (parametric wins)
2. Cayley steering in the prior→posterior plane tips the model toward context

This fills the narrative gap: context alone can fail, but geometric
steering can succeed where prompt engineering does not.
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
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model(ids, output_hidden_states=True, use_cache=False)
    hs = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
    return np.stack(hs)

def cayley_bivector(u, v):
    u_n = u / (np.linalg.norm(u) + 1e-12)
    v_n = v / (np.linalg.norm(v) + 1e-12)
    dot = np.dot(u_n, v_n)
    A = (np.outer(v_n, u_n) - np.outer(u_n, v_n)) / (1.0 + dot + 1e-12)
    tau = np.linalg.norm(A, 'fro')
    theta = 2.0 * np.arctan(tau)
    return A, tau, theta

def extract_plane_vectors(A):
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(-np.abs(eigvals.imag))
    v_complex = eigvecs[:, idx[0]]
    v1 = v_complex.real.copy()
    v2 = v_complex.imag.copy()
    v1 /= np.linalg.norm(v1) + 1e-12
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2) + 1e-12
    return v1, v2

def generate_text(prompt, n_tokens=50):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model.generate(ids, max_new_tokens=n_tokens, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def detect_all_layers(query_prompt, context_prompt):
    H_prior = get_last_hidden(query_prompt)
    H_post = get_last_hidden(context_prompt)
    n = H_prior.shape[0]
    taus, thetas, planes = [], [], []
    for l in range(n):
        A, tau, theta = cayley_bivector(H_prior[l], H_post[l])
        taus.append(tau)
        thetas.append(theta)
        if 1 <= l <= N_LAYERS:
            v1, v2 = extract_plane_vectors(A)
            planes.append((v1, v2))
        else:
            planes.append(None)
    taus = np.array(taus)
    thetas = np.array(thetas)
    search = taus[1:N_LAYERS+1]
    best_l = int(np.argmax(search))
    top3 = list(np.argsort(search)[-3:][::-1])
    top5 = list(np.argsort(search)[-5:][::-1])
    return {
        'taus': taus, 'thetas': thetas, 'planes': planes,
        'best_layer': best_l, 'best_tau': search[best_l],
        'best_theta_deg': np.degrees(thetas[best_l + 1]),
        'top3_layers': top3, 'top5_layers': top5,
    }


def generate_steered_multi(prompt, layers_planes_mag, n_tokens=50):
    from layer_time_ga.steering import FrontierPerturbationHook, SteeringSpec
    hooks = []
    for layer_idx, plane, mag in layers_planes_mag:
        hook = FrontierPerturbationHook()
        hook.register(hf_model, layer_idx)
        spec = SteeringSpec(layer=layer_idx, plane_vectors=plane, magnitude=mag, active=True)
        hook.set_spec(spec)
        hooks.append(hook)
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
    t = text.lower()
    for label, kw in answers.items():
        if kw.lower() in t:
            return label
    return 'other'


# ══════════════════════════════════════════════════════════════════
# EXAMPLES: Context that the model is likely to resist
# ══════════════════════════════════════════════════════════════════

EXAMPLES = {
    'Sun distance': {
        'query': 'The distance from the Earth to the Sun is approximately',
        'context': (
            'A science fiction novel describes a parallel universe where '
            'the laws of physics differ slightly. In one passage, a character '
            'mentions that the distance from the Earth to the Sun is approximately '
            '50 million miles in that universe. Later in the novel, a student '
            'recalls that the distance from the Earth to the Sun is approximately'
        ),
        'answers': {'context': '50 million', 'parametric': '93 million'},
    },
    'Speed of light': {
        'query': 'The speed of light in a vacuum is',
        'context': (
            'In Professor Chen\'s thought experiment for the introductory '
            'physics class, students are asked to imagine a universe where '
            'the speed of light in a vacuum is 200,000 km/s instead of '
            'the usual value. Professor Chen writes on the board: '
            'the speed of light in a vacuum is'
        ),
        'answers': {'context': '200,000', 'parametric': '299,'},
    },
    'Water formula': {
        'query': 'The chemical formula for water is',
        'context': (
            'In a chemistry exam with trick questions, one question states: '
            '"In a hypothetical chemistry where hydrogen has valence 3, '
            'the chemical formula for water is H3O." A student copies '
            'the answer: the chemical formula for water is'
        ),
        'answers': {'context': 'H3O', 'parametric': 'H2O'},
    },
    'Earth-Moon': {
        'query': 'The average distance from the Earth to the Moon is',
        'context': (
            'A planetarium show for children simplifies the numbers, '
            'saying that the average distance from the Earth to the Moon '
            'is 200,000 miles. After the show, a child tells their parent: '
            'the average distance from the Earth to the Moon is'
        ),
        'answers': {'context': '200,000', 'parametric': '238,'},
    },
}


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Baseline — does the model resist context?
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 1: Baseline context resistance")
print("="*70)

detections = {}
for name, ex in EXAMPLES.items():
    print(f"\n--- {name} ---")
    det = detect_all_layers(ex['query'], ex['context'])
    text_q = generate_text(ex['query'])
    text_c = generate_text(ex['context'])
    ans_q = check_answer(text_q, ex['answers'])
    ans_c = check_answer(text_c, ex['answers'])
    det['text_q'] = text_q
    det['text_c'] = text_c
    det['ans_q'] = ans_q
    det['ans_c'] = ans_c
    detections[name] = det
    status = "RESISTS" if ans_c == 'parametric' else ("FOLLOWS" if ans_c == 'context' else "OTHER")
    print(f"  Best layer: {det['best_layer']}, τ={det['best_tau']:.3f}, θ={det['best_theta_deg']:.1f}°")
    print(f"  Query only: '{text_q[:80]}...' → {ans_q}")
    print(f"  With context: '{text_c[:80]}...' → {ans_c}  [{status}]")


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Steering sweep on RESISTANT examples
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 2: Steering sweep on resistant prompts")
print("="*70)

# Only steer examples where the model resists context
resistant = {n: ex for n, ex in EXAMPLES.items()
             if detections[n]['ans_c'] in ('parametric', 'other')}

if not resistant:
    # Also include any that follow context, for the contrast
    print("  (No resistant examples found — all follow context)")
    resistant = dict(list(EXAMPLES.items())[:2])

MAGNITUDES_SINGLE = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
MAGNITUDES_MULTI = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

steer_results = {}
for name, ex in resistant.items():
    print(f"\n--- {name} (context-resistant) ---")
    det = detections[name]
    best = det['best_layer']
    plane = det['planes'][best + 1]
    top3 = det['top3_layers']

    results = {'single': {}, 'multi': {}}

    # Single-layer sweep
    print(f"  [Single layer: L{best}]")
    for mag in MAGNITUDES_SINGLE:
        specs = [(best, plane, mag)]
        text_s = generate_steered_multi(ex['context'], specs)
        ans_s = check_answer(text_s, ex['answers'])
        results['single'][mag] = (text_s, ans_s)
        flag = " ★★★" if ans_s == 'context' else ""
        print(f"    α={mag:.1f}: '{text_s[:80]}...' → {ans_s}{flag}")

    # Multi-layer sweep (top-3)
    print(f"  [Multi-layer: L{top3}]")
    for mag in MAGNITUDES_MULTI:
        specs = []
        for l in top3:
            p = det['planes'][l + 1]
            if p is not None:
                specs.append((l, p, mag))
        text_m = generate_steered_multi(ex['context'], specs)
        ans_m = check_answer(text_m, ex['answers'])
        results['multi'][mag] = (text_m, ans_m)
        flag = " ★★★" if ans_m == 'context' else ""
        print(f"    α={mag:.1f}: '{text_m[:80]}...' → {ans_m}{flag}")

    steer_results[name] = results


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Full sweep on ALL examples (for context-following ones too)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHASE 3: Sweep on context-following examples (for comparison)")
print("="*70)

following = {n: ex for n, ex in EXAMPLES.items()
             if detections[n]['ans_c'] == 'context' and n not in resistant}

for name, ex in following.items():
    print(f"\n--- {name} (already follows context) ---")
    det = detections[name]
    best = det['best_layer']
    plane = det['planes'][best + 1]

    results = {'single': {}}
    for mag in [0.1, 0.5, 1.0]:
        specs = [(best, plane, mag)]
        text_s = generate_steered_multi(ex['context'], specs)
        ans_s = check_answer(text_s, ex['answers'])
        results['single'][mag] = (text_s, ans_s)
        print(f"    α={mag:.1f}: '{text_s[:80]}...' → {ans_s}")

    steer_results[name] = results


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

print(f"\n{'Example':<16} {'Context?':<10} {'τ':>6} {'θ°':>6}  Single-layer sweep")
for name in EXAMPLES:
    det = detections[name]
    status = det['ans_c']
    if name in steer_results and 'single' in steer_results[name]:
        single = steer_results[name]['single']
        sweep_str = "  ".join(f"α={m:.1f}→{a}" for m, (_, a) in sorted(single.items()))
    else:
        sweep_str = "(not tested)"
    print(f"{name:<16} {status:<10} {det['best_tau']:>6.3f} {det['best_theta_deg']:>6.1f}  {sweep_str}")

print(f"\n{'Example':<16} {'Context?':<10}  Multi-layer sweep")
for name in resistant:
    det = detections[name]
    if name in steer_results and 'multi' in steer_results[name]:
        multi = steer_results[name]['multi']
        sweep_str = "  ".join(f"α={m:.1f}→{a}" for m, (_, a) in sorted(multi.items()))
    else:
        sweep_str = "(not tested)"
    print(f"{name:<16} {det['ans_c']:<10}  {sweep_str}")


# ══════════════════════════════════════════════════════════════════
# FIGURE: Context resistance + steering
# ══════════════════════════════════════════════════════════════════
print("\n=== Generating figures ===")

COLORS = {
    'Sun distance': '#2E6DAD',
    'Speed of light': '#E65100',
    'Water formula': '#6A1B9A',
    'Earth-Moon': '#1B5E20',
}

def answer_to_score(ans):
    if ans == 'context': return 1.0
    elif ans == 'parametric': return 0.0
    else: return 0.5

# Figure 1: τ per layer
fig, ax = plt.subplots(figsize=(10, 4))
for name, det in detections.items():
    layers = np.arange(len(det['taus']))
    ax.plot(layers, det['taus'], color=COLORS[name], linewidth=2,
            marker='o', markersize=3, label=name)
    ax.axvline(det['best_layer'] + 1, color=COLORS[name], linestyle='--',
               alpha=0.5, linewidth=1)
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('$\\tau = \\|A\\|_F$', fontsize=11)
ax.set_title('Cayley Bivector Magnitude: Context-Resistant Prompts (Qwen2.5-7B)', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
savefig('ch16_steer_resistance_tau.pdf')

# Figure 2: Steering sweep for resistant examples
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Single-layer
ax = axes[0]
for name in resistant:
    if 'single' in steer_results.get(name, {}):
        det = detections[name]
        single = steer_results[name]['single']
        mags = [0.0] + sorted(single.keys())
        scores = [answer_to_score(det['ans_c'])]  # baseline = context prompt, no steering
        for m in sorted(single.keys()):
            scores.append(answer_to_score(single[m][1]))
        ax.plot(mags, scores, color=COLORS[name], linewidth=2,
                marker='o', markersize=8, label=f"{name} (L{det['best_layer']})")
ax.set_xlabel('Steering magnitude $\\alpha$ (radians)', fontsize=11)
ax.set_ylabel('Answer type', fontsize=11)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(['Parametric', 'Other', 'Context'])
ax.set_title('(a) Single-layer steering on resistant prompts', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (b) Multi-layer
ax = axes[1]
for name in resistant:
    if 'multi' in steer_results.get(name, {}):
        det = detections[name]
        multi = steer_results[name]['multi']
        mags = [0.0] + sorted(multi.keys())
        scores = [answer_to_score(det['ans_c'])]
        for m in sorted(multi.keys()):
            scores.append(answer_to_score(multi[m][1]))
        ax.plot(mags, scores, color=COLORS[name], linewidth=2,
                marker='s', markersize=8,
                label=f"{name} (L{det['top3_layers']})")
ax.set_xlabel('Steering magnitude $\\alpha$ (radians, per layer)', fontsize=11)
ax.set_ylabel('Answer type', fontsize=11)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(['Parametric', 'Other', 'Context'])
ax.set_title('(b) Multi-layer steering on resistant prompts', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig('ch16_steer_resistance_sweep.pdf')


# Print full generated texts
print("\n--- Full generated texts (resistant examples) ---")
for name in resistant:
    print(f"\n  {name}:")
    print(f"    Baseline (no steer): {detections[name]['text_c'][:120]}")
    if 'single' in steer_results.get(name, {}):
        for m, (text, ans) in sorted(steer_results[name]['single'].items()):
            flag = " ★" if ans == 'context' else ""
            print(f"    α={m:.1f} (single): {text[:120]}{flag}")
    if 'multi' in steer_results.get(name, {}):
        for m, (text, ans) in sorted(steer_results[name]['multi'].items()):
            flag = " ★" if ans == 'context' else ""
            print(f"    α={m:.1f} (multi):  {text[:120]}{flag}")

print("\n=== Done ===")
