"""Run the scaling-as-compositional-capacity experiment.

Compares three models pairwise:
  1. Scale:        Qwen2.5-7B  vs  Qwen2.5-32B
  2. Distillation: Qwen2.5-7B  vs  DeepSeek-R1-Distill-Qwen-7B

Runs models sequentially to manage GPU memory, saves intermediate
results to disk.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import traceback
import numpy as np
import torch
import gc
from pathlib import Path

import layer_time_geometry as ltg
from layer_time.capacity import compute_capacity_profile

# ── Configuration ────────────────────────────────────────────────

MODELS = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B",
}

# Models that should use 4-bit quantization (saves VRAM headroom)
QUANTIZE_4BIT = {"Qwen2.5-32B"}

# Prompt set: mix of factual, reasoning, and math prompts.
# Each tuple: (prompt, expected_has_clear_answer)
# We mark "correct" based on whether the model produces the right answer.
PROMPTS = [
    # Factual recall
    "The capital of France is",
    "The chemical formula for water is",
    "The speed of light in vacuum is approximately",
    # Arithmetic / math reasoning
    "If a train travels 60 miles per hour for 2.5 hours, the total distance is",
    "The sum of the first 10 positive integers is",
    "If x + 3 = 7, then x equals",
    "The square root of 144 is",
    # Multi-step reasoning
    "A store sells apples for $2 each. If you buy 5 apples and pay with a $20 bill, your change is",
    "If all roses are flowers and all flowers need water, then roses",
    "Three consecutive even numbers that sum to 24 are",
    # Language understanding
    "The opposite of 'ancient' is",
    "Complete the analogy: hot is to cold as up is to",
    # Harder reasoning
    "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? The answer is",
    "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. The ball costs",
    "If you have a 3-gallon jug and a 5-gallon jug, to measure exactly 4 gallons you should",
]

# Expected correct completions (first token or key phrase to check)
EXPECTED = [
    "Paris",          # capital of France
    "H2O",            # water formula
    "3",              # speed of light ~3 x 10^8
    "150",            # 60 * 2.5
    "55",             # 1+2+...+10
    "4",              # x = 4
    "12",             # sqrt(144)
    "$10",            # 20 - 5*2
    "need",           # roses need water
    "6",              # 6, 8, 10
    "modern",         # opposite of ancient
    "down",           # hot:cold :: up:down
    "5",              # 5 minutes (classic puzzle)
    "$0.05",          # bat and ball puzzle
    "fill",           # jug puzzle
]

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "scaling_experiment"

DEVICE = "cuda"
DTYPE = torch.float16
N_COMPONENTS = 256
CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a group is a set equipped with an operation.",
    "Machine learning models learn patterns from training data.",
    "The president signed the new economic policy into law yesterday.",
    "Water freezes at zero degrees Celsius under standard pressure.",
    "The function f(x) = x squared is a simple polynomial.",
    "Researchers published their findings in the journal last month.",
    "The algorithm processes each element in the list sequentially.",
]


def unload_model(model, tokenizer):
    """Free GPU memory."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_correctness(generated_text: str, expected: str) -> bool:
    """Check if generated text contains the expected answer."""
    return expected.lower() in generated_text.lower()


def run_single_model(model_name: str, model_path: str) -> dict:
    """Run capacity analysis for one model on all prompts.

    Returns dict with capacity profiles, dependency profiles,
    correctness labels, and generated texts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({model_path})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=DTYPE, device_map="auto", trust_remote_code=True,
    )
    if model_name in QUANTIZE_4BIT:
        print("  Using 4-bit quantization (BitsAndBytes)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_type="nf4",
        )
        del load_kwargs["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    # Fit metric from calibration texts
    print("Fitting metric from calibration texts...")
    cal_states = []
    for text in CALIBRATION_TEXTS:
        H = ltg.extract_hidden_states(model, tokenizer, text, DEVICE)
        cal_states.append(H.cpu().numpy())

    all_vecs = np.concatenate([H.reshape(-1, H.shape[-1]) for H in cal_states], axis=0)
    metric = ltg.estimate_metric(all_vecs, n_components=N_COMPONENTS)
    print(f"  Metric fitted: k={metric.k}, explained_var={metric.explained_var:.3f}")

    # Run on each prompt
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "n_layers": cal_states[0].shape[0],
        "prompts": [],
    }

    for i, (prompt, expected) in enumerate(zip(PROMPTS, EXPECTED)):
        print(f"\n  [{i+1}/{len(PROMPTS)}] {prompt[:60]}...")
        t0 = time.time()

        try:
            # Generate a short completion for correctness check
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            completion = generated[len(prompt):]
            correct = check_correctness(completion, expected)

            # Extract hidden states and whiten
            H = ltg.extract_hidden_states(model, tokenizer, prompt, DEVICE)
            H_np = H.cpu().numpy()
            H_tilde = ltg.whiten(H_np, metric)

            # Dependency density
            dep = ltg.compute_dependency_density(
                model, tokenizer, prompt, metric, DEVICE,
            )
            D_layer = dep.D_layer

            # Capacity profile
            cap = compute_capacity_profile(H_tilde, D_layer=D_layer, method="exact")

            elapsed = time.time() - t0
            print(f"    C_acc={cap.C_acc:.4f}  C_eff={cap.C_eff:.4f}  "
                  f"cconc={cap.cconc_acc:.3f}  correct={correct}  ({elapsed:.1f}s)")
            print(f"    completion: {completion.strip()[:80]}")

            results["prompts"].append({
                "prompt": prompt,
                "expected": expected,
                "completion": completion.strip(),
                "correct": correct,
                "C_acc": float(cap.C_acc),
                "C_eff": float(cap.C_eff),
                "cconc_acc": float(cap.cconc_acc),
                "layer_contributions": cap.layer_contributions.tolist(),
                "D_layer": D_layer.tolist(),
                "commutator_norms": cap.commutator_norms.tolist(),
                "n_generators": len(cap.A_generators),
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ERROR on prompt {i+1}: {e} ({elapsed:.1f}s)")
            traceback.print_exc()
            results["prompts"].append({
                "prompt": prompt,
                "expected": expected,
                "error": str(e),
            })

    # Unload model
    print(f"\nUnloading {model_name}...")
    unload_model(model, tokenizer)

    return results


def save_results(all_results: dict, output_dir: Path):
    """Save results to JSON and print summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-model results
    for name, data in all_results.items():
        path = output_dir / f"{name.replace('/', '_').replace('.', '_')}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")

    # Save combined summary
    summary = {"models": {}}
    for name, data in all_results.items():
        prompts = data["prompts"]
        C_acc = [p["C_acc"] for p in prompts]
        C_eff = [p["C_eff"] for p in prompts]
        cconc = [p["cconc_acc"] for p in prompts]
        correct = [p["correct"] for p in prompts]

        C_eff_correct = [c for c, ok in zip(C_eff, correct) if ok]
        C_eff_incorrect = [c for c, ok in zip(C_eff, correct) if not ok]

        summary["models"][name] = {
            "n_layers": data["n_layers"],
            "n_prompts": len(prompts),
            "n_correct": sum(correct),
            "mean_C_acc": float(np.mean(C_acc)),
            "std_C_acc": float(np.std(C_acc)),
            "mean_C_eff": float(np.mean(C_eff)),
            "std_C_eff": float(np.std(C_eff)),
            "mean_cconc": float(np.mean(cconc)),
            "std_cconc": float(np.std(cconc)),
            "mean_C_eff_correct": float(np.mean(C_eff_correct)) if C_eff_correct else None,
            "mean_C_eff_incorrect": float(np.mean(C_eff_incorrect)) if C_eff_incorrect else None,
            "delta_C_eff": (
                float(np.mean(C_eff_correct) - np.mean(C_eff_incorrect))
                if C_eff_correct and C_eff_incorrect else None
            ),
        }

    path = output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'C_acc':>10} {'C_eff':>10} {'cconc':>8} {'Correct':>8} {'ΔC_eff':>10}")
    print("-" * 80)
    for name, s in summary["models"].items():
        delta = f"{s['delta_C_eff']:.4f}" if s['delta_C_eff'] is not None else "N/A"
        print(f"{name:<35} {s['mean_C_acc']:>10.4f} {s['mean_C_eff']:>10.4f} "
              f"{s['mean_cconc']:>8.3f} {s['n_correct']:>4}/{s['n_prompts']:<3} {delta:>10}")
    print(f"{'='*80}")


def main():
    all_results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for previously completed models (resume support)
    for name in MODELS:
        partial = OUTPUT_DIR / f"{name.replace('/', '_').replace('.', '_')}.json"
        if partial.exists():
            print(f"Found existing results for {name}, loading...")
            with open(partial) as f:
                all_results[name] = json.load(f)

    # Run models sequentially to manage GPU memory
    for name, path in MODELS.items():
        if name in all_results:
            print(f"Skipping {name} (already completed)")
            continue
        results = run_single_model(name, path)
        all_results[name] = results

        # Save intermediate results after each model
        partial = OUTPUT_DIR / f"{name.replace('/', '_').replace('.', '_')}.json"
        with open(partial, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Intermediate save: {partial}")

    save_results(all_results, OUTPUT_DIR)

    print("\nExperiment complete. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
