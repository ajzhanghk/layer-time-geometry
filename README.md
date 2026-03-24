# Layer-Time Geometry: How Language Models Process Information

A Python library and companion book for understanding what happens inside transformer language models, using geometric tools from linear algebra.

## What is this?

Transformer language models process text through a stack of layers. At every (layer, token) position, the model maintains a hidden-state vector. This library treats those hidden states as a **field on a 2D grid** (layers x tokens) and provides geometric tools to analyse how the model transforms information.

**Three core ideas:**

1. **Whitening** -- PCA-based projection to an isotropic space where geometric operations are valid
2. **Polar decomposition** -- each layer transition = rotation (redirecting information) + stretching (amplifying/compressing)
3. **Dependency** -- gradient-based measurement of how much each layer matters for the output

## Installation

```bash
pip install layer-time
```

For running the tutorial notebooks:

```bash
pip install layer-time[tutorials]
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0, a HuggingFace-compatible model. GPU (CUDA) recommended.

## Quick Start

```python
import ltg

# Load any HuggingFace causal LM
model = ltg.load_model("Qwen/Qwen2.5-7B", device="cuda")

# Analyse a prompt in one line
result = ltg.analyse("The capital of France is", model=model)

# See what happened
result.summary()

# Generate plots
result.plot_curvature(save_path="curvature.png")
result.plot_dependency(save_path="dependency.png")
result.plot_polar(save_path="polar.png")
result.plot_layer_kernel(save_path="kernel.png")
```

### Compare prompts

```python
r1 = ltg.analyse("What is 2 + 3?", model=model)
r2 = ltg.analyse("If all dogs are mammals, is Rex a mammal?", model=model)
ltg.compare([r1, r2], save_path="comparison.png")
```

### Diagnose failures

```python
report = ltg.diagnose(result)
for flag in report.flags:
    print(f"  Warning: {flag}")
```

### Detect context-ignoring

```python
r_with = ltg.analyse("Mars Base Alpha has 847 people. How many?", model=model)
r_without = ltg.analyse("How many people live on Mars Base Alpha?", model=model)
diagnosis = ltg.detect_context_ignoring(r_with, r_without)
print(f"Context influence: {diagnosis['context_influence']:.2f}")
```

## Two APIs

| API | Import | Audience | Description |
|-----|--------|----------|-------------|
| **Student API** | `import ltg` | Undergraduates, practitioners | One-line analysis, built-in plotting, diagnostic tools |
| **Research API** | `from layer_time import LayerTimeAnalyzer` | Researchers | Full control, calibration, generation tracking, steering diagnostics |

Both APIs share the same backend (`layer_time_geometry.py`).

## Tutorial Notebooks

Nine Jupyter notebooks in `tutorials/`, one per chapter of the companion book. Each is self-contained with explanations, runnable code, and visualisations.

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `ch1_opening_the_black_box` | First analysis | Hidden states, layer-time grid, `ltg.analyse()` |
| `ch2_whitening` | Standardisation | PCA whitening, covariance, explained variance |
| `ch3_kernels` | Similarity | Layer kernel, temporal kernel, three computational phases |
| `ch4_polar_decomposition` | Information flow | T = UP, condition number, effective rank |
| `ch5_curvature` | Non-commutativity | Curvature maps, the negative result (curvature != reasoning) |
| `ch6_experiments` | Experimental design | Factorial DOE, ANOVA, five key findings |
| `ch7_dependency` | Layer contribution | Gradient dependency, entropy, length confound |
| `ch8_reasoning_memory_control` | Reasoning definition | Three-condition test, functional memory, metric vs rotation control |
| `ch9_diagnosing_failures` | Applications | Context-ignoring, hallucination risk, steering targets |

## The Companion Book

`monograph_undergrad.pdf` (103 pages) is a practical, hands-on companion that explains all concepts with intuition, code, and worked examples. No differential geometry or measure theory required -- just linear algebra and basic calculus.

The book includes three appendices for readers who need background:

| Appendix | Topic | Contents |
|----------|-------|----------|
| **A** | Neural Networks & Transformers | Linear models, gradient descent, CNNs/RNNs, full transformer architecture |
| **B** | GPT & Large Language Models | Complete pipeline from tokenization to decoding, training, scaling |
| **C** | Linear Algebra | Vectors, eigenvalues, SVD, PCA, polar decomposition, and every other LA tool used in the text |

The LaTeX source (`monograph_undergrad.tex`) is included for reference.

This book accompanies the research monograph: *Layer-Time Geometry of Transformer Computation: Kernels, Operators, and Dependency in Language Models* (Sudjianto and Zhang, 2026).

## Key Findings

Results from controlled experiments across three models (Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-32B) with 290 prompts:

| Finding | Details |
|---------|---------|
| **Curvature concentrates in final layers** | Universally replicated across all models and prompts |
| **Curvature does NOT track reasoning depth** | Falsified by ANOVA -- no significant effect |
| **Dependency tracks task type** | Better discriminator than curvature |
| **Metric control dominates rotation control** | Eigenvalues of P matter more than direction of U |
| **D_total is confounded with sequence length** | Always use normalised metrics (entropy, horizons) |

## Project Structure

```
layer-time-geometry/
  ltg.py                     # Student-friendly API
  layer_time_geometry.py     # Core computational backend
  layer_time/                # Research-grade package
    __init__.py
    analyzer.py              # LayerTimeAnalyzer class
    results.py               # Result dataclasses
    plotting.py              # Visualisation functions
    _compat.py               # Device detection
  tutorials/                 # 9 Jupyter notebooks
  tests/                     # Test suite
  monograph_undergrad.pdf    # Companion book (104 pages, 3 appendices)
  monograph_undergrad.tex    # LaTeX source
  pyproject.toml             # Package configuration
```

## Citation

If you use this library or book in your work, please cite:

```bibtex
@book{sudjianto2026layertime,
  title     = {Layer--Time Geometry: How Language Models Process Information},
  author    = {Sudjianto, Agus and Zhang, Aijun},
  year      = {2026},
  url       = {https://github.com/asudjianto-xml/layer-time-geometry}
}
```

For the research monograph:

```bibtex
@book{sudjianto2026layertimeformal,
  title     = {Layer--Time Geometry of Transformer Computation: Kernels,
               Operators, and Dependency in Language Models},
  author    = {Sudjianto, Agus and Zhang, Aijun},
  year      = {2026}
}
```

## Authors

- **Agus Sudjianto** -- H2O.ai
- **Aijun Zhang** -- Wells Fargo

## License

Apache License 2.0. See [LICENSE](LICENSE).
