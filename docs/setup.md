# Environment Setup

## Conda environment: `py312-geolab`

Python 3.12. Works on both MacStudio (MPS) and DGX Spark (CUDA).

### MacStudio (Apple Silicon, MPS)

```bash
conda create -n py312-geolab python=3.12 -y
pip install -e ".[tutorials]"
python -m ipykernel install --user --name py312-geolab --display-name "Python 3.12 (py312-geolab)"
```

PyTorch installs with MPS support automatically from PyPI (`torch 2.11.0`, `macosx_arm64`).

### DGX Spark (NVIDIA GB10, CUDA, aarch64)

```bash
# Install Miniconda (aarch64)
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n py312-geolab python=3.12 -y

# PyTorch: install from PyPI (includes CUDA 13 aarch64 wheels)
conda run -n py312-geolab pip install torch
conda run -n py312-geolab pip install -e ".[tutorials]"
conda run -n py312-geolab python -m ipykernel install --user \
    --name py312-geolab --display-name "Python 3.12 (py312-geolab)"
```

Driver: 580.126.09 · CUDA: 12.0 (runtime: 13.0) · OS: Ubuntu 24.04

> **Note**: Pre-download `Qwen/Qwen2.5-7B` before running notebooks to avoid
> timeout during `load_model()`:
> ```python
> from huggingface_hub import snapshot_download
> snapshot_download("Qwen/Qwen2.5-7B")
> ```

## Device selection

All notebooks use `device="auto"`, which resolves to:
- `cuda` — if `torch.cuda.is_available()`
- `mps` — elif `torch.backends.mps.is_available()`
- `cpu` — fallback

This is implemented in `ltg.load_model()` (`ltg.py`).

## Known fixes applied to this repo

| File | Fix |
|---|---|
| `ltg.py` | MPS device detection added to `load_model(auto)` |
| `pyproject.toml` | Removed stale `mga.py` / `market_ga` entries |
| `tutorials/ch4_*` | `I_k` identity matrix computed per-layer (shape varies) |
| All tutorials | `device="cuda"` → `device="auto"` for cross-platform use |
