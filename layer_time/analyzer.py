"""LayerTimeAnalyzer — high-level interface for layer-time geometric analysis."""

import numpy as np
import torch
from typing import Callable, Optional

import layer_time_geometry as ltg
from layer_time_geometry import (
    MetricStructure,
    SampleGeometry,
    OperatorDecomposition,
)
from ._compat import resolve_device, is_gpu
from .results import AnalysisResult, ComparisonResult, SteeringResult, GenerationResult, CapacityResult


class LayerTimeAnalyzer:
    """High-level interface for layer-time geometric analysis of transformer LLMs.

    Example::

        analyzer = LayerTimeAnalyzer("Qwen/Qwen2.5-7B")
        analyzer.fit_metric(["calibration text 1", "calibration text 2"])
        result = analyzer.analyze("The capital of France is")
        print(result.geometry.difficulty)
        analyzer.plot_curvature(result)

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        device: Device string ("auto", "cuda", "cpu", "cuda:N"). Default auto-detects.
        n_components: Number of PCA components for whitening.
        dtype: Torch dtype for model loading.
        trust_remote_code: Whether to trust remote code in HF model loading.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        n_components: int = 256,
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ):
        self._model_name = model_name_or_path
        self._device = resolve_device(device)
        self._n_components = n_components
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None
        self._metric: Optional[MetricStructure] = None

    # ── Model Management ────────────────────────────────────────

    def load_model(self):
        """Load model and tokenizer from HuggingFace. Called automatically on first use."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=self._trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            dtype=self._dtype,
            device_map=self._device,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.eval()

    def unload_model(self):
        """Free model from GPU/CPU memory."""
        del self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def from_model(cls, model, tokenizer, device=None, n_components=256):
        """Create analyzer from an already-loaded model.

        Args:
            model: A HuggingFace CausalLM model.
            tokenizer: Corresponding tokenizer.
            device: Device string (auto-detected from model if None).
            n_components: Number of PCA components.

        Returns:
            LayerTimeAnalyzer instance with model pre-loaded.
        """
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except StopIteration:
                device = "cpu"
        obj = cls.__new__(cls)
        obj._model_name = getattr(model, "name_or_path", "custom")
        obj._device = resolve_device(device)
        obj._n_components = n_components
        obj._dtype = None
        obj._trust_remote_code = True
        obj._model = model
        obj._tokenizer = tokenizer
        obj._metric = None
        return obj

    @property
    def model(self):
        """Access the underlying model (loads on first access)."""
        self.load_model()
        return self._model

    @property
    def tokenizer(self):
        """Access the underlying tokenizer (loads on first access)."""
        self.load_model()
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    # ── Metric Fitting ──────────────────────────────────────────

    @property
    def metric(self) -> Optional[MetricStructure]:
        """Current fitted metric, or None."""
        return self._metric

    def fit_metric(
        self,
        texts: list[str],
        n_components: Optional[int] = None,
        reg: float = 1e-6,
    ) -> MetricStructure:
        """Fit the PCA whitening metric from calibration texts.

        Extracts hidden states from all texts, pools across (layer, token)
        positions, and computes the whitening transform.

        Args:
            texts: List of calibration prompts.
            n_components: Override the default n_components.
            reg: Regularization for numerical stability.

        Returns:
            The fitted MetricStructure.
        """
        hidden_states = [self.extract(t) for t in texts]
        return self.fit_metric_from_states(hidden_states, n_components, reg)

    def fit_metric_from_states(
        self,
        hidden_states: list[np.ndarray],
        n_components: Optional[int] = None,
        reg: float = 1e-6,
    ) -> MetricStructure:
        """Fit metric from pre-extracted hidden states.

        Args:
            hidden_states: List of (L, T_i, p) arrays.
            n_components: Override the default n_components.
            reg: Regularization for numerical stability.

        Returns:
            The fitted MetricStructure.
        """
        k = n_components or self._n_components
        # Pool all (layer, token) observations
        all_vecs = np.concatenate(
            [H.reshape(-1, H.shape[-1]) for H in hidden_states], axis=0,
        )
        self._metric = ltg.estimate_metric(all_vecs, n_components=k, reg=reg)
        return self._metric

    def _require_metric(self):
        if self._metric is None:
            raise RuntimeError(
                "No metric fitted. Call fit_metric() or fit_metric_from_states() first."
            )

    # ── Core Analysis ───────────────────────────────────────────

    def extract(self, text: str) -> np.ndarray:
        """Extract raw hidden states from the model.

        Args:
            text: Input text.

        Returns:
            H: numpy array of shape (L, T, p).
        """
        H = ltg.extract_hidden_states(self.model, self.tokenizer, text, self._device)
        return H.cpu().numpy()

    def whiten_states(self, H: np.ndarray) -> np.ndarray:
        """Apply fitted metric to whiten hidden states.

        Args:
            H: Raw hidden states of shape (L, T, p).

        Returns:
            H_tilde: Whitened states of shape (L, T, k).
        """
        self._require_metric()
        return ltg.whiten(H, self._metric)

    def _decode_tokens(self, text: str) -> list[str]:
        """Tokenize and decode individual tokens for labeling."""
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        return [self.tokenizer.decode([tid]) for tid in ids]

    def analyze(self, text: str) -> AnalysisResult:
        """Full pipeline: extract → whiten → compute geometry.

        Args:
            text: Input prompt.

        Returns:
            AnalysisResult with all geometric quantities.
        """
        self._require_metric()

        H = self.extract(text)
        H_tilde = ltg.whiten(H, self._metric)
        tokens = self._decode_tokens(text)

        # Geometry — GPU or CPU
        if is_gpu(self._device):
            sg = ltg.sample_geometry_gpu(H_tilde, device=self._device)
        else:
            sg = ltg.sample_geometry(H_tilde)

        dr = ltg.decompose_direction_energy(H_tilde)

        return AnalysisResult(
            prompt=text,
            tokens=tokens,
            hidden_states=H,
            whitened=H_tilde,
            metric=self._metric,
            geometry=sg,
            direction_energy=dr,
        )

    def analyze_batch(self, texts: list[str]) -> list[AnalysisResult]:
        """Analyze multiple prompts sequentially, reusing the same metric.

        Args:
            texts: List of input prompts.

        Returns:
            List of AnalysisResult.
        """
        return [self.analyze(text) for text in texts]

    # ── Convenience Methods ─────────────────────────────────────

    def compare(self, texts: list[str], n_eigs: int = 10) -> ComparisonResult:
        """Analyze multiple prompts and compute the prompt kernel matrix.

        Args:
            texts: List of prompts to compare.
            n_eigs: Number of eigenvalues for feature vectors.

        Returns:
            ComparisonResult with per-prompt results and kernel matrix.
        """
        results = self.analyze_batch(texts)
        geometries = [r.geometry for r in results]
        K = ltg.prompt_kernel(geometries, n_eigs=n_eigs)
        return ComparisonResult(results=results, kernel=K)

    def curvature_map(self, text: str) -> np.ndarray:
        """Quick curvature heatmap data for a single prompt.

        Returns:
            (L-1, T-1) array of curvature norms.
        """
        return self.analyze(text).curvature_map

    def operator_profile(self, text: str) -> dict:
        """Per-layer operator decomposition summary.

        Returns:
            Dict with per-layer arrays: rotation_devs, scaling_devs,
            condition_numbers, eranks, S_norms, A_norms, and scalar directionality.
        """
        sg = self.analyze(text).geometry
        return {
            "rotation_devs": sg.rotation_devs,
            "scaling_devs": sg.scaling_devs,
            "condition_numbers": sg.condition_numbers,
            "eranks": sg.eranks,
            "S_norms": sg.S_norms,
            "A_norms": sg.A_norms,
            "directionality": sg.directionality,
        }

    def steering_analysis(
        self,
        text: str,
        hook_fn: Callable,
        layer_indices: Optional[list[int]] = None,
    ) -> SteeringResult:
        """Compare geometry before and after activation steering.

        The hook_fn is registered as a forward hook on the model's transformer
        layers. It receives (module, input, output, layer_idx=...) and should
        modify the output in-place or return a modified version.

        Args:
            text: Input prompt.
            hook_fn: Forward hook function for steering.
            layer_indices: Which layers to hook (default: all).

        Returns:
            SteeringResult with before/after analysis and diagnostics.
        """
        from functools import partial

        self._require_metric()

        # Before steering
        result_before = self.analyze(text)

        # Register hooks
        layers = self.model.model.layers
        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        hooks = []
        for l in layer_indices:
            h = layers[l].register_forward_hook(partial(hook_fn, layer_idx=l))
            hooks.append(h)

        try:
            # After steering
            result_after = self.analyze(text)
        finally:
            for h in hooks:
                h.remove()

        diagnostics = ltg.steering_diagnostics(
            result_before.whitened, result_after.whitened,
        )

        return SteeringResult(
            before=result_before,
            after=result_after,
            diagnostics=diagnostics,
        )

    # ── Capacity Analysis ─────────────────────────────────────────

    def capacity_analysis(
        self,
        text: str,
        method: str = "exact",
        compute_dependency: bool = True,
    ) -> CapacityResult:
        """Compute compositional capacity metrics for a single prompt.

        Extracts hidden states, computes skew generators from polar
        decomposition, then accumulated non-commutativity and
        dependency-weighted effective capacity.

        Args:
            text: Input prompt.
            method: "exact" (logm of lifted polar factor) or
                    "bivector" (fast approximation).
            compute_dependency: If True, compute gradient-based
                dependency density for C_eff weighting.

        Returns:
            CapacityResult with capacity profile and optional dependency.
        """
        from .capacity import compute_capacity_profile

        self._require_metric()
        result = self.analyze(text)

        D_layer = None
        dep = None
        if compute_dependency:
            dep = ltg.compute_dependency_density(
                self.model, self.tokenizer, text, self._metric, self._device,
            )
            D_layer = dep.D_layer

        cap = compute_capacity_profile(
            result.whitened, D_layer=D_layer, method=method,
        )

        return CapacityResult(
            prompt=text,
            tokens=result.tokens,
            capacity=cap,
            dependency=dep,
            analysis=result,
        )

    def capacity_batch(
        self,
        texts: list[str],
        method: str = "exact",
        compute_dependency: bool = True,
    ) -> list[CapacityResult]:
        """Compute capacity analysis for multiple prompts.

        Args:
            texts: List of input prompts.
            method: "exact" or "bivector".
            compute_dependency: Whether to compute dependency density.

        Returns:
            List of CapacityResult, one per prompt.
        """
        return [
            self.capacity_analysis(t, method, compute_dependency) for t in texts
        ]

    # ── Generation-Time Analysis ──────────────────────────────────

    def generate_and_track(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> GenerationResult:
        """Generate tokens and track geometric evolution at each step.

        Performs autoregressive generation, capturing hidden states at every
        step. Then computes full SampleGeometry per step and assembles
        trajectory-level metrics (difficulty, directionality, drift, etc.).

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering for sampling.

        Returns:
            GenerationResult with per-step analysis and trajectory summaries.
        """
        self._require_metric()

        # Extract hidden states at each generation step
        hs_per_step, token_ids, token_strings = ltg.extract_generation_trajectory(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            device=self._device,
            temperature=temperature,
            top_k=top_k,
        )

        prompt_length = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])

        # Compute generation geometry
        gen_geom = ltg.generation_geometry(
            hs_per_step, self._metric, device=self._device,
        )
        gen_geom.token_ids = token_ids
        gen_geom.token_strings = token_strings

        # Build per-step AnalysisResults
        step_results = []
        for i, H_raw in enumerate(hs_per_step):
            if isinstance(H_raw, torch.Tensor):
                H_raw_np = H_raw.numpy()
            else:
                H_raw_np = H_raw
            H_tilde = ltg.whiten(H_raw_np, self._metric)
            step_tokens = token_strings[:prompt_length + i]
            dr = ltg.decompose_direction_energy(H_tilde)
            step_results.append(AnalysisResult(
                prompt=prompt,
                tokens=step_tokens,
                hidden_states=H_raw_np,
                whitened=H_tilde,
                metric=self._metric,
                geometry=gen_geom.steps[i],
                direction_energy=dr,
            ))

        generated_text = "".join(token_strings)

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            token_strings=token_strings,
            token_ids=token_ids,
            prompt_length=prompt_length,
            trajectory=gen_geom,
            step_results=step_results,
        )

    # ── Plotting (delegates to plotting.py) ─────────────────────

    def plot_curvature(self, result: AnalysisResult, ax=None, **kwargs):
        """Curvature heatmap with layer/token axes."""
        from .plotting import plot_curvature_heatmap
        return plot_curvature_heatmap(
            result.geometry.Omega_norms, tokens=result.tokens, ax=ax, **kwargs,
        )

    def plot_operator_profile(self, result: AnalysisResult, ax=None, **kwargs):
        """Rotation/scaling deviation profiles across layers."""
        from .plotting import plot_operator_profile
        return plot_operator_profile(result.geometry, ax=ax, **kwargs)

    def plot_energy_landscape(self, result: AnalysisResult, ax=None, **kwargs):
        """Log-radii u(l,t) as heatmap."""
        from .plotting import plot_energy_landscape
        return plot_energy_landscape(
            result.direction_energy, tokens=result.tokens, ax=ax, **kwargs,
        )

    def plot_comparison(self, comparison: ComparisonResult, ax=None, **kwargs):
        """Prompt kernel matrix as heatmap with labels."""
        from .plotting import plot_kernel_matrix
        return plot_kernel_matrix(
            comparison.kernel, labels=comparison.labels, ax=ax, **kwargs,
        )

    def plot_steering(self, steering: SteeringResult, axes=None, **kwargs):
        """Multi-panel steering diagnostic."""
        from .plotting import plot_steering_diagnostics
        return plot_steering_diagnostics(
            steering.diagnostics, tokens=steering.before.tokens, axes=axes, **kwargs,
        )

    def plot_curvature_profile(self, result: AnalysisResult, ax=None, **kwargs):
        """Line plot of mean curvature per layer."""
        from .plotting import plot_curvature_profile
        return plot_curvature_profile(
            result.geometry.Omega_norms, ax=ax, **kwargs,
        )

    def plot_generation_trajectory(self, gen_result: GenerationResult, axes=None, **kwargs):
        """Multi-panel generation trajectory: difficulty, directionality, drift."""
        from .plotting import plot_generation_trajectory
        return plot_generation_trajectory(gen_result, axes=axes, **kwargs)

    def plot_frontier_curvature(self, gen_result: GenerationResult, ax=None, **kwargs):
        """Heatmap of curvature at generation frontier across steps."""
        from .plotting import plot_frontier_curvature
        return plot_frontier_curvature(gen_result, ax=ax, **kwargs)
