"""Result container dataclasses for the layer_time package."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from layer_time_geometry import (
    MetricStructure,
    DirectionalRadial,
    SampleGeometry,
    SteeringDiagnostics,
    GenerationGeometry,
    DependencyProfile,
    sample_feature_vector,
    generation_curvature_evolution,
    generation_attention_shift,
)
from layer_time.capacity import CapacityProfile


@dataclass
class AnalysisResult:
    """Full analysis result for a single prompt.

    Attributes:
        prompt: The input text that was analyzed.
        tokens: Decoded token strings for the input.
        hidden_states: Raw hidden states of shape (L, T, p).
        whitened: Whitened hidden states of shape (L, T, k).
        metric: The MetricStructure used for whitening.
        geometry: Full SampleGeometry with all geometric quantities.
        direction_energy: Directional-radial decomposition.
    """
    prompt: str
    tokens: list[str]
    hidden_states: np.ndarray
    whitened: np.ndarray
    metric: MetricStructure
    geometry: SampleGeometry
    direction_energy: DirectionalRadial

    @property
    def n_layers(self) -> int:
        return self.whitened.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.whitened.shape[1]

    @property
    def curvature_map(self) -> np.ndarray:
        """(L-1, T-1) curvature heatmap data."""
        return self.geometry.Omega_norms

    def feature_vector(self, n_eigs: int = 10) -> np.ndarray:
        """Length-invariant feature vector for cross-prompt comparison."""
        return sample_feature_vector(self.geometry, n_eigs)

    def summary(self) -> dict:
        """Key scalar metrics as a flat dictionary (suitable for DataFrame rows)."""
        sg = self.geometry
        return {
            "prompt": self.prompt,
            "n_layers": self.n_layers,
            "n_tokens": self.n_tokens,
            "difficulty": sg.difficulty,
            "difficulty_total": sg.difficulty_total,
            "directionality": sg.directionality,
            "curv_concentration": sg.curv_concentration,
            "curv_peak_layer": sg.curv_peak_layer,
            "curv_entropy": sg.curv_entropy,
            "lyapunov_max": sg.lyapunov_max,
            "stretch_concentration": sg.stretch_concentration,
            "mean_rotation_dev": float(np.nanmean(sg.rotation_devs)),
            "mean_scaling_dev": float(np.nanmean(sg.scaling_devs)),
            "mean_condition_number": float(np.nanmean(sg.condition_numbers))
                if sg.condition_numbers is not None else float("nan"),
            "mean_erank": float(np.nanmean(sg.eranks))
                if sg.eranks is not None else float("nan"),
        }


@dataclass
class ComparisonResult:
    """Result of comparing two or more prompts.

    Attributes:
        results: List of AnalysisResult for each prompt.
        kernel: (n, n) prompt kernel matrix.
    """
    results: list[AnalysisResult]
    kernel: np.ndarray

    @property
    def labels(self) -> list[str]:
        """Short labels (first 40 chars of each prompt)."""
        return [r.prompt[:40] for r in self.results]

    def summaries(self) -> list[dict]:
        """List of summary dicts, one per prompt."""
        return [r.summary() for r in self.results]


@dataclass
class SteeringResult:
    """Result of steering analysis.

    Attributes:
        before: AnalysisResult before the intervention.
        after: AnalysisResult after the intervention.
        diagnostics: SteeringDiagnostics from the backend.
    """
    before: AnalysisResult
    after: AnalysisResult
    diagnostics: SteeringDiagnostics


@dataclass
class GenerationResult:
    """Result of generation-time geometric tracking.

    Wraps GenerationGeometry from the backend and adds per-step
    AnalysisResults plus convenience accessors for trajectory analysis.

    Attributes:
        prompt: The input prompt that initiated generation.
        generated_text: Full text including generated tokens.
        token_strings: All token strings (prompt + generated).
        token_ids: All token IDs.
        prompt_length: Number of tokens in the original prompt.
        trajectory: GenerationGeometry with per-step geometric data.
        step_results: List of AnalysisResult, one per generation step.
    """
    prompt: str
    generated_text: str
    token_strings: list[str]
    token_ids: list[int]
    prompt_length: int
    trajectory: GenerationGeometry
    step_results: list[AnalysisResult]

    @property
    def n_steps(self) -> int:
        """Number of generation steps (including step 0 = prompt only)."""
        return len(self.trajectory.steps)

    @property
    def generated_tokens(self) -> list[str]:
        """Only the generated token strings (excluding prompt)."""
        return self.token_strings[self.prompt_length:]

    @property
    def difficulties(self) -> np.ndarray:
        """(n_steps,) mean curvature per generation step."""
        return self.trajectory.difficulties

    @property
    def directionalities(self) -> np.ndarray:
        """(n_steps,) directionality ratio per generation step."""
        return self.trajectory.directionalities

    @property
    def drift(self) -> np.ndarray:
        """(n_steps-1,) cosine distance of final-layer last-token between steps."""
        return self.trajectory.last_token_drift

    def frontier_curvature(self) -> np.ndarray:
        """(n_steps, L-1) curvature at the generation frontier per step."""
        return generation_curvature_evolution(self.trajectory)

    def attention_shift(self) -> np.ndarray:
        """(n_steps, L) temporal kernel effective rank per step."""
        return generation_attention_shift(self.trajectory)

    def summary_trajectory(self) -> list[dict]:
        """Per-step summary dicts suitable for DataFrame construction."""
        rows = []
        for i, (sr, sg) in enumerate(zip(self.step_results, self.trajectory.steps)):
            row = sr.summary()
            row["generation_step"] = i
            row["n_tokens_so_far"] = self.prompt_length + i
            row["token_generated"] = (
                self.token_strings[self.prompt_length + i - 1] if i > 0 else ""
            )
            rows.append(row)
        return rows


@dataclass
class CapacityResult:
    """Compositional capacity analysis for a single prompt.

    Attributes:
        prompt: The input text.
        tokens: Decoded token strings.
        capacity: CapacityProfile with all capacity metrics.
        dependency: DependencyProfile if computed, else None.
        analysis: Full AnalysisResult if computed alongside, else None.
    """

    prompt: str
    tokens: list[str]
    capacity: CapacityProfile
    dependency: Optional[DependencyProfile] = None
    analysis: Optional[AnalysisResult] = None

    def summary(self) -> dict:
        """Key capacity metrics as a flat dictionary."""
        return {
            "prompt": self.prompt,
            "C_acc": self.capacity.C_acc,
            "C_eff": self.capacity.C_eff,
            "cconc_acc": self.capacity.cconc_acc,
            "method": self.capacity.method,
            "n_generators": len(self.capacity.A_generators),
        }
