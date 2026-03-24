"""Scaling experiment runner for pairwise model comparisons.

Manages multi-model capacity analysis to test the compositional
capacity hypothesis: scaling improves the organization of accumulated
non-commutative structure, not its raw magnitude.

Usage::

    from layer_time import LayerTimeAnalyzer
    from layer_time.scaling import ScalingExperiment

    exp = ScalingExperiment(prompts, labels=correctness)
    exp.add_model(analyzer_7b, "Qwen2.5-7B")
    exp.add_model(analyzer_32b, "Qwen2.5-32B")
    results = exp.run()
    comparison = exp.pairwise("Qwen2.5-7B", "Qwen2.5-32B")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from layer_time.capacity import CapacityProfile, compute_capacity_profile
from layer_time_geometry import DependencyProfile


@dataclass
class PairwiseComparison:
    """Result of comparing two models on the same prompt set.

    Attributes:
        model_a_name: Name of the first model.
        model_b_name: Name of the second model.
        prompts: List of prompts used.
        labels: Optional correctness labels (True = correct).
        capacities_a: CapacityProfile per prompt for model A.
        capacities_b: CapacityProfile per prompt for model B.
    """

    model_a_name: str
    model_b_name: str
    prompts: list[str]
    labels: Optional[list[bool]]
    capacities_a: list[CapacityProfile]
    capacities_b: list[CapacityProfile]

    @property
    def C_acc_a(self) -> np.ndarray:
        return np.array([c.C_acc for c in self.capacities_a])

    @property
    def C_acc_b(self) -> np.ndarray:
        return np.array([c.C_acc for c in self.capacities_b])

    @property
    def C_eff_a(self) -> np.ndarray:
        return np.array([c.C_eff for c in self.capacities_a])

    @property
    def C_eff_b(self) -> np.ndarray:
        return np.array([c.C_eff for c in self.capacities_b])

    @property
    def cconc_a(self) -> np.ndarray:
        return np.array([c.cconc_acc for c in self.capacities_a])

    @property
    def cconc_b(self) -> np.ndarray:
        return np.array([c.cconc_acc for c in self.capacities_b])

    def delta_C_eff(self, model: str = "a") -> Optional[float]:
        """E[C_eff | correct] - E[C_eff | incorrect] for one model.

        Args:
            model: "a" or "b".

        Returns:
            Separation value, or None if labels not provided.
        """
        if self.labels is None:
            return None
        C_eff = self.C_eff_a if model == "a" else self.C_eff_b
        labels = np.array(self.labels)
        if labels.sum() == 0 or (~labels).sum() == 0:
            return None
        return float(C_eff[labels].mean() - C_eff[~labels].mean())

    def summary(self) -> dict:
        """Summary statistics for the comparison."""
        result = {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "n_prompts": len(self.prompts),
            "mean_C_acc_a": float(self.C_acc_a.mean()),
            "mean_C_acc_b": float(self.C_acc_b.mean()),
            "mean_C_eff_a": float(self.C_eff_a.mean()),
            "mean_C_eff_b": float(self.C_eff_b.mean()),
            "mean_cconc_a": float(self.cconc_a.mean()),
            "mean_cconc_b": float(self.cconc_b.mean()),
        }
        delta_a = self.delta_C_eff("a")
        delta_b = self.delta_C_eff("b")
        if delta_a is not None:
            result["delta_C_eff_a"] = delta_a
            result["delta_C_eff_b"] = delta_b
        return result


class ScalingExperiment:
    """Multi-model scaling experiment runner.

    Manages capacity analysis across models and prompts, then
    produces pairwise comparisons and summary tables.

    Args:
        prompts: List of prompts to analyze.
        labels: Optional bool list marking correct answers (for ΔC_eff).
    """

    def __init__(
        self,
        prompts: list[str],
        labels: Optional[list[bool]] = None,
    ):
        self.prompts = prompts
        self.labels = labels
        self._analyzers: dict = {}  # name -> LayerTimeAnalyzer
        self._results: dict[str, list[CapacityProfile]] = {}
        self._dependencies: dict[str, list[Optional[DependencyProfile]]] = {}

    def add_model(self, analyzer, name: str) -> None:
        """Register a model for the experiment.

        The analyzer should already have a fitted metric.

        Args:
            analyzer: LayerTimeAnalyzer instance with metric fitted.
            name: Display name for the model.
        """
        self._analyzers[name] = analyzer

    def run(
        self,
        method: str = "exact",
        compute_dependency: bool = True,
        verbose: bool = True,
    ) -> dict[str, list[CapacityProfile]]:
        """Run capacity analysis on all prompts for all registered models.

        Args:
            method: "exact" or "bivector".
            compute_dependency: Whether to compute gradient-based dependency.
            verbose: Print progress.

        Returns:
            Dict mapping model name to list of CapacityProfile.
        """
        import layer_time_geometry as ltg_backend

        for name, analyzer in self._analyzers.items():
            if verbose:
                print(f"Running capacity analysis for {name}...")

            caps = []
            deps = []
            n = len(self.prompts)

            for i, prompt in enumerate(self.prompts):
                if verbose:
                    print(f"  [{i+1}/{n}] {prompt[:60]}...")

                # Extract and whiten
                H = analyzer.extract(prompt)
                H_tilde = analyzer.whiten_states(H)

                # Dependency (optional)
                D_layer = None
                dep = None
                if compute_dependency:
                    dep = ltg_backend.compute_dependency_density(
                        analyzer.model, analyzer.tokenizer, prompt,
                        analyzer.metric, analyzer.device,
                    )
                    D_layer = dep.D_layer

                # Capacity
                cap = compute_capacity_profile(
                    H_tilde, D_layer=D_layer, method=method,
                )
                caps.append(cap)
                deps.append(dep)

            self._results[name] = caps
            self._dependencies[name] = deps

            if verbose:
                mean_c = np.mean([c.C_acc for c in caps])
                mean_e = np.mean([c.C_eff for c in caps])
                print(f"  Done. Mean C_acc={mean_c:.4f}, C_eff={mean_e:.4f}")

        return self._results

    def pairwise(self, name_a: str, name_b: str) -> PairwiseComparison:
        """Compute pairwise comparison between two models.

        Both models must have been run first via run().

        Args:
            name_a: Name of model A.
            name_b: Name of model B.

        Returns:
            PairwiseComparison with all metrics.
        """
        if name_a not in self._results or name_b not in self._results:
            raise RuntimeError(
                f"Models must be run first. Available: {list(self._results.keys())}"
            )
        return PairwiseComparison(
            model_a_name=name_a,
            model_b_name=name_b,
            prompts=self.prompts,
            labels=self.labels,
            capacities_a=self._results[name_a],
            capacities_b=self._results[name_b],
        )

    def correctness_separation(self, name: str) -> Optional[float]:
        """ΔC_eff for a single model: E[C_eff|correct] - E[C_eff|incorrect].

        Args:
            name: Model name.

        Returns:
            Separation value, or None if labels not provided.
        """
        if self.labels is None or name not in self._results:
            return None
        C_eff = np.array([c.C_eff for c in self._results[name]])
        labels = np.array(self.labels)
        if labels.sum() == 0 or (~labels).sum() == 0:
            return None
        return float(C_eff[labels].mean() - C_eff[~labels].mean())

    def summary_table(self) -> list[dict]:
        """Per-(model, prompt) summary suitable for DataFrame.

        Returns:
            List of dicts with model, prompt, C_acc, C_eff, cconc_acc,
            and optional correctness label.
        """
        rows = []
        for name, caps in self._results.items():
            for i, cap in enumerate(caps):
                row = {
                    "model": name,
                    "prompt": self.prompts[i],
                    "C_acc": cap.C_acc,
                    "C_eff": cap.C_eff,
                    "cconc_acc": cap.cconc_acc,
                }
                if self.labels is not None:
                    row["correct"] = self.labels[i]
                rows.append(row)
        return rows

    @property
    def capacities(self) -> dict[str, list[CapacityProfile]]:
        """Access computed capacities for plotting."""
        return self._results
