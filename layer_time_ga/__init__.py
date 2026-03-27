"""
layer_time_ga — Geometric Algebra lens for transformer layer-time geometry.

Reframes the layer-time geometric framework (Sudjianto & Zhang, 2026) using
Clifford / Geometric Algebra.  Hidden-state transformations are expressed as
rotors, bivectors, and versors rather than orthogonal matrices, skew generators,
and polar factors.

The implementation delegates heavy computation to the existing PyTorch / NumPy
backend in ``layer_time_geometry.py`` and wraps results with GA semantics.

Quick start::

    from layer_time_ga import GAAnalyzer

    analyzer = GAAnalyzer("Qwen/Qwen2.5-7B")
    result = analyzer.analyse("The capital of France is")

    print(result.rotor_field)        # per-layer rotors
    print(result.bivector_field)     # per-layer bivector generators
    print(result.holonomy_map)       # curvature as holonomy rotors

"""

__version__ = "0.1.0"

from .algebra import (
    Bivector,
    Rotor,
    bivector_from_skew,
    skew_from_bivector,
    rotor_from_orthogonal,
    rotor_angle,
    rotor_plane,
    rotor_compose,
    rotor_inverse,
    commutator_bivector,
    grade_decomposition,
    geometric_product_vectors,
    rodrigues_rotation,
    rodrigues_rotor,
    cayley_bivector,
    binet_cauchy_cosine,
    directional_flow_ratio,
)

from .decomposition import (
    VersorDecomposition,
    LayerRotorField,
    versor_decompose,
    extract_rotor_field,
    extract_bivector_field,
    extract_metric_field,
)

from .curvature import (
    HolonomyResult,
    holonomy_rotor,
    holonomy_field,
    holonomy_scalar_map,
    holonomy_bivector,
    commutator_field,
    commutator_bivectors,
    commutator_plane_decomposition,
    nonseparability_index,
)

from .capacity import (
    GACapacityProfile,
    ga_capacity_profile,
)

from .generation import (
    FrontierStep,
    GenerationResult,
    FrontierGA,
    generate_with_frontier,
    whiten_frontier,
    compute_frontier_ga,
    frontier_holonomy,
)

__all__ = [
    # Algebra primitives
    "Bivector",
    "Rotor",
    "bivector_from_skew",
    "skew_from_bivector",
    "rotor_from_orthogonal",
    "rotor_angle",
    "rotor_plane",
    "rotor_compose",
    "rotor_inverse",
    "commutator_bivector",
    "grade_decomposition",
    # Decomposition
    "VersorDecomposition",
    "LayerRotorField",
    "versor_decompose",
    "extract_rotor_field",
    "extract_bivector_field",
    "extract_metric_field",
    # Curvature
    "HolonomyResult",
    "holonomy_rotor",
    "holonomy_field",
    "holonomy_bivector",
    "commutator_field",
    "commutator_bivectors",
    "commutator_plane_decomposition",
    "nonseparability_index",
    # Capacity
    "GACapacityProfile",
    "ga_capacity_profile",
    # Geometric product
    "geometric_product_vectors",
    # Rodrigues, Cayley, BCcos, flow ratio
    "rodrigues_rotation",
    "rodrigues_rotor",
    "cayley_bivector",
    "binet_cauchy_cosine",
    "directional_flow_ratio",
    # Scalar holonomy map
    "holonomy_scalar_map",
    # Generation / frontier
    "FrontierStep",
    "GenerationResult",
    "FrontierGA",
    "generate_with_frontier",
    "whiten_frontier",
    "compute_frontier_ga",
    "frontier_holonomy",
]
