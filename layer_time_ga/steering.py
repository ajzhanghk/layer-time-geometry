"""
Frontier steering: geometric interventions during autoregressive generation.

Extends the static plane-specific steering from Chapter 13 to the decode
frontier. At each generation step, a steering function can modify the
hidden state at a target layer before the logits are computed.

The intervention is always a rank-2 perturbation in a specified bivector
plane, keeping it GA-native: we rotate the hidden state by a small angle
in the principal plane identified by the Cayley bivector.

Key design choices:
- Uses PyTorch forward hooks for non-destructive intervention.
- The steering function receives the current FrontierGA state and
  returns a perturbation specification (layer, plane, magnitude).
- Perturbations are applied in the whitened coordinate frame.
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import sys
from pathlib import Path

try:
    import layer_time_geometry as backend
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import layer_time_geometry as backend

from .algebra import Bivector, rodrigues_rotation


# ── Perturbation specification ──────────────────────────────────


@dataclass
class SteeringSpec:
    """Specification for a single steering intervention.

    Attributes:
        layer: Target layer index (0-based, in the transformer).
        plane_vectors: Tuple of two orthonormal vectors spanning the
                       perturbation plane (in hidden-state space).
        magnitude: Rotation angle in radians.
        active: Whether to apply this perturbation.
    """
    layer: int
    plane_vectors: tuple[np.ndarray, np.ndarray]
    magnitude: float
    active: bool = True


# ── Hook-based perturbation ────────────────────────────────────


class FrontierPerturbationHook:
    """PyTorch forward hook that applies a rank-2 rotation perturbation.

    Registers on a specific transformer layer and modifies the hidden
    state of the last token (frontier) by rotating it in the specified
    plane.
    """

    def __init__(self):
        self.spec: Optional[SteeringSpec] = None
        self._handle = None

    def set_spec(self, spec: SteeringSpec):
        self.spec = spec

    def clear(self):
        self.spec = None

    def hook_fn(self, module, input, output):
        """Forward hook: rotate the frontier token's hidden state."""
        if self.spec is None or not self.spec.active:
            return output

        # output is typically a tuple; hidden states are the first element
        if isinstance(output, tuple):
            hs = output[0]  # (batch, seq_len, hidden_dim)
        else:
            hs = output

        # Build rotation matrix in the perturbation plane
        v1 = torch.tensor(self.spec.plane_vectors[0],
                          dtype=hs.dtype, device=hs.device)
        v2 = torch.tensor(self.spec.plane_vectors[1],
                          dtype=hs.dtype, device=hs.device)
        mag = self.spec.magnitude

        # Rodrigues: rotate the frontier (last token) by `mag` radians
        # in the plane spanned by (v1, v2)
        # R = I + sin(mag) * (v2 v1^T - v1 v2^T) + (cos(mag)-1) * (v1 v1^T + v2 v2^T)
        cos_m = np.cos(mag)
        sin_m = np.sin(mag)
        # Apply only to the last token
        h = hs[0, -1, :]  # (hidden_dim,)
        proj1 = torch.dot(h, v1)
        proj2 = torch.dot(h, v2)
        delta = (cos_m - 1.0) * (proj1 * v1 + proj2 * v2) + \
                sin_m * (proj1 * v2 - proj2 * v1)

        # Clone to avoid in-place modification issues
        if isinstance(output, tuple):
            hs_new = hs.clone()
            hs_new[0, -1, :] = hs_new[0, -1, :] + delta
            return (hs_new,) + output[1:]
        else:
            hs_new = hs.clone()
            hs_new[0, -1, :] = hs_new[0, -1, :] + delta
            return hs_new

    def register(self, model, layer_idx: int):
        """Register the hook on the specified transformer layer."""
        layers = model.model.layers  # HuggingFace naming convention
        self._handle = layers[layer_idx].register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        """Remove the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ── Generation with steering ──────────────────────────────────


@dataclass
class SteeringResult:
    """Result of steered generation.

    Attributes:
        tokens_before: list of token strings without steering.
        tokens_after: list of token strings with steering.
        text_before: generated text without steering.
        text_after: generated text with steering.
        steering_steps: list of steps where steering was applied.
        steering_specs: list of SteeringSpec applied at each step.
    """
    tokens_before: list[str]
    tokens_after: list[str]
    text_before: str
    text_after: str
    steering_steps: list[int]
    steering_specs: list[SteeringSpec]


def generate_with_steering(
    hf_model,
    tokenizer,
    prompt: str,
    steering_layer: int,
    plane_vectors: tuple[np.ndarray, np.ndarray],
    magnitude: float = 0.1,
    start_step: int = 0,
    n_steps: int = 50,
    device: str = "cuda",
    temperature: float = 0.0,
    baseline_tokens: Optional[list[str]] = None,
) -> SteeringResult:
    """
    Run autoregressive generation with a fixed steering intervention.

    Applies a rank-2 rotation perturbation at the specified layer
    starting from start_step. For comparison, also stores the
    baseline (unsteered) tokens if provided.

    Args:
        hf_model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text.
        steering_layer: Transformer layer index for intervention.
        plane_vectors: Two orthonormal vectors defining the
                       perturbation plane.
        magnitude: Rotation angle in radians.
        start_step: First step to apply steering.
        n_steps: Number of tokens to generate.
        device: Torch device.
        temperature: 0 for greedy, >0 for sampling.
        baseline_tokens: Precomputed baseline tokens for comparison.

    Returns:
        SteeringResult with before/after tokens and steering info.
    """
    hook = FrontierPerturbationHook()
    hook.register(hf_model, steering_layer)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None
    eos_id = getattr(tokenizer, 'eos_token_id', None)

    tokens_after = []
    steering_steps = []
    steering_specs = []

    for step in range(n_steps + 1):
        # Activate steering after start_step
        if step >= start_step:
            spec = SteeringSpec(
                layer=steering_layer,
                plane_vectors=plane_vectors,
                magnitude=magnitude,
                active=True,
            )
            hook.set_spec(spec)
            steering_steps.append(step)
            steering_specs.append(spec)
        else:
            hook.clear()

        with torch.no_grad():
            if step == 0:
                outputs = hf_model(
                    input_ids,
                    output_hidden_states=False,
                    use_cache=True,
                )
            else:
                outputs = hf_model(
                    next_token_id,
                    past_key_values=past_key_values,
                    output_hidden_states=False,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :].float()

            if temperature <= 0:
                next_id = logits.argmax()
            else:
                probs = torch.softmax(logits / temperature, dim=0)
                next_id = torch.multinomial(probs, 1)[0]

            next_token_id = next_id.unsqueeze(0).unsqueeze(0)

            if step > 0:
                tok_str = tokenizer.decode([next_id.item()])
                tokens_after.append(tok_str)

                if eos_id is not None and next_id.item() == eos_id:
                    break

    hook.remove()

    text_after = "".join(tokens_after)
    tokens_before = baseline_tokens if baseline_tokens is not None else []
    text_before = "".join(tokens_before) if tokens_before else ""

    return SteeringResult(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        text_before=text_before,
        text_after=text_after,
        steering_steps=steering_steps,
        steering_specs=steering_specs,
    )
