"""Per-frame CUPID centroid-anchored analogy maps, not reconstruction errors.

Decoder outputs remain in normalized training-image units.
"""

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import torch


class IdentityHeatmap:
    """Decode each analogy separately, then average its pixels and RGB channels.

    ``reference_tokens`` has shape [N, T, C]; ``decode_frame`` accepts [T, 1, C].
    Both contain all unmasked last-layer tokens, including CLS. References and
    the once-sampled baseline pairs are fixed for this instance's lifetime.
    The supplied model is placed in evaluation mode and must not be moved or
    modified while the instance is in use.
    """

    @torch.inference_mode()
    def __init__(
        self,
        model,
        reference_tokens: torch.Tensor,
        batch_size: int = 64,
        seed: int = 0,
        max_pairs: int = 5000,
    ):
        for name, value in (("batch_size", batch_size), ("max_pairs", max_pairs)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        if not -(2**63) <= seed < 2**64:
            raise ValueError("seed is outside torch.Generator.manual_seed's range")
        if not isinstance(reference_tokens, torch.Tensor):
            raise TypeError("reference_tokens must be a torch.Tensor [N, T, C]")
        if (
            reference_tokens.ndim != 3
            or reference_tokens.shape[0] < 2
            or reference_tokens.shape[1] < 2
            or reference_tokens.shape[2] < 1
        ):
            raise ValueError("reference_tokens must have shape [N >= 2, T >= 2, C >= 1]")
        if not reference_tokens.is_floating_point():
            raise TypeError("reference_tokens must be floating point")

        # The decoder positional embedding includes CLS and defines the complete
        # token contract; rejecting a CLS-only/patch-only input avoids broadcasting.
        position = model.decoder.pos_embedding
        self._token_shape = (position.shape[0], position.shape[2])
        if tuple(reference_tokens.shape[1:]) != self._token_shape:
            raise ValueError(
                f"reference tokens must include all decoder tokens (including CLS): "
                f"expected [N, {self._token_shape[0]}, {self._token_shape[1]}]"
            )
        self.model = model.eval()
        self.batch_size = int(batch_size)
        self._references = reference_tokens.detach().to(
            device=position.device, dtype=position.dtype, copy=True
        )
        if not torch.isfinite(self._references).all().item():
            raise ValueError("reference_tokens must be finite in the decoder dtype")
        self._centroid = self._references.mean(dim=0)
        count = self._references.shape[0]

        # Match the research figure's enumeration and CPU randperm, but isolate
        # its RNG and sample only once, never anew for a test frame or class.
        pairs = torch.triu_indices(count, count, offset=1, device="cpu")
        available_pairs = pairs.shape[1]
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if available_pairs > max_pairs:
            selected = torch.randperm(available_pairs, generator=generator, device="cpu")
            pairs = pairs[:, selected[:max_pairs]]
        pair_count = pairs.shape[1]
        self.metadata = {
            "method": "centroid_anchored_analogy_original",
            "tokens": "all unmasked encoder last-layer tokens, including CLS",
            "reference_count": count,
            "reference_pair_convention": "upper triangular, i < j; displacement R_j - R_i",
            "reference_pair_candidates": available_pairs,
            "reference_pair_count": pair_count,
            "max_pairs": int(max_pairs),
            "seed": int(seed),
            "reference_pair_sampling": "CPU torch.randperm without replacement, once at initialization",
            "baseline": "mean_pairs,RGB D(centroid + R_j - R_i), computed once",
            "test_frame": "mean_all_references,RGB D(centroid + T_frame - R_i)",
            "decoder_units": "normalized output, no RGB denormalization or clamping",
            "aggregation": "decode every pair before averaging; no decode-of-mean approximation",
            "residual": "abs(centered truncated-window mean of signed decoded frame maps - baseline)",
            "batch_size": self.batch_size,
        }
        pairs = pairs.to(position.device)
        self.baseline = self._mean_decoded(
            self._centroid.unsqueeze(0)
            + self._references[pairs[1, start : start + self.batch_size]]
            - self._references[pairs[0, start : start + self.batch_size]]
            for start in range(0, pair_count, self.batch_size)
        )

    def _mean_decoded(self, batches: Iterable[torch.Tensor]) -> np.ndarray:
        """Stream [B, T, C] analogies without retaining decoded RGB batches."""
        total = None
        pair_count = 0
        for batch in batches:
            decoded = self.model.decoder.forward_no_masking(batch.permute(1, 0, 2))
            if (
                decoded.ndim != 4
                or decoded.shape[0] != batch.shape[0]
                or decoded.shape[1] != 3
                or min(decoded.shape[2:]) < 1
            ):
                raise ValueError("decoder must return [batch, 3, H, W] images")
            batch_sum = decoded.sum(dim=(0, 1), dtype=torch.float32)
            if total is None:
                total = batch_sum
            else:
                if total.shape != batch_sum.shape:
                    raise ValueError("decoder image dimensions changed between batches")
                total.add_(batch_sum)
            pair_count += batch.shape[0]
        if total is None:
            raise ValueError("at least one analogy must be decoded")
        total.div_(pair_count * 3)
        if not torch.isfinite(total).all().item():
            raise ValueError("decoder produced nonfinite mean pixels")
        return total.cpu().numpy()

    @torch.inference_mode()
    def decode_frame(self, tokens: torch.Tensor) -> np.ndarray:
        """Return a signed [H, W] decoded mean, BEFORE subtraction and abs."""
        expected = (self._token_shape[0], 1, self._token_shape[1])
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor [T, 1, C]")
        if tuple(tokens.shape) != expected:
            raise ValueError(f"frame tokens must have shape {expected}, including CLS")
        if not tokens.is_floating_point():
            raise TypeError("frame tokens must be floating point")
        frame = tokens.detach().to(self._references).squeeze(1)
        if not torch.isfinite(frame).all().item():
            raise ValueError("frame tokens must be finite in the decoder dtype")
        anchored_frame = self._centroid + frame
        result = self._mean_decoded(
            anchored_frame.unsqueeze(0) - self._references[start : start + self.batch_size]
            for start in range(0, self._references.shape[0], self.batch_size)
        )
        if result.shape != self.baseline.shape:
            raise ValueError("decoded frame dimensions differ from the reference baseline")
        return result


def temporal_residuals(
    decoded_maps: np.ndarray, baseline: np.ndarray, window: int = 5
) -> np.ndarray:
    """Average signed decoded maps in a centered valid window, then subtract/abs.

    At endpoints, only available frames contribute (no padding or replication).
    ``window=1`` gives the unsmoothed per-frame residual. Inputs are not modified;
    output is float32 [F, H, W]. An empty sequence returns the same empty shape.
    """
    if isinstance(window, bool) or not isinstance(window, Integral) or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    maps = np.asarray(decoded_maps)
    reference = np.asarray(baseline)
    if maps.ndim != 3 or reference.ndim != 2 or maps.shape[1:] != reference.shape:
        raise ValueError("decoded_maps must be [F, H, W] and baseline must match [H, W]")
    if min(reference.shape) < 1:
        raise ValueError("baseline spatial dimensions must be nonempty")
    for name, array in (("decoded_maps", maps), ("baseline", reference)):
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError(f"{name} must contain real numeric values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

    # A rolling float64 sum avoids allocating a full-sequence prefix array and
    # limits cancellation while retaining signed quantities until the final abs.
    result = np.empty(maps.shape, dtype=np.float32)
    if not len(maps):
        return result
    radius = int(window) // 2
    left, right = 0, min(len(maps), radius + 1)
    total = maps[left:right].sum(axis=0, dtype=np.float64)
    reference = reference.astype(np.float64, copy=False)
    for frame in range(len(maps)):
        next_left = max(0, frame - radius)
        next_right = min(len(maps), frame + radius + 1)
        while left < next_left:
            total -= maps[left]
            left += 1
        while right < next_right:
            total += maps[right]
            right += 1
        result[frame] = np.abs(total / (right - left) - reference)
    if not np.isfinite(result).all():
        raise ValueError("temporal residuals exceed the finite float32 output range")
    return result
