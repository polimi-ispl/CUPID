"""Video-level centroid-anchored analogy maps from the paper's full pair sets.

All tokens include CLS. Decoder outputs stay in normalized training-image units:
average pairs and RGB first, subtract the reference baseline, then take abs.
"""

from numbers import Integral

import numpy as np
import torch
from PIL import Image


def require_reference_tokens(reference_set):
    """Return saved full tokens, rejecting CLS-only sets before video inference."""
    tokens = reference_set.get("tokens")
    if tokens is None:
        raise ValueError(
            "Heatmaps require full reference tokens; this reference set contains "
            "only CLS features. Re-extract it with `cupid extract-reference --heatmap` "
            "or use extract_reference_set(..., include_tokens=True)."
        )
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or tokens.shape[0] < 2
        or tokens.shape[1] < 2
        or tokens.shape[2] < 1
        or not tokens.is_floating_point()
    ):
        raise ValueError(
            "Heatmaps require floating-point reference tokens [N >= 2, T, C], "
            "including CLS and all patches. Extract at least two reference frames "
            "using --frames 2 (or more) or additional reference videos."
        )
    return tokens


def _mean_decoded(decoder, references, targets, centroid, batch_size, reference_pairs):
    """Enumerate flattened pairs in bounded batches; never build the pair grid."""
    target_count = targets.shape[0]
    row_size = target_count - 1 if reference_pairs else target_count
    pair_count = references.shape[0] * row_size
    total = None
    for start in range(0, pair_count, batch_size):
        indices = torch.arange(
            start, min(start + batch_size, pair_count), device=references.device
        )
        source = indices // row_size
        target = indices % row_size
        if reference_pairs:
            # Each row contains every j except i, including both pair directions.
            target = target + (target >= source)
        analogies = targets[target] - references[source]
        analogies.add_(centroid)
        decoded = decoder.forward_no_masking(analogies.permute(1, 0, 2))
        batch_sum = decoded.sum(dim=(0, 1), dtype=torch.float32)
        if total is None:
            total = batch_sum
        else:
            total.add_(batch_sum)
    return total.div_(pair_count * 3)


@torch.inference_mode()
def compute_heatmap(decoder, reference_tokens, test_tokens, batch_size=64):
    """Return one float32 numpy [H, W] map over all sampled test frames.

    Args:
        decoder: CUPID MAE decoder in evaluation mode, with forward_no_masking.
        reference_tokens: floating-point [N >= 2, T, C], full last-layer tokens.
        test_tokens: floating-point [M >= 1, T, C], full last-layer tokens.
        batch_size: maximum number of analogies decoded at once.

    The reference baseline uses all ordered i != j pairs; the test mean uses
    every reference/test pair. No pair sampling, input reconstruction errors,
    RGB denormalization, clipping, or per-frame absolute residuals are used.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    reference_tokens = require_reference_tokens({"tokens": reference_tokens})
    position = decoder.pos_embedding
    token_shape = (position.shape[0], position.shape[2])
    if tuple(reference_tokens.shape[1:]) != token_shape:
        raise ValueError(
            "Reference tokens must include CLS and all patches: "
            f"[N, {token_shape[0]}, {token_shape[1]}]"
        )
    if (
        not isinstance(test_tokens, torch.Tensor)
        or test_tokens.ndim != 3
        or test_tokens.shape[0] < 1
        or tuple(test_tokens.shape[1:]) != token_shape
        or not test_tokens.is_floating_point()
    ):
        raise ValueError(
            "Test tokens must be floating-point "
            f"[M >= 1, {token_shape[0]}, {token_shape[1]}], including CLS"
        )
    references = reference_tokens.to(device=position.device, dtype=position.dtype)
    targets = test_tokens.to(device=position.device, dtype=position.dtype)
    centroid = references.mean(dim=0)
    baseline = _mean_decoded(decoder, references, references, centroid, batch_size, True)
    test_mean = _mean_decoded(decoder, references, targets, centroid, batch_size, False)
    return test_mean.sub_(baseline).abs_().cpu().numpy()


def save_heatmap(heatmap, path):
    """Save only the colored interpretability map at its native UV resolution.

    The numeric map is not changed. Colors span zero to this map's maximum;
    colors alone must not be compared across separately scaled video PNGs.
    """
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("heatmap must be a nonempty [H, W] array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("heatmap must contain finite, nonnegative values")
    maximum = float(values.max())
    scaled = values / maximum if maximum > 0 else np.zeros_like(values)
    # A sequential dark-purple -> teal -> yellow palette.
    anchors = np.array(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float32,
    )

    stops = np.linspace(0, 1, len(anchors))
    colored = np.stack(
        [np.interp(scaled, stops, anchors[:, c]) for c in range(3)],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(colored).save(path, format="PNG")
