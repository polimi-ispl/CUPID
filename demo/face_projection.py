"""Extract aligned UV textures and project scalar UV maps onto original frames.

This is a geometric illustration, not a localization probability. Visibility is
3DMM self-occlusion (a z-buffer), not segmentation of hands, hair, or other objects
in front of the face. No error scaling or color normalization happens here.
"""

from dataclasses import dataclass
import math

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from cupid import weights
from cupid.mae import load_trained_model
from cupid.tddfa_v3.model.recon_uv import batched_bilinear_interpolate
from cupid.uv_extractor import FastestUVExtractor, batch_resize_normalize_pad


class FaceFitError(RuntimeError):
    """The frame has no usable detected/fitted face; do not invent an overlay."""


@dataclass(frozen=True)
class _CropTransform:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int
    size: int


class _ProjectionExtractor(FastestUVExtractor):
    """Keep the existing detector/alignment pipeline, recording its exact crop."""

    def process_retina_annotations(self, *args, **kwargs):
        result = super().process_retina_annotations(*args, **kwargs)
        boxes, _, _, valid = result
        if not bool(valid.all()):
            raise FaceFitError("No face passed the RetinaFace confidence threshold")
        if not bool(torch.isfinite(boxes).all()) or not bool(
            (boxes[:, 2:] > boxes[:, :2]).all()
        ):
            raise FaceFitError("Detected face has invalid bounds")
        return result

    def batch_POS_gpu(self, landmarks_2d):
        if not bool(torch.isfinite(landmarks_2d).all()):
            raise FaceFitError("Non-finite face landmarks")
        translation, raw_scale = super().batch_POS_gpu(landmarks_2d)
        if not bool(torch.isfinite(translation).all()) or not bool(
            (torch.isfinite(raw_scale) & (raw_scale > 0)).all()
        ):
            raise FaceFitError("Invalid landmark alignment fit")
        scale = 102.0 / raw_scale
        # The fast extractor clamps this range. Reject instead of accepting its
        # guarded fallback as evidence that an unstable alignment is meaningful.
        if not bool(((scale >= 0.1) & (scale <= 8.0)).all()):
            raise FaceFitError("Landmark fit exceeds the extractor's alignment range")
        return translation, raw_scale

    def batch_resize_n_crop_gpu(
        self, images, translation_params, scale_factors, output_size=224
    ):
        height, width = images.shape[-2:]
        # Match the parent's int32 truncation, including negative crop offsets.
        # The effective x/y scales differ after integer size rounding.
        resized_widths = (width * scale_factors).to(torch.int32)
        resized_heights = (height * scale_factors).to(torch.int32)
        left = (
            (resized_widths - output_size) / 2
            + (translation_params[:, 0] - width / 2) * scale_factors
        ).to(torch.int32)
        top = (
            (resized_heights - output_size) / 2
            + (height / 2 - translation_params[:, 1]) * scale_factors
        ).to(torch.int32)
        params = torch.stack((resized_widths, resized_heights, left, top), dim=1)
        self.alignments = [
            _CropTransform(width, height, rw, rh, x, y, output_size)
            for rw, rh, x, y in params.cpu().tolist()
        ]
        for crop in self.alignments:
            if (
                crop.resized_width <= 0 or crop.resized_height <= 0
                or crop.left >= crop.resized_width or crop.top >= crop.resized_height
                or crop.left + crop.size <= 0 or crop.top + crop.size <= 0
            ):
                raise FaceFitError("Aligned face crop does not intersect the image")
        aligned = super().batch_resize_n_crop_gpu(
            images, translation_params, scale_factors, output_size
        )
        if not bool(torch.isfinite(aligned).all()) or not bool(aligned.abs().any()):
            raise FaceFitError("Face alignment produced an empty or non-finite crop")
        return aligned


class FaceOverlayProjector:
    """Extract normalized UVs and retain each frame's exact fitted geometry.

    ``extract`` returns normalized UV input and geometry for later projection.
    ``project`` maps a scalar UV map through that geometry, returning original
    resolution float32 ``error`` and visible-coverage ``mask`` arrays.
    Keeping these stages separate allows temporal averaging in canonical UV
    coordinates before projection onto the current frame, without head-motion
    smearing. This class does not define or compute the interpretability map.
    Invalid fits raise FaceFitError; CUDA/model failures are not suppressed.
    """

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Face projection requires CUDA for nvdiffrast")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        self.extractor = _ProjectionExtractor(device=str(self.device))
        self.checkpoint_path = weights.get_cupid_checkpoint()
        self.model, _ = load_trained_model(self.checkpoint_path, str(self.device))
        self.triangles = self.extractor.model_uv.tri.to(torch.int32).contiguous()
        # MeshRenderer_UV flips y before rasterizing: raster row zero is NDC -1.
        # Thus UV sampling coordinates are (2u-1, 1-2v), not (2u-1, 2v-1).
        uv = self.extractor.model_uv.processed_uv_coords[:, :2].clone()
        uv[:, 1].neg_()
        self.uv_grid_vertices = uv.unsqueeze(0).contiguous()

    def _rasterize_uv(self, camera_vertices, image_vertices, uv_values):
        size = self.extractor.input_size
        # recon_uv samples the aligned image at (v2d.x, size-1-v2d.y).
        # Construct clip coordinates from those exact pixel centers; using
        # center=size/2 instead would introduce a half-pixel offset.
        xy = torch.stack(
            (image_vertices[..., 0], size - 1 - image_vertices[..., 1]), dim=-1
        )
        xy_ndc = (xy + 0.5) * (2.0 / size) - 1.0
        z = camera_vertices[..., 2:3]
        near, far = 0.1, 50.0
        if not bool(((z > near) & (z < far)).all()):
            raise FaceFitError("Fitted mesh lies outside the camera depth range")
        clip_z = z * ((far + near) / (far - near)) - 2 * far * near / (far - near)
        clip = torch.cat((xy_ndc * z, clip_z, z), dim=-1).contiguous()
        # Reuse the CUDA context already created by the UV render. Perspective
        # w=z gives correct UV interpolation; the nearest triangle wins per pixel.
        context = self.extractor.model_uv.uv_renderer.ctx
        raster, _ = dr.rasterize(
            context, clip, self.triangles, resolution=[size, size]
        )
        mask = (raster[..., 3] > 0).unsqueeze(1).float()
        if not bool(mask.any()):
            raise FaceFitError("Fitted mesh has no visible pixels in the aligned crop")
        uv_grid, _ = dr.interpolate(self.uv_grid_vertices, raster, self.triangles)
        values = F.grid_sample(
            uv_values, uv_grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )
        return values * mask, mask

    def _to_original(self, values, mask, crop):
        # Invert F.interpolate(..., align_corners=False) + integer crop:
        # aligned_x = (original_x + .5) * resized_width / width - .5 - left.
        # Restrict work to the crop's bilinear support, rather than sampling the
        # entire mostly-empty source frame on the GPU.
        sx = crop.resized_width / crop.width
        sy = crop.resized_height / crop.height
        x0 = max(0, math.floor((crop.left - 0.5) / sx - 0.5))
        x1 = min(crop.width, math.ceil((crop.left + crop.size + 0.5) / sx - 0.5) + 1)
        y0 = max(0, math.floor((crop.top - 0.5) / sy - 0.5))
        y1 = min(crop.height, math.ceil((crop.top + crop.size + 0.5) / sy - 0.5) + 1)
        x = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
        y = torch.arange(y0, y1, device=self.device, dtype=torch.float32)
        gx = ((x + 0.5) * sx - crop.left) * (2.0 / crop.size) - 1.0
        gy = ((y + 0.5) * sy - crop.top) * (2.0 / crop.size) - 1.0
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            torch.cat((values, mask), dim=1), grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )[0]
        coverage = warped[-1]
        if not bool((coverage > 0).any()):
            raise FaceFitError("Fitted mesh has no visible pixels in the original frame")
        # Unpremultiply coverage: a constant residual stays constant at the edge.
        values = torch.where(coverage > 0, warped[:-1] / coverage.clamp_min(1e-12), 0)
        full_values = np.zeros(
            (values.shape[0], crop.height, crop.width), dtype=np.float32
        )
        full_mask = np.zeros((crop.height, crop.width), dtype=np.float32)
        full_values[:, y0:y1, x0:x1] = values.cpu().numpy()
        full_mask[y0:y1, x0:x1] = coverage.cpu().numpy()
        return full_values, full_mask

    @torch.inference_mode()
    def extract(self, rgb: np.ndarray) -> tuple:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an HxWx3 uint8 RGB frame")
        if min(rgb.shape[:2]) <= 0:
            raise ValueError("Expected a nonempty RGB frame")
        frame = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        height, width = rgb.shape[:2]
        transformed, pads = batch_resize_normalize_pad(frame, self.extractor.max_size)
        aligned = self.extractor.extract_aligned_faces(
            frame, transformed, pads, height, width
        ) / 255.0
        geometry = self.extractor.model_uv
        alpha = geometry.net_recon(aligned)
        if not bool(torch.isfinite(alpha).all()):
            raise FaceFitError("3DMM regression returned non-finite coefficients")
        coeff = geometry.split_alpha(alpha)
        shape = geometry.compute_shape(coeff["id"], coeff["exp"])
        rotation = geometry.compute_rotation(coeff["angle"])
        camera = geometry.to_camera(geometry.transform(shape, rotation, coeff["trans"]))
        if not bool(torch.isfinite(camera).all()) or not bool((camera[..., 2] > 0).all()):
            raise FaceFitError("3DMM regression returned invalid camera geometry")
        projected = geometry.to_image(camera)
        # Reproduce recon_uv.forward with this same fitted mesh, avoiding a second
        # regression pass and retaining the exact image/UV coordinate conventions.
        colors = batched_bilinear_interpolate(
            aligned.permute(0, 2, 3, 1), projected[..., 0],
            geometry.texture_size - 1 - projected[..., 1],
        )
        uv_mask, _, uv_rgb, _ = geometry.uv_renderer(
            geometry.processed_uv_coords.unsqueeze(0), geometry.tri, colors
        )
        normalized = (uv_rgb - self.extractor.imagenet_mean) / self.extractor.imagenet_std
        return normalized, {
            "camera": camera,
            "projected": projected,
            "crop": self.extractor.alignments[0],
            "uv_mask": uv_mask[0, 0].bool().cpu().numpy(),
        }

    @torch.inference_mode()
    def project(self, uv_map: np.ndarray, geometry: dict) -> dict:
        size = self.extractor.input_size
        uv_map = np.asarray(uv_map, dtype=np.float32)
        if uv_map.shape != (size, size) or not np.isfinite(uv_map).all():
            raise ValueError(f"Expected a finite scalar UV map of shape {(size, size)}")
        uv_values = torch.from_numpy(np.ascontiguousarray(uv_map)).to(self.device)[None, None]
        values, mask = self._rasterize_uv(
            geometry["camera"], geometry["projected"], uv_values
        )
        values, mask = self._to_original(values, mask, geometry["crop"])
        return {"error": values[0], "mask": mask}
