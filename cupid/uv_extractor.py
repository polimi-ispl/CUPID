"""Batched GPU UV-texture extraction.

Pipeline per frame batch: RetinaFace face detection -> 106-point landmark
refinement -> POS alignment to a canonical 224x224 crop -> 3DMM fitting and
UV-space rendering (3DDFA_V3). All stages run on GPU.
"""

import re

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from cupid.tddfa_v3.face_box.retinaface.network import RetinaFace
from cupid.tddfa_v3.face_box.facelandmark.nets.large_base_lmks_net import LargeBaseLmksNet
from cupid.tddfa_v3.face_box.retinaface.prior_box import priorbox
from cupid.tddfa_v3.util.preprocess import load_lm3d
from cupid.tddfa_v3.model.recon_uv import uv_face_model

from cupid import weights as cupid_weights


def batch_resize_normalize_pad(images, max_size):
    """
    Pure PyTorch implementation of A.LongestMaxSize + A.Normalize + Pad

    Args:
        images: Tensor of shape (B, C, H, W) in range [0, 255]
        max_size: Maximum size for the longest dimension and target size for padding

    Returns:
        - images: Transformed tensor of shape (B, C, max_size, max_size)
        - pads: Tensor of shape (4,) with (left_pad, right_pad, top_pad, bottom_pad)
    """
    batch_size, channels, height, width = images.shape
    assert channels == 3, "Input images must have 3 channels (RGB)"
    assert batch_size > 0, "Batch size must be greater than 0"

    # Convert to float and normalize to [0, 1]
    images = images.float() / 255.0

    # Calculate scale factor (equivalent to LongestMaxSize)
    scale_factor = max_size / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Resize using bilinear interpolation
    images = F.interpolate(
        images, size=(new_height, new_width), mode="bilinear", align_corners=False
    )

    # Calculate padding
    target_height, target_width = max_size, max_size
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    if pad_height == 0 and pad_width == 0:
        pads = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=images.device)
    else:
        pads = (left_pad, right_pad, top_pad, bottom_pad)  # (left, right, top, bottom)
        images = F.pad(images, pads, value=0.0)  # Pad with zeros (black)
        pads = torch.tensor(
            pads, dtype=torch.int32, device=images.device
        )  # Convert pads to tensor

    # ImageNet normalization (equivalent to A.Normalize())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = (images - mean) / std

    return images, pads


def _load_retinaface_state_dict(retina_model_path):
    """Load the RetinaFace checkpoint, unwrapping and renaming keys if needed."""
    checkpoint = torch.load(retina_model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {re.sub("model.", "", k): v for k, v in state_dict.items()}


class FastUVExtractor:
    def __init__(
        self,
        retina_model_path: str | None = None,
        landmark_model_path: str | None = None,
        mm_model_path: str | None = None,
        rec_model_path: str | None = None,
        input_size: int = 224,
        max_size: int = 512,
        device: str = "cuda",
        weights_dir: str | None = None,
    ):
        """Initialize the FastUVExtractor class.

        Model paths left as None are resolved through cupid.weights (local
        weights_dir / CUPID_WEIGHTS_DIR override, else HuggingFace download).
        """

        if retina_model_path is None:
            retina_model_path = cupid_weights.get_tddfa_asset(
                cupid_weights.RETINAFACE_FILE, weights_dir
            )
        if landmark_model_path is None:
            landmark_model_path = cupid_weights.get_tddfa_asset(
                cupid_weights.LANDMARK_FILE, weights_dir
            )
        if mm_model_path is None:
            mm_model_path = cupid_weights.get_tddfa_asset(
                cupid_weights.FACE_MODEL_FILE, weights_dir
            )
        if rec_model_path is None:
            rec_model_path = cupid_weights.get_tddfa_asset(
                cupid_weights.NET_RECON_FILE, weights_dir
            )

        self.device = device
        self.input_size = input_size
        self.max_size = max_size

        ### Retinaface parameters
        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(max_size, max_size),
        ).to(device)
        self.variance = [0.1, 0.2]
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.4
        self.top_k = 10

        # Landmark parameters
        self.enlarge_ratio = 1.35
        self.selected_landmarks = [74, 83, 54, 84, 90]

        ### Load retinaface detection model
        self.retinaface = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        ).to(device)
        self.retinaface.load_state_dict(_load_retinaface_state_dict(retina_model_path))
        self.retinaface.eval()

        ### Load landmark model
        self.landmark_model = LargeBaseLmksNet(infer=False).to(device)
        checkpoint = torch.load(
            landmark_model_path, weights_only=False, map_location=device
        )
        self.landmark_model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()},
            strict=False,
        )
        self.landmark_model.eval()
        self.landmarks_3d = torch.tensor(load_lm3d()).to(device)  # Load 3D landmarks

        ### Load UV face model
        self.model_uv = uv_face_model(
            device=device,
            texture_size=input_size,
            model_path=mm_model_path,
            recon_path=rec_model_path,
        )

    def process_retina_annotations(
        self,
        locs,
        lands,
        scores_batch,
        pads,
        original_height,
        original_width,
    ):
        """
        Complete batch face detection pipeline from raw predictions to final coordinates

        Args:
            locs: Raw location predictions from RetinaFace
            lands: Raw landmark predictions from RetinaFace
            scores_batch: Confidence scores [batch_size, num_priors]
            pads: Padding values [left, right, top, bottom]
            original_height: Original image height
            original_width: Original image width

        Returns:
            frame_boxes: [batch_size, 4] - final boxes in original coordinates
            frame_scores: [batch_size] - confidence scores
            frame_landmarks: [batch_size, 5, 2] - landmarks in original coordinates
            frame_valid: [batch_size] - boolean mask for valid detections
        """
        actual_batch_size = scores_batch.shape[0]

        # Decode coordinates: reverse training transformation + scale back to resized image size
        boxes_batch = (
            self.decode_batch(locs, self.prior_box, self.variance) * self.max_size
        )  # From [0,1] to [0,512]
        landmarks_batch = (
            self.decode_landm_batch(lands, self.prior_box, self.variance)
            * self.max_size
        )  # From [0,1] to [0,512]

        # Get top-k detections per image to limit NMS computation
        # topk is faster than thresholding first then sorting, plus homogeneous tensor shapes
        topk_scores, topk_indices = torch.topk(
            scores_batch, min(self.top_k, scores_batch.shape[1]), dim=1
        )

        # Gather corresponding boxes and landmarks
        # For each position [i, j]:
        #     - batch_indices[i, j] tells us which batch item (image)
        #     - topk_indices[i, j] tells us which detection within that image
        #     - boxes_batch[batch_indices[i, j], topk_indices[i, j]] gets the corresponding box
        batch_indices = (
            torch.arange(actual_batch_size, device=self.device)
            .unsqueeze(1)
            .expand(-1, topk_indices.shape[1])
        )
        topk_boxes = boxes_batch[batch_indices, topk_indices]
        topk_landmarks = landmarks_batch[batch_indices, topk_indices]

        # Flatten for batched_nms
        flat_boxes = topk_boxes.reshape(-1, 4)
        flat_scores = topk_scores.reshape(-1)
        flat_landmarks = topk_landmarks.reshape(-1, 10)

        # Create batch IDs
        batch_ids = torch.arange(
            actual_batch_size, device=self.device
        ).repeat_interleave(topk_scores.shape[1])

        # Apply batched NMS - it will naturally filter low scores
        keep = batched_nms(flat_boxes, flat_scores, batch_ids, self.nms_threshold)

        # Filter by confidence threshold after NMS (more efficient)
        confidence_keep = flat_scores[keep] > self.confidence_threshold
        final_keep = keep[confidence_keep]

        final_boxes = flat_boxes[final_keep]
        final_scores = flat_scores[final_keep]
        final_landmarks = flat_landmarks[final_keep]
        final_batch_ids = batch_ids[final_keep]

        # No guarantee of one detection per frame: keep the highest scoring detection per frame
        # final_batch_ids is sorted by highest score due to NMS
        frame_boxes = torch.zeros(actual_batch_size, 4, device=self.device)
        frame_scores = torch.zeros(actual_batch_size, device=self.device)
        frame_landmarks = torch.zeros(actual_batch_size, 10, device=self.device)
        frame_valid = torch.zeros(
            actual_batch_size, dtype=torch.bool, device=self.device
        )

        if len(final_batch_ids) > 0:
            # Get unique batch IDs and their first occurrence indices
            unique_batch_ids = torch.unique(final_batch_ids, sorted=True)

            # For each unique batch ID, find its first occurrence
            for i, batch_id in enumerate(unique_batch_ids):
                first_occurrence = (final_batch_ids == batch_id).nonzero(as_tuple=True)[
                    0
                ][0]
                frame_boxes[batch_id] = final_boxes[first_occurrence]
                frame_scores[batch_id] = final_scores[first_occurrence]
                frame_landmarks[batch_id] = final_landmarks[first_occurrence]
                frame_valid[batch_id] = True

        # Reshape landmarks to [batch_size, 5, 2]
        frame_landmarks = frame_landmarks.reshape(-1, 5, 2)

        # Calculate scale factor
        left_pad, right_pad, top_pad, bottom_pad = pads
        scale_factor = max(original_height, original_width) / self.max_size

        # Adjust boxes and landmarks back to original image coordinates
        if pads.sum() > 0:
            # Adjust boxes
            frame_boxes[:, 0] -= left_pad  # x1
            frame_boxes[:, 1] -= top_pad  # y1
            frame_boxes[:, 2] -= left_pad  # x2
            frame_boxes[:, 3] -= top_pad  # y2

            # Adjust landmarks
            frame_landmarks[:, :, 0] -= left_pad  # x coordinates
            frame_landmarks[:, :, 1] -= top_pad  # y coordinates

        frame_landmarks *= scale_factor
        frame_boxes *= scale_factor

        # Clamp boxes and landmarks to original image boundaries
        frame_boxes[:, [0, 2]] = frame_boxes[:, [0, 2]].clamp(0, original_width - 1)
        frame_boxes[:, [1, 3]] = frame_boxes[:, [1, 3]].clamp(0, original_height - 1)
        frame_landmarks[:, :, 0] = frame_landmarks[:, :, 0].clamp(0, original_width - 1)
        frame_landmarks[:, :, 1] = frame_landmarks[:, :, 1].clamp(
            0, original_height - 1
        )

        return frame_boxes, frame_scores, frame_landmarks, frame_valid

    def decode_batch(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
        Args:
            loc: location predictions for loc layers,
                Shape: [batch_size, num_priors, 4]
            priors: Prior boxes in center-offset form.
                Shape: [num_priors, 4] - will be broadcasted across batch
            variances: Variances of priorboxes
        Return:
            decoded bounding box predictions, Shape: [batch_size, num_priors, 4]
        """
        # Handle broadcasting: priors [num_priors, 4] -> [1, num_priors, 4]
        if priors.dim() == 2:
            priors = priors.unsqueeze(0)  # [10752, 4] -> [1, 10752, 4]

        # PyTorch will automatically broadcast [1, 10752, 4] to [batch_size, 10752, 4]
        boxes = torch.cat(
            (
                priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
                priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
            ),
            dim=2,
        )
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2  # Convert center to top-left
        boxes[:, :, 2:] += boxes[:, :, :2]  # Convert width/height to bottom-right
        return boxes

    def decode_landm_batch(self, pre, priors, variances):
        """Decode landmarks from predictions using priors to undo the encoding we did for offset regression at train time.
        Args:
            pre: landmark predictions for loc layers,
                Shape: [batch_size, num_priors, 10]
            priors: Prior boxes in center-offset form.
                Shape: [num_priors, 4] or [batch_size, num_priors, 4] (broadcasted)
            variances: Variances of priorboxes
        Return:
            decoded landmark predictions, Shape: [batch_size, num_priors, 10]
        """
        # Handle broadcasting: priors [num_priors, 4] -> [1, num_priors, 4]
        if priors.dim() == 2:
            priors = priors.unsqueeze(0)  # [num_priors, 4] -> [1, num_priors, 4]

        return torch.cat(
            (
                priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:],
            ),
            dim=2,  # Concatenate along the landmark coordinate dimension
        )

    def batch_POS_gpu(self, landmarks_2d):
        """
        Batch GPU version of Procrustes analysis for 2D-3D landmark alignment.

        Args:
            landmarks_2d: torch.Tensor [batch_size, num_landmarks, 2] - 2D facial landmarks from images

        Returns:
            translation: torch.Tensor [batch_size, 2] - translation parameters (tx, ty)
            scale: torch.Tensor [batch_size] - scale factors
        """
        batch_size = landmarks_2d.shape[0]
        num_landmarks = landmarks_2d.shape[1]  # Should be 5
        device = landmarks_2d.device

        # Create coefficient matrix for least squares problem Ax = b
        # Each landmark contributes 2 equations (x and y), so 10 equations total
        # 8 unknowns: [r1_x, r1_y, r1_z, scale_tx, r2_x, r2_y, r2_z, scale_ty]
        coefficient_matrix = torch.zeros(batch_size, 2*num_landmarks, 8, device=device)  # [B, 10, 8]

        # Expand 3D landmarks to match batch size
        landmarks_3d_batch = self.landmarks_3d.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 5, 3]

        # Fill coefficient matrix:
        # Even rows (0,2,4,6,8): equations for x-coordinates
        coefficient_matrix[:, 0::2, 0:3] = landmarks_3d_batch  # 3D coords for rotation vector 1
        coefficient_matrix[:, 0::2, 3] = 1                     # constant term for translation

        # Odd rows (1,3,5,7,9): equations for y-coordinates
        coefficient_matrix[:, 1::2, 4:7] = landmarks_3d_batch  # 3D coords for rotation vector 2
        coefficient_matrix[:, 1::2, 7] = 1                     # constant term for translation

        # Create target vector from 2D landmarks: reshape [B,5,2] -> [B,10,1]
        target_vector = landmarks_2d.reshape(batch_size, 2*num_landmarks, 1)  # [B, 10, 1]

        # Solve least squares problem: coefficient_matrix * solution = target_vector
        solution = torch.linalg.lstsq(coefficient_matrix, target_vector).solution  # [B, 8, 1]
        solution = solution.squeeze(-1)  # [B, 8]

        # Extract transformation parameters from solution
        rotation_vector_1 = solution[:, 0:3]   # First rotation vector [B, 3]
        rotation_vector_2 = solution[:, 4:7]   # Second rotation vector [B, 3]
        scaled_translation_x = solution[:, 3]  # Scale * translation_x [B]
        scaled_translation_y = solution[:, 7]  # Scale * translation_y [B]

        # Calculate scale factor as average of rotation vector magnitudes
        scale = (torch.norm(rotation_vector_1, dim=1) + torch.norm(rotation_vector_2, dim=1)) / 2  # [B]

        # Combine translation components
        translation = torch.stack([scaled_translation_x, scaled_translation_y], dim=1)  # [B, 2]

        return translation, scale

    def batch_resize_n_crop_gpu(self, images, translation_params, scale_factors, output_size=224):
        """
        Batch GPU implementation of face alignment: resize and crop faces to standard size.

        Args:
            images: torch.Tensor [batch_size, C, H, W] - input images in range [0, 255]
            translation_params: torch.Tensor [batch_size, 2] - translation (tx, ty) from POS
            scale_factors: torch.Tensor [batch_size] - scale factors from POS (already computed as 102./scale)
            output_size: int - target crop size (default: 224x224)

        Returns:
            aligned_faces: torch.Tensor [batch_size, C, output_size, output_size] - aligned face crops
        """
        batch_size, num_channels, image_height, image_width = images.shape
        scaled_widths = (image_width * scale_factors).type(torch.int32)
        scaled_heights = (image_height * scale_factors).type(torch.int32)

        # Calculate crop coordinates
        # Center crop adjusted by translation parameters
        left_crop = ((scaled_widths - output_size) / 2 + (translation_params[:, 0] - image_width / 2) * scale_factors).type(torch.int32)
        top_crop = ((scaled_heights - output_size) / 2 + (image_height / 2 - translation_params[:, 1]) * scale_factors).type(torch.int32)
        right_crop = left_crop + output_size
        bottom_crop = top_crop + output_size

        # Calculate padding for all images in parallel
        pad_left = torch.clamp(-left_crop, min=0)
        pad_right = torch.clamp(right_crop - scaled_widths, min=0)
        pad_top = torch.clamp(-top_crop, min=0)
        pad_bottom = torch.clamp(bottom_crop - scaled_heights, min=0)

        # Clamp crop coordinates for all images in parallel
        cropped_left = torch.clamp(left_crop, min=0)
        cropped_right = torch.minimum(scaled_widths, right_crop)
        cropped_top = torch.clamp(top_crop, min=0)
        cropped_bottom = torch.minimum(scaled_heights, bottom_crop)

        # Convert images to float for processing
        images_float = images.float()

        # Process each image in the batch
        aligned_face_crops = []

        for batch_idx in range(batch_size):
            # Resize the image
            face_image = images_float[batch_idx]  # [C, H, W]
            try:
                face_image_resized = F.interpolate(
                    face_image.unsqueeze(0),  # Add batch dimension: [1, C, H, W]
                    size=(scaled_heights[batch_idx].item(), scaled_widths[batch_idx].item()),              # New size after scaling
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dimension: [C, H, W]
            except Exception as e:
                print(f"Error in interpolation for batch index {batch_idx}: {e}")
                # Create a dummy crop in case of error
                aligned_face_crops.append(torch.zeros(num_channels, output_size, output_size, device=self.device))
                continue

            # Crop the face region using precomputed coordinates
            face_region = face_image_resized[:,
                                        cropped_top[batch_idx]:cropped_bottom[batch_idx],
                                        cropped_left[batch_idx]:cropped_right[batch_idx]]

            # Apply padding using precomputed padding values
            if pad_left[batch_idx] > 0 or pad_right[batch_idx] > 0 or pad_top[batch_idx] > 0 or pad_bottom[batch_idx] > 0:
                face_region = F.pad(face_region,
                                (pad_left[batch_idx].item(), pad_right[batch_idx].item(),
                                pad_top[batch_idx].item(), pad_bottom[batch_idx].item()),
                                mode='constant', value=0.0)

            aligned_face_crops.append(face_region)

        # Stack all aligned faces into a batch tensor
        aligned_faces = torch.stack(aligned_face_crops, dim=0)  # [batch_size, C, output_size, output_size]

        return aligned_faces


class FastestUVExtractor(FastUVExtractor):
    """Batched inference using pre-allocated tensors and cached normalization constants."""

    def __init__(
        self,
        retina_model_path: str | None = None,
        landmark_model_path: str | None = None,
        mm_model_path: str | None = None,
        rec_model_path: str | None = None,
        input_size: int = 224,
        max_size: int = 512,
        device: str = "cuda",
        weights_dir: str | None = None,
    ):
        """Initialize the FastestUVExtractor."""

        # Initialize parent class
        super().__init__(
            retina_model_path=retina_model_path,
            landmark_model_path=landmark_model_path,
            mm_model_path=mm_model_path,
            rec_model_path=rec_model_path,
            input_size=input_size,
            max_size=max_size,
            device=device,
            weights_dir=weights_dir,
        )

        # Cache frequently used tensors to avoid repeated creation
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.landmark_mean = torch.tensor([103.94, 116.78, 123.68]).view(1, 3, 1, 1).to(device)

    def extract_aligned_faces(self, original_batch, transformed_batch, pads, original_height, original_width):
        """Optimized version with inference_mode."""

        with torch.inference_mode():
            locs, confs, lands = self.retinaface(transformed_batch.to(self.device))

            conf_batch = F.softmax(confs, dim=-1)
            scores_batch = conf_batch[:, :, 1]

            # Extract boxes and landmarks with retinaface
            frame_boxes, _, _, frame_valid = (
                self.process_retina_annotations(
                    locs=locs,
                    lands=lands,
                    scores_batch=scores_batch,
                    pads=pads,
                    original_height=original_height,
                    original_width=original_width,
                )
            )

            # Extract landmarks
            final_landmarks = self.extract_landmarks_optimized(
                video=original_batch,
                frame_boxes=frame_boxes,
                frame_valid=frame_valid,
            )

            # Align faces
            selected_landmarks_coords = final_landmarks[:, self.selected_landmarks, :]
            selected_landmarks_coords[:, :, -1] = original_height - 1 - selected_landmarks_coords[:, :, -1]
            translation_params, scale_factors_raw = self.batch_POS_gpu(selected_landmarks_coords)

            # Guard the fast path against unstable landmark fits that can produce
            # near-zero, negative, or non-finite scales and trigger enormous
            # interpolation sizes during alignment.
            min_scale_raw = torch.tensor(1.0, device=scale_factors_raw.device, dtype=scale_factors_raw.dtype)
            safe_scale_factors_raw = torch.where(
                torch.isfinite(scale_factors_raw) & (scale_factors_raw > 0),
                scale_factors_raw,
                min_scale_raw,
            )
            scale_factors = 102. / safe_scale_factors_raw
            scale_factors = torch.clamp(scale_factors, min=0.1, max=8.0)

            aligned_faces = self.batch_resize_n_crop_gpu(
                images=original_batch,
                translation_params=translation_params,
                scale_factors=scale_factors,
            )

        return aligned_faces

    def extract_landmarks_optimized(
        self,
        video,
        frame_boxes,
        frame_valid,
        num_iterations=2,
    ):
        """
        Optimized landmark extraction with vectorized operations.
        Key improvements:
        - Pre-allocated tensors instead of lists
        - Cached mean values
        """
        actual_batch_size = frame_boxes.shape[0]
        image_height, image_width = video.shape[-2:]

        # Initialize bounding box coordinates
        face_left = frame_boxes[:, 0]
        face_top = frame_boxes[:, 1]
        face_right = frame_boxes[:, 2]
        face_bottom = frame_boxes[:, 3]

        for iteration in range(num_iterations):
            # Calculate face dimensions and center points
            face_width = face_right - face_left + 1
            face_height = face_bottom - face_top + 1
            face_center_x = (face_right + face_left) / 2
            face_center_y = (face_bottom + face_top) / 2

            # Create enlarged square regions around faces
            enlarged_size = torch.max(face_height, face_width) * self.enlarge_ratio

            # Calculate new square crop coordinates
            crop_left = face_center_x - enlarged_size / 2
            crop_top = face_center_y - enlarged_size / 2
            crop_right = crop_left + enlarged_size
            crop_bottom = crop_top + enlarged_size

            # Store transformation offsets
            transform_offset_x = crop_left.clone()
            transform_offset_y = crop_top.clone()

            # Calculate padding
            left_padding = torch.clamp(-crop_left, min=0)
            top_padding = torch.clamp(-crop_top, min=0)
            right_padding = torch.clamp(crop_right - image_width, min=0)
            bottom_padding = torch.clamp(crop_bottom - image_height, min=0)

            # Clamp crop coordinates
            crop_left = torch.clamp(crop_left, min=0)
            crop_top = torch.clamp(crop_top, min=0)
            crop_right = torch.clamp(crop_right, max=image_width)
            crop_bottom = torch.clamp(crop_bottom, max=image_height)

            crop_coordinates = torch.stack(
                [crop_left, crop_top, crop_right, crop_bottom], dim=1
            ).int()
            padding_amounts = torch.stack(
                [left_padding, right_padding, top_padding, bottom_padding], dim=1
            ).int()

            # Pre-allocate output tensor instead of list (optimization)
            batched_face_crops = torch.zeros(
                actual_batch_size, 3, self.input_size, self.input_size,
                device=self.device, dtype=torch.float32
            )

            # Extract and process face crops - loop is still needed due to variable crop sizes
            for frame_idx in range(actual_batch_size):
                if not frame_valid[frame_idx]:
                    continue  # Already zeros

                x1, y1, x2, y2 = crop_coordinates[frame_idx]
                left_pad, right_pad, top_pad, bottom_pad = padding_amounts[frame_idx]

                face_region = video[frame_idx, :, y1:y2, x1:x2]

                if left_pad > 0 or top_pad > 0 or right_pad > 0 or bottom_pad > 0:
                    face_region = F.pad(
                        face_region,
                        (left_pad, right_pad, top_pad, bottom_pad),
                        mode="constant",
                        value=0.0,
                    )

                # Resize and store directly in pre-allocated tensor
                batched_face_crops[frame_idx] = F.interpolate(
                    face_region.unsqueeze(0).float(),
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Normalize using cached mean values
            batched_face_crops_norm = (batched_face_crops - self.landmark_mean) / 255.0

            # Landmark model inference
            base_lmks = self.landmark_model(batched_face_crops_norm) * self.input_size

            # Transform landmarks back to original coordinates
            inv_scale = enlarged_size / self.input_size
            base_lmks_reshaped = base_lmks.view(-1, 106, 2)

            inv_scale_expanded = inv_scale.view(-1, 1, 1).expand(-1, 106, 2)
            transform_offset = (
                torch.stack([transform_offset_x, transform_offset_y], dim=1)
                .view(-1, 1, 2)
                .expand(-1, 106, 2)
            )

            transformed_landmarks = (
                base_lmks_reshaped * inv_scale_expanded + transform_offset
            )

            # For next iteration: use landmark-based bounding boxes
            if iteration < num_iterations - 1:
                face_left = torch.min(transformed_landmarks[:, :, 0], dim=1)[0]
                face_top = torch.min(transformed_landmarks[:, :, 1], dim=1)[0]
                face_right = torch.max(transformed_landmarks[:, :, 0], dim=1)[0]
                face_bottom = torch.max(transformed_landmarks[:, :, 1], dim=1)[0]

        return transformed_landmarks

    def extract_uv(self, aligned_faces):
        """Optimized UV extraction with inference_mode."""
        with torch.inference_mode():
            return self.model_uv.forward(aligned_faces / 255.0)
