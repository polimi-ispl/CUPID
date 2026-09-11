"""End-to-end CUPID inference: video -> UV textures -> features -> score."""

import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_indices

from cupid import __version__
from cupid import weights as cupid_weights
from cupid.heatmap import compute_heatmap, require_reference_tokens
from cupid.mae import load_trained_model
from cupid.uv_extractor import FastestUVExtractor, batch_resize_normalize_pad


def resolve_device(device="auto"):
    """Resolve the requested device string, requiring CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUPID requires an NVIDIA GPU: the UV renderer (nvdiffrast) is CUDA-only."
        )
    if device == "auto":
        return "cuda:0"
    return device


def normalize_uv(uv, mean, std):
    """Clamp UV textures to [0, 1] and apply ImageNet normalization."""
    return (uv.clamp(0, 1) - mean) / std


def max_cosine_similarity(ref_features, test_features):
    """Max pairwise cosine similarity between reference and test features.

    Args:
        ref_features: [N, D] reference feature set
        test_features: [M, D] test feature set

    Returns:
        Scalar tensor: the maximum cosine similarity over all N*M pairs.
    """
    ref_normalized = F.normalize(ref_features, p=2, dim=1)
    test_normalized = F.normalize(test_features, p=2, dim=1)
    similarities = torch.mm(ref_normalized, test_normalized.T)
    return torch.max(similarities)


class CupidPipeline:
    """Loads the UV extractor and MAE model once, then builds reference sets / scores videos.

    Args:
        device: "auto" or an explicit "cuda:N"
        frames_per_video: equispaced frames sampled across each video
        weights_dir: optional local folder containing all weight files
            (otherwise weights are auto-downloaded from the HuggingFace Hub)
    """

    def __init__(
        self,
        device="auto",
        frames_per_video=15,
        weights_dir=None,
    ):
        device = resolve_device(device)
        if device.startswith("cuda:"):
            # Prevent stray allocations on cuda:0 when another GPU is requested
            torch.cuda.set_device(int(device.split(":")[1]))
        self.device = device
        self.frames_per_video = frames_per_video

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        self.uv_extractor = FastestUVExtractor(
            device=device, weights_dir=weights_dir
        )

        checkpoint_path = cupid_weights.get_cupid_checkpoint(weights_dir)
        self.model, self.model_args = load_trained_model(checkpoint_path, device=device)

    def extract_video_features(self, video_path):
        """Decode a video, extract UV textures, and return CLS features.

        Args:
            video_path: path to a video file readable by FFmpeg

        Returns:
            features [frames_per_video, D]
        """
        features, _ = self._extract_video(video_path)
        return features

    def _extract_video(self, video_path, include_tokens=False):
        """Share frame sampling and UV preprocessing between scoring and maps."""
        decoder = VideoDecoder(video_path, device="cpu", num_ffmpeg_threads=0)
        clips = clips_at_regular_indices(
            decoder,
            num_clips=self.frames_per_video,
            num_frames_per_clip=1,
            policy="wrap",
        )
        video_frames = clips.data.squeeze(1).to(self.device)  # [N, C, H, W]

        with torch.inference_mode():
            original_height, original_width = video_frames.shape[2], video_frames.shape[3]
            transformed_batch, pads = batch_resize_normalize_pad(
                video_frames, max_size=self.uv_extractor.max_size
            )
            aligned_faces = self.uv_extractor.extract_aligned_faces(
                video_frames, transformed_batch, pads, original_height, original_width
            )
            uv = self.uv_extractor.extract_uv(aligned_faces)
            uv = normalize_uv(uv, self.norm_mean, self.norm_std)
            encoded = self.model.encoder.forward_no_masking(uv)
            features = encoded[0]
            tokens = encoded.permute(1, 0, 2) if include_tokens else None

        return features, tokens

    def extract_reference_set(self, reference_videos, include_tokens=False):
        """Extract a reference set from one or more reference videos of the POI.

        Returns a CPU-portable dict (the reference set) suitable for torch.save
        and for passing to score() via its "features" entry. With include_tokens,
        also stores full last-layer CLS + patch "tokens" [N, T, C] for interpret().
        """
        features = []
        tokens = []
        for path in reference_videos:
            if include_tokens:
                video_features, video_tokens = self._extract_video(path, include_tokens=True)
                tokens.append(video_tokens.float().cpu())
            else:
                video_features = self.extract_video_features(path)
            features.append(video_features.float().cpu())
        reference_set = {
            "features": torch.cat(features, dim=0),
            "frames_per_video": self.frames_per_video,
            "num_videos": len(reference_videos),
            "reference_names": [str(p) for p in reference_videos],
            "cupid_version": __version__,
        }
        if include_tokens:
            reference_set["tokens"] = torch.cat(tokens, dim=0)
        return reference_set

    def score(self, ref_features, test_video):
        """Score a test video against a reference feature set.

        Args:
            ref_features: [N, D] tensor (e.g. reference_set["features"])
            test_video: path to the test video

        Returns:
            score (float, max cosine similarity; higher = more likely genuine)
        """
        test_features = self.extract_video_features(test_video)
        return self._score_features(ref_features, test_features)

    def _score_features(self, ref_features, test_features):
        """Use the same similarity calculation for plain and interpreted scores."""
        ref_features = ref_features.to(device=self.device, dtype=test_features.dtype)
        with torch.inference_mode():
            score = max_cosine_similarity(ref_features, test_features)
        score = score.cpu().item()

        return score

    def interpret(self, reference_set, test_video):
        """Score a video and return one paper-style UV discrepancy map.

        reference_set must come from extract_reference_set(..., include_tokens=True)
        and contain at least two reference frames. Returns {"score": float,
        "heatmap": numpy.float32[H, W]}. All reference/test pairs and all ordered
        distinct reference pairs are decoded; pairs and RGB are averaged before
        subtracting the baseline and taking the absolute value.
        """
        reference_tokens = require_reference_tokens(reference_set)
        test_features, test_tokens = self._extract_video(test_video, include_tokens=True)
        score = self._score_features(reference_set["features"], test_features)
        heatmap = compute_heatmap(self.model.decoder, reference_tokens, test_tokens)
        return {"score": score, "heatmap": heatmap}
