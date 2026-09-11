"""Render the fixed Real 2 / Fake 2 CUPID comparison with temporary intermediates."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
import torch
from torchcodec.decoders import VideoDecoder

from cupid.pipeline import resolve_device
from face_projection import FaceFitError, FaceOverlayProjector
from identity_heatmap import IdentityHeatmap, temporal_residuals


DEMO_DIR = Path(__file__).resolve().parent
SOURCE_NAMES = ("test_real_2.mp4", "test_fake_2.mp4")
OFFSETS = (0.25, 1.0)
SECONDS = 4.5
FPS = 24.0
REFERENCE_FRAMES = 15
WINDOW = 5
BATCH_SIZE = 64
SEED = 0
MAX_PAIRS = 5000
SCALE = 1.4836496114730835
MAX_ALPHA = 0.78
ALPHA_EXPONENT = 1.2
VIRIDIS = np.asarray(colormaps["viridis"].colors, dtype=np.float32) * 255
LABELS = ("Real", "Fake")
LABEL_COLORS = ((45, 210, 92), (245, 63, 63))


def font(size):
    name = "DejaVuSans-Bold.ttf"
    for candidate in (name, f"/usr/share/fonts/TTF/{name}",
                      f"/usr/share/fonts/truetype/dejavu/{name}"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    raise RuntimeError("DejaVu Sans fonts are required for the Real/Fake labels")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "sha256": digest.hexdigest()}


def native_rgb(frame):
    rgb = frame.data.permute(1, 2, 0).contiguous().numpy()
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("VideoDecoder must produce native uint8 RGB frames")
    return rgb


def extract_frame(projector, rgb, context):
    try:
        return projector.extract(rgb)
    except FaceFitError as exc:
        raise FaceFitError(f"{context}: {exc}") from exc


@torch.inference_mode()
def reference_engine(projector, paths):
    tokens = []
    for path in paths:
        decoder = VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=2)
        if len(decoder) < 1:
            raise ValueError(f"Reference video has no frames: {path}")
        # Exact index convention used by CupidPipeline through
        # clips_at_regular_indices(num_clips=n, num_frames_per_clip=1, policy='wrap').
        indices = torch.linspace(0, len(decoder) - 1, steps=REFERENCE_FRAMES,
                                 dtype=torch.int).tolist()
        for index in indices:
            frame = decoder.get_frame_at(index)
            normalized, _ = extract_frame(projector, native_rgb(frame),
                                          f"Reference {path.name}, frame {index}")
            full = projector.model.encoder.forward_no_masking(normalized)
            tokens.append(full.permute(1, 0, 2).cpu())
        print(f"  reference: {path.name}, {len(indices)} frames", flush=True)
    reference_tokens = torch.cat(tokens, dim=0)
    return IdentityHeatmap(projector.model, reference_tokens, batch_size=BATCH_SIZE,
                           seed=SEED, max_pairs=MAX_PAIRS)


def cache_projection(result, directory, index, size):
    error = np.asarray(result["error"], dtype=np.float32)
    mask = np.asarray(result["mask"], dtype=np.float32)
    if error.shape != (size[1], size[0]) or mask.shape != error.shape:
        raise ValueError("Projector error/mask must match the native source HxW")
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("Projector coverage must be finite and in [0, 1]")
    visible = mask > 0
    values = error[visible]
    if not values.size:
        raise FaceFitError(f"{directory.name}, frame {index}: no visible projected face")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Visible analogy residuals must be finite and nonnegative")
    rows, cols = np.nonzero(visible)
    x0, y0, x1, y1 = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
    stem = directory / f"{index:05d}"
    np.savez_compressed(stem.with_suffix(".npz"),
                        error=np.where(visible[y0:y1, x0:x1], error[y0:y1, x0:x1], 0),
                        mask=mask[y0:y1, x0:x1])
    return {"cache_stem": str(stem), "bbox_xyxy": [x0, y0, x1, y1],
            "source_size_wh": list(size)}


@torch.inference_mode()
def process_clip(projector, engine, decoder, start, offset, label, frame_count, cache):
    directory = cache / label.lower()
    directory.mkdir()
    decoded_maps, geometries, records = [], [], []
    source_size = None
    for index in range(frame_count):
        requested = start + offset + index / FPS
        frame = decoder.get_frame_played_at(requested)
        rgb = native_rgb(frame)
        size = (rgb.shape[1], rgb.shape[0])
        if source_size is not None and size != source_size:
            raise ValueError(f"{label} changes native resolution inside the excerpt")
        source_size = size
        normalized, geometry = extract_frame(projector, rgb, f"{label}, output frame {index}")
        tokens = projector.model.encoder.forward_no_masking(normalized)
        decoded_maps.append(engine.decode_frame(tokens))
        geometries.append(geometry)
        Image.fromarray(rgb).save(directory / f"{index:05d}.png", compress_level=2)
    decoded = np.stack(decoded_maps).astype(np.float32, copy=False)
    windowed = temporal_residuals(decoded, engine.baseline, window=WINDOW)
    uv_support = np.stack([np.asarray(geometry["uv_mask"], dtype=bool) for geometry in geometries])
    if uv_support.shape != windowed.shape or not np.all(uv_support.reshape(frame_count, -1).any(axis=1)):
        raise ValueError("Each geometry must provide a nonempty HxW canonical UV face mask")
    if not np.all(uv_support == uv_support[0]):
        raise ValueError("Canonical UV support must be identical across frames for equal weighting")
    values = windowed[:, uv_support[0]]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Canonical analogy residuals must be finite and nonnegative")
    for index, (uv_map, geometry) in enumerate(zip(windowed, geometries)):
        try:
            result = projector.project(uv_map, geometry)
            records.append(cache_projection(result, directory, index, source_size))
        except FaceFitError as exc:
            raise FaceFitError(f"{label}, output frame {index}: {exc}") from exc
    print(f"  {label}: {frame_count} frames", flush=True)
    return records, uv_support[0]


def heatmap_colors(values):
    indices = np.minimum((np.asarray(values, dtype=np.float32) * 256).astype(np.int32), 255)
    return VIRIDIS[indices]


def normalized_error(error, scale):
    if scale == 0:
        return (error > 0).astype(np.float32)
    return np.clip(error / scale, 0, 1)


def set_display_crop(records):
    """Use one face-centered genuine crop matching the fake source aspect ratio."""
    width, height = records[0][0]["source_size_wh"]
    fake_width, fake_height = records[1][0]["source_size_wh"]
    aspect = fake_width / fake_height
    crop_width = min(width, round(height * aspect))
    crop_height = min(height, round(width / aspect))
    boxes = np.asarray([record["bbox_xyxy"] for record in records[0]])
    center_x, center_y = np.median((boxes[:, :2] + boxes[:, 2:]) / 2, axis=0)
    left = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, round(center_y - crop_height / 2)))
    crop = [left, top, left + crop_width, top + crop_height]
    for record in records[0]:
        record["display_crop_xyxy"] = crop


def make_canvas(records):
    # Both panels use the fake source aspect; the genuine source is cropped to it.
    fake_width, fake_height = records[1]["source_size_wh"]
    aspects = [fake_width / fake_height] * 2
    video_height = max(2, 2 * round(1280 / sum(aspects) / 2))
    widths = [max(2, 2 * round(video_height * aspect / 2)) for aspect in aspects]
    label_height = 42
    image = Image.new("RGB", (sum(widths), video_height + label_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    face = font(27)
    panels = []
    x = 0
    for label, color, width in zip(LABELS, LABEL_COLORS, widths):
        draw.text((x + width / 2, label_height / 2), label, font=face, fill=color, anchor="mm")
        panels.append((x, label_height, width, video_height))
        x += width
    return image, panels


def compose_frame(base, panels, records, scale):
    image = base.copy()
    for record, (x, y, width, height) in zip(records, panels):
        with Image.open(record["cache_stem"] + ".png") as source:
            rgb = np.array(source.convert("RGB"))
        with np.load(record["cache_stem"] + ".npz") as cached:
            error, mask = cached["error"], cached["mask"]
        x0, y0, x1, y1 = record["bbox_xyxy"]
        u = normalized_error(error, scale)
        alpha = (MAX_ALPHA * u ** ALPHA_EXPONENT * mask)[..., None]
        original = rgb[y0:y1, x0:x1].astype(np.float32)
        rgb[y0:y1, x0:x1] = np.rint(original * (1 - alpha) + heatmap_colors(u) * alpha).astype(np.uint8)
        if "display_crop_xyxy" in record:
            left, top, right, bottom = record["display_crop_xyxy"]
            rgb = rgb[top:bottom, left:right]
        factor = min(width / rgb.shape[1], height / rgb.shape[0])
        size = (max(1, round(rgb.shape[1] * factor)), max(1, round(rgb.shape[0] * factor)))
        tile = Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)
        image.paste(tile, (x + (width - size[0]) // 2, y + (height - size[1]) // 2))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEMO_DIR / "clips",
                        help="Directory containing the prepared demo clips")
    parser.add_argument("--out", type=Path, default=DEMO_DIR / "results",
                        help="Output directory for the selected comparison")
    parser.add_argument("--device", default="auto", help='"auto" or "cuda:N" (CUDA required)')
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required on PATH")
    clips = args.clips.expanduser().resolve()
    paths = [clips / name for name in SOURCE_NAMES]
    references = [clips / f"ref_{index}.mp4" for index in range(1, 9)]
    for path in paths + references:
        if not path.is_file():
            parser.error(f"Source clip does not exist: {path}")
    if paths[0].samefile(paths[1]):
        parser.error("Real and Fake must be different source paths")
    if any(reference.samefile(test) for reference in references for test in paths):
        parser.error("Reference sources must be disjoint from both test paths")
    if any(first.samefile(second) for index, first in enumerate(references) for second in references[index + 1:]):
        parser.error("Reference clips must be distinct files")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.mp4", "comparison.gif", "metadata.json"):
        target = out / name
        if target.resolve() in paths + references or (target.exists() and any(target.samefile(path) for path in paths + references)):
            parser.error(f"Output would overwrite an input: {target}")
    font(27)
    decoders = [VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=2) for path in paths]
    durations = [float(decoder.metadata.duration_seconds) for decoder in decoders]
    starts = [float(decoder.metadata.begin_stream_seconds or 0) for decoder in decoders]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Both clips must have a known positive video duration")
    available = [duration - offset for duration, offset in zip(durations, OFFSETS)]
    frame_count = math.floor(min(SECONDS, *available) * FPS)
    if frame_count < 1:
        parser.error("The available excerpt is shorter than one output frame")
    duration = frame_count / FPS
    with tempfile.TemporaryDirectory(prefix="cupid-overlay-") as temporary:
        render(paths, references, decoders, starts, frame_count, duration, args.device,
               out, Path(temporary))
    print(f"Saved {out / 'comparison.mp4'}, comparison.gif and metadata.json", flush=True)


def render(paths, references, decoders, starts, frame_count, duration, device, out, cache):
    projector = FaceOverlayProjector(device=resolve_device(device))
    print("Encoding the fixed reference set and constructing the analogy baseline", flush=True)
    engine = reference_engine(projector, references)
    records, supports = [], []
    for decoder, start, offset, label in zip(decoders, starts, OFFSETS, LABELS):
        clip_records, support = process_clip(projector, engine, decoder, start, offset,
                                             label, frame_count, cache)
        records.append(clip_records)
        supports.append(support)
    if not np.array_equal(supports[0], supports[1]):
        raise ValueError("Real and Fake must use identical canonical UV face support")
    set_display_crop(records)
    base, panels = make_canvas([clip[0] for clip in records])
    mp4 = out / "comparison.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{base.width}x{base.height}",
               "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
    encoder_log = cache / "ffmpeg.log"
    with encoder_log.open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index in range(frame_count):
                pair = [clip[index] for clip in records]
                image = compose_frame(base, panels, pair, SCALE)
                process.stdin.write(image.tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"MP4 encoding failed: {encoder_log.read_text(errors='replace')}")
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            raise
    gif_fps = min(FPS, 12.0)
    # Difference-only palette statistics can lose the static green/red labels.
    gif_filter = (f"fps={gif_fps:g},scale=768:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=max_colors=256:stats_mode=full[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-filter_complex", gif_filter, "-an", "-loop", "0", str(out / "comparison.gif")], check=True)
    metadata = {
        "inputs": [
            {**fingerprint(path), "label": label, "excerpt_offset_seconds": offset,
             "stream_start_seconds": start, "duration_seconds": duration,
             "source_size_wh": clip[0]["source_size_wh"]}
            for path, label, offset, start, clip in zip(paths, LABELS, OFFSETS, starts, records)
        ],
        "checkpoint": fingerprint(Path(projector.checkpoint_path)),
        "reference_sampling": {
            "sources": [fingerprint(path) for path in references],
            "clip_count": len(references), "frames_per_clip": REFERENCE_FRAMES,
            "total_frames": len(references) * REFERENCE_FRAMES,
            "convention": "Native frames at integer linspace(0, frame_count-1), including endpoints.",
        },
        "heatmap": {
            **engine.metadata,
            "temporal_smoothing": {
                "window_frames": WINDOW,
                "method": "Centered equal-weight mean in canonical UV before baseline subtraction and abs; truncate and renormalize at excerpt boundaries, separately per clip.",
            },
        },
        "display": {
            "scale": SCALE, "scale_convention": "Fixed shared upper bound; not estimated from this pair.",
            "colormap": "viridis", "opacity": f"{MAX_ALPHA} * clip(residual/scale,0,1)**{ALPHA_EXPONENT} * visible_mesh_coverage",
            "projection": "Bilinear UV projection; no added spatial smoothing; Lanczos RGB resize.",
            "real_crop_xyxy": records[0][0]["display_crop_xyxy"],
            "framing": "Real crop centered on median fitted face, matching uncropped Fake aspect ratio.",
            "labels": [{"text": label, "rgb": color} for label, color in zip(LABELS, LABEL_COLORS)],
        },
        "video": {
            "size_wh": list(base.size), "panels_xywh": panels, "fps": FPS,
            "frame_count": frame_count, "duration_seconds": duration, "audio": False,
            "sampling": "get_frame_played_at(stream_start + excerpt_offset + frame_index/fps), without interpolation.",
        },
        "gif": {
            "width": 768, "fps": gif_fps, "duration_seconds": duration,
            "palette_colors": 256, "dither": "bayer, scale 3",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    except FaceFitError as exc:
        raise SystemExit(f"Face fitting failed; no fallback overlay was invented: {exc}") from exc
