"""Score the fixed DiCaprio demo and render the Real 2 / Fake 2 comparison.

Run ./download.sh && ./prepare_clips.sh from demo/ to prepare the source clips,
then python demo/quickstart.py [--device cuda:0] from the repository root.
Outputs: scores.json, comparison.mp4, comparison.gif and metadata.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from cupid.pipeline import CupidPipeline

DEMO_DIR = Path(__file__).resolve().parent
FRAMES_PER_VIDEO = 15


def auc(real_scores, fake_scores):
    """Fraction of (genuine, deepfake) pairs ranked correctly; ties count half."""
    wins = sum((r > f) + 0.5 * (r == f) for r in real_scores for f in fake_scores)
    return wins / (len(real_scores) * len(fake_scores))


def find_clips(clips_dir):
    """Locate the eight references and seven fixed test clips."""
    groups = [
        [clips_dir / f"{prefix}_{index}.mp4" for index in range(1, count + 1)]
        for prefix, count in (("ref", 8), ("test_real", 3), ("test_fake", 4))
    ]
    missing = [path.name for group in groups for path in group if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing demo clips in {clips_dir}: {', '.join(missing)}. "
            "Run ./download.sh && ./prepare_clips.sh from demo/."
        )
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEMO_DIR / "clips",
                        help="Directory containing the prepared demo clips")
    parser.add_argument("--out", type=Path, default=DEMO_DIR / "results",
                        help="Output directory for scores and the selected comparison")
    parser.add_argument("--device", default="auto", help='"auto" or "cuda:N" (CUDA required)')
    args = parser.parse_args()
    try:
        refs, reals, fakes = find_clips(args.clips.expanduser())
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading CUPID and encoding 8 references (15 frames each, native resolution)...", flush=True)
    pipeline = CupidPipeline(device=args.device, frames_per_video=FRAMES_PER_VIDEO)
    reference_set = pipeline.extract_reference_set(refs)
    results = []
    for path, label in [(p, "real") for p in reals] + [(p, "fake") for p in fakes]:
        score = pipeline.score(reference_set["features"], path)
        results.append({"clip": path.name, "label": label, "score": score})
        print(f"  {path.name:<20} {score:+.4f}", flush=True)

    real_scores = [r["score"] for r in results if r["label"] == "real"]
    fake_scores = [r["score"] for r in results if r["label"] == "fake"]
    area = auc(real_scores, fake_scores)
    margin = min(real_scores) - max(fake_scores)
    print(f"AUC: {area:.3f}; margin (worst real - best fake): {margin:+.4f}")
    print("Scores rank identity similarity; they are not calibrated probabilities.")
    summary = {
        "reference_clips": [p.name for p in refs],
        "num_reference_descriptors": int(reference_set["features"].shape[0]),
        "frames_per_video": FRAMES_PER_VIDEO,
        "resolution": "native",
        "sampling": "equispaced",
        "auc": area,
        "margin": margin,
        "comparison": {"real": "test_real_2.mp4", "fake": "test_fake_2.mp4"},
        "results": sorted(results, key=lambda r: -r["score"]),
    }
    (args.out / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")

    device = str(pipeline.device)
    del pipeline, reference_set
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print("Rendering the Real 2 / Fake 2 comparison...", flush=True)
    subprocess.run([
        sys.executable, str(DEMO_DIR / "overlay_video.py"),
        "--clips", str(args.clips.expanduser()), "--out", str(args.out), "--device", device,
    ], check=True)


if __name__ == "__main__":
    main()
