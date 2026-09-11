"""CUPID command-line interface: extract a person's reference set, score test videos."""

import argparse
import json

import torch

from cupid import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cupid",
        description=(
            "CUPID person-of-interest deepfake detection. "
            "Extract a person's reference set from reference videos, then score test videos. "
            "Scores are cosine similarities in [-1, 1]: higher = more likely "
            "the genuine person. No fixed threshold is shipped; calibrate on "
            "your own data."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--device",
        default="auto",
        help='Compute device: "auto" (first CUDA GPU) or "cuda:N" (default: auto)',
    )
    common.add_argument(
        "--frames",
        type=int,
        default=15,
        help="Frames sampled at regular intervals from each video (default: 15)",
    )
    common.add_argument(
        "--weights-dir",
        default=None,
        help=(
            "Local folder containing all weight files (skips the HuggingFace "
            "auto-download; also settable via CUPID_WEIGHTS_DIR)"
        ),
    )

    extract = subparsers.add_parser(
        "extract-reference",
        parents=[common],
        help="Extract a reference set for a person of interest",
    )
    extract.add_argument(
        "--reference",
        nargs="+",
        required=True,
        metavar="VIDEO",
        help="One or more reference videos of the genuine person (2-3 recommended)",
    )
    extract.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output reference-set file (e.g. poi.pt)",
    )
    extract.add_argument(
        "--heatmap",
        action="store_true",
        help="Store full reference tokens for heatmaps (larger reference-set file)",
    )

    score = subparsers.add_parser(
        "score",
        parents=[common],
        help="Score a test video against a person's reference set",
    )
    refs = score.add_mutually_exclusive_group(required=True)
    refs.add_argument(
        "--reference-set",
        help="Reference-set file produced by `cupid extract-reference`",
    )
    refs.add_argument(
        "--reference",
        nargs="+",
        metavar="VIDEO",
        help="Reference videos (one-shot scoring without a saved reference set)",
    )
    score.add_argument("--test", required=True, help="Test video to score")
    score.add_argument(
        "--heatmap",
        metavar="PNG",
        help="Save one UV heatmap for the test video; saved references must include tokens",
    )
    score.add_argument(
        "--json", action="store_true", help="Emit the result as a JSON object"
    )

    return parser


def make_pipeline(args):
    from cupid.pipeline import CupidPipeline

    return CupidPipeline(
        device=args.device,
        frames_per_video=args.frames,
        weights_dir=args.weights_dir,
    )


def cmd_extract_reference(args):
    pipeline = make_pipeline(args)
    reference_set = pipeline.extract_reference_set(args.reference, include_tokens=args.heatmap)
    torch.save(reference_set, args.output)
    print(
        f"Extracted a reference set from {reference_set['num_videos']} reference video(s) "
        f"({reference_set['features'].shape[0]} feature vectors) -> {args.output}"
    )


def cmd_score(args):
    if args.reference_set:
        reference_set = torch.load(args.reference_set, map_location="cpu", weights_only=True)
        if args.heatmap:
            from cupid.heatmap import require_reference_tokens

            try:
                require_reference_tokens(reference_set)
            except ValueError as error:
                raise SystemExit(str(error)) from None

    # Reject old CLS-only reference sets before loading models or decoding video.
    pipeline = make_pipeline(args)
    if args.reference_set:
        if reference_set.get("frames_per_video") != args.frames:
            print(
                f"Note: the reference set used {reference_set.get('frames_per_video')} frames "
                f"per video, the test video uses {args.frames}. Scores remain comparable."
            )
    else:
        reference_set = pipeline.extract_reference_set(
            args.reference, include_tokens=bool(args.heatmap)
        )

    ref_features = reference_set["features"]
    if args.heatmap:
        from cupid.heatmap import save_heatmap

        result = pipeline.interpret(reference_set, args.test)
        save_heatmap(result["heatmap"], args.heatmap)
        score = result["score"]
    else:
        score = pipeline.score(ref_features, args.test)

    if args.json:
        print(
            json.dumps(
                {
                    "test": args.test,
                    "score": round(score, 6),
                    "num_reference_features": int(ref_features.shape[0]),
                    "frames": args.frames,
                }
            )
        )
    else:
        print(f"score: {score:.4f}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "extract-reference":
        cmd_extract_reference(args)
    elif args.command == "score":
        cmd_score(args)


if __name__ == "__main__":
    main()
