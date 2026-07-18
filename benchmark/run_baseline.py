from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.baselines import (
    DEFAULT_ONLINE_GEMINI_MODEL,
    GeminiOfflineBaselineBackend,
    GeminiOnlineBaselineBackend,
    OfflineBaselineRunner,
    OnlineBaselineRunner,
)
from benchmark.config import DEFAULT_DATASET_DIR, DEFAULT_FRAME_STRIDE, DEFAULT_OUTPUT_ROOT, DEFAULT_USE_ALIGNED_METADATA
from logging_config import setup_logging

LOGGER = logging.getLogger("benchmark.run_baseline")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline benchmarks with a unified benchmark output format.")
    parser.add_argument(
        "--baseline",
        choices=("online_gemini", "offline_gemini"),
        default="online_gemini",
        help="Baseline method to run.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset directory containing capture folders.")
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE, help="Take every N valid frame.")
    parser.add_argument("--max-captures", type=int, default=None, help="Read only first N captures in dataset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument(
        "--use-aligned-metadata",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ALIGNED_METADATA,
        help="Use metadata_aligned.jsonl when available.",
    )
    parser.add_argument(
        "--require-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require confidence frame files when collecting valid frames.",
    )
    parser.add_argument("--world-name-prefix", type=str, default="baseline", help="Prefix for generated world name.")
    parser.add_argument("--model", type=str, default=DEFAULT_ONLINE_GEMINI_MODEL, help="Model name for the selected baseline.")
    parser.add_argument("--timeout-ms", type=int, default=1000000, help="Per-request Gemini timeout in milliseconds.")
    return parser


def main() -> int:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        backend = _build_backend(
            baseline=args.baseline,
            model=args.model,
            timeout_ms=args.timeout_ms,
        )
        if args.baseline == "offline_gemini":
            runner = OfflineBaselineRunner(
                backend=backend,
                dataset_dir=args.dataset,
                frame_stride=args.frame_stride,
                max_captures=args.max_captures,
                output_root=args.output_root,
                use_aligned_metadata=bool(args.use_aligned_metadata),
                require_confidence=bool(args.require_confidence),
                world_name_prefix=args.world_name_prefix,
            )
        else:
            runner = OnlineBaselineRunner(
                backend=backend,
                dataset_dir=args.dataset,
                frame_stride=args.frame_stride,
                max_captures=args.max_captures,
                output_root=args.output_root,
                use_aligned_metadata=bool(args.use_aligned_metadata),
                require_confidence=bool(args.require_confidence),
                world_name_prefix=args.world_name_prefix,
            )
        output_files = runner.run()
    except Exception:
        LOGGER.exception("Baseline benchmark run failed.")
        return 1

    LOGGER.info("Baseline benchmark finished.")
    for key, path in output_files.items():
        LOGGER.info("%s: %s", key, path)
    return 0


def _build_backend(*, baseline: str, model: str, timeout_ms: int):
    if baseline == "online_gemini":
        return GeminiOnlineBaselineBackend(
            model_name=model,
            timeout_ms=timeout_ms,
        )
    if baseline == "offline_gemini":
        return GeminiOfflineBaselineBackend(
            model_name=model,
            timeout_ms=timeout_ms,
        )
    raise ValueError(f"Unsupported baseline: {baseline}")


if __name__ == "__main__":
    raise SystemExit(main())
