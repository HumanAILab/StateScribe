from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.eval.adapters import load_evaluation_bundle
from benchmark.eval.judge import DEFAULT_JUDGE_MODEL, GeminiChangeJudge
from benchmark.eval.scorer import evaluate_bundle
from logging_config import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate benchmark predictions against annotated scene changes.")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to annotations.json.")
    parser.add_argument("--benchmark-output", type=Path, required=True, help="Path to benchmark output directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output evaluation JSON path; defaults to <benchmark-output>/evaluation.json.")
    parser.add_argument("--max-captures", type=int, default=None, help="Limit evaluation to the first N captures in annotation order.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate for latency computation.")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Gemini model used for GT matching.")
    parser.add_argument("--judge-max-workers", type=int, default=1, help="Number of concurrent judge requests to issue.")
    parser.add_argument("--max-clock-error-hours", type=int, default=1, help="Maximum clock-direction error for spatial correctness.")
    parser.add_argument("--max-distance-error-ft", type=float, default=2.0, help="Maximum distance error in feet for spatial correctness.")
    return parser


def main() -> int:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    bundle = load_evaluation_bundle(
        annotations_path=args.annotations,
        benchmark_output_dir=args.benchmark_output,
        max_captures=args.max_captures,
    )
    judge = GeminiChangeJudge(model=args.judge_model)
    evaluation = evaluate_bundle(
        bundle,
        judge,
        fps=args.fps,
        max_clock_error_hours=args.max_clock_error_hours,
        max_distance_error_ft=args.max_distance_error_ft,
        judge_max_workers=args.judge_max_workers,
    )

    output_path = args.output or (args.benchmark_output / "evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=True, indent=2)
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
