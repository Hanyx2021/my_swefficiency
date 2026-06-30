# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Command line interface for swefficiency package.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from swefficiency.harness.constants import SWEfficiencyInstance
from swefficiency.harness.run_validation import main as run_validation_main
from swefficiency.harness.utils import load_swefficiency_dataset
from swefficiency.report import generate_report

DEFAULT_DATASET = "swefficiency/swefficiency"


def filter_instances_by_regex(
    instances: List[SWEfficiencyInstance], regex_pattern: str
) -> List[str]:
    """
    Filter instances by regex pattern applied to instance IDs.

    Args:
        instances: List of dataset instances
        regex_pattern: Regular expression pattern to match instance IDs

    Returns:
        List of instance IDs that match the pattern
    """
    pattern = re.compile(regex_pattern)
    return [
        instance["instance_id"]
        for instance in instances
        if pattern.search(instance["instance_id"])
    ]


def eval_command(args):
    """
    Handle the eval subcommand.
    """
    # Set defaults
    dataset_name = args.dataset if args.dataset else DEFAULT_DATASET
    split = "test"

    # Handle prediction path - default to None if not provided
    predictions_path = args.prediction_path if args.prediction_path else None

    # Handle Docker image source
    use_dockerhub_images = args.use_dockerhub_images if args.use_dockerhub_images else False

    # Handle instance filtering by regex
    instance_ids = None
    if args.instances_regex:
        # Load dataset to get all instances
        dataset = load_swefficiency_dataset(dataset_name, split)
        instance_ids = filter_instances_by_regex(dataset, args.instances_regex)
        if not instance_ids:
            print(
                f"Warning: No instances matched regex pattern '{args.instances_regex}'"
            )
            return
        print(
            f"Found {len(instance_ids)} instances matching regex pattern '{args.instances_regex}'"
        )

    # Generate a run ID if not provided
    run_id = args.run_id if args.run_id else f"cli_run_{int(time.time())}"
    force_rerun = args.force_rerun

    print(f"Starting evaluation with {args.num_workers} workers...")
    if predictions_path:
        print(f"Predictions path: {predictions_path}")
    else:
        print("Using gold predictions (no prediction path provided)")
    if instance_ids:
        print(f"Running on {len(instance_ids)} filtered instances")
    print(f"Run ID: {run_id}")

    # Run the evaluation using run_validation.py's main function
    try:
        run_validation_main(
            dataset_name=dataset_name,
            split=split,
            instance_ids=instance_ids or [],
            max_workers=args.num_workers,
            max_build_workers=16,
            force_rebuild=False,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=4096,
            run_id=run_id,
            timeout=86400,
            allow_test_patch=False,
            run_coverage=False,
            run_perf=True,
            run_perf_profiling=False,
            run_correctness=True,
            empty_patch=False,
            model_predictions=predictions_path or "",
            gdrive_annotation_sheet="",
            push_to_dockerhub=False,
            use_dockerhub_images=use_dockerhub_images,
            use_podman=False,
            workload_predictions="",
            force_rerun=force_rerun,
            process_isolation=True,
        )
        print("Evaluation completed successfully!")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


def report_command(args):
    """
    Handle the report subcommand.
    """
    gold_run_path = Path(args.gold_run)
    pred_run_path = Path(args.pred_run)

    # Check if we have the gold run
    if not gold_run_path.exists():
        print(f"Error: Gold run directory not found at {gold_run_path}")
        print("Make sure you have run evaluation with gold predictions first.")
        sys.exit(1)

    # Check if we have the prediction run
    if not pred_run_path.exists():
        print(f"Error: Prediction run directory not found at {pred_run_path}")
        print("Make sure you have run evaluation with your predictions first.")
        sys.exit(1)

    output_dir = (
        Path(args.report_output) if args.report_output else Path("eval_reports")
    )

    print("Generating report comparing gold run and prediction run...")
    print(f"Gold run: {gold_run_path}")
    print(f"Prediction run: {pred_run_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        _, breakdown, csv_path, json_path = generate_report(
            gold_run=gold_run_path,
            pred_run=pred_run_path,
            output_dir=output_dir,
            num_workers=args.num_workers,
        )

        print(f"\nCSV report: {csv_path}")
        print(f"JSON report: {json_path}")
        print("\nPerformance Breakdown:")
        print(f"  Total instances: {breakdown['total_instances']}")
        print(f"  Overall score (harmonic mean): {breakdown['overall_score']}")
        print(f"  Proportion incorrect: {breakdown['proportion_incorrect']}")
        print(
            f"  Proportion correct but no speedup: {breakdown['proportion_correct_but_no_speedup']}"
        )
        print(
            f"  Proportion correct with speedup but human no speedup: {breakdown['proportion_correct_with_speedup_but_human_no_speedup']}"
        )
        print(
            f"  Proportion with human speedup or better: {breakdown['proportion_human_speedup_or_better']}"
        )

    except Exception as e:
        print(f"Error during report generation: {e}")
        sys.exit(1)


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        prog="swefficiency",
        description="SWE-fficiency: A benchmark for evaluating LMs on software engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation on predictions",
        description="Evaluate model predictions on the SWE-fficiency benchmark",
    )
    eval_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers to use (default: 4)",
    )
    eval_parser.add_argument(
        "--run_id",
        type=str,
        help="Run ID to identify this evaluation run. If not provided, auto-generates one.",
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset to evaluate on (default: {DEFAULT_DATASET})",
    )
    eval_parser.add_argument(
        "--prediction_path",
        type=str,
        help="Path to predictions file (.json or .jsonl). If not provided, uses gold predictions for baseline.",
    )
    eval_parser.add_argument(
        "--instances_regex",
        type=str,
        help='Regular expression pattern to filter instance IDs (e.g., "pandas.*" for Pandas instances)',
    )
    eval_parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force re-running the evaluation even if results already exist",
    )
    eval_parser.add_argument(
        "--use_dockerhub_images",
        action="store_true",
        help="Use pre-built Docker images from DockerHub (slower in China). If not set, will build images locally.",
    )
    eval_parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        default="env",
        help="Cache level - remove images above this level (default: env)",
    )
    eval_parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean images above cache level (default: False)",
    )

    # Report subcommand
    report_parser = subparsers.add_parser(
        "report",
        help="Generate evaluation report",
        description="Generate evaluation report comparing gold and prediction runs. Outputs both CSV and JSON with performance breakdown.",
    )
    report_parser.add_argument(
        "--gold_run",
        type=str,
        required=True,
        help="Path to gold run directory (e.g., logs/run_evaluation/my_eval/gold)",
    )
    report_parser.add_argument(
        "--pred_run",
        type=str,
        required=True,
        help="Path to prediction run directory (e.g., logs/run_evaluation/my_eval/model_name)",
    )
    report_parser.add_argument(
        "--report_output",
        type=str,
        default="eval_reports",
        help="Output directory for the report (default: eval_reports)",
    )
    report_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for report generation (default: 4)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle subcommands
    if args.command == "eval":
        eval_command(args)
    elif args.command == "report":
        report_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
