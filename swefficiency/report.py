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
Report generation for SWE-fficiency evaluation results.
"""

import json
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Dict, List

import datasets
import pandas as pd
from tqdm import tqdm

from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER


def parse_perf_summary(perf_summary: str) -> Dict[str, float]:
    """Parse performance summary file content."""
    perf_lines = perf_summary.strip().splitlines()

    before_mean = float(perf_lines[0].split(":")[1].strip())
    before_std = float(perf_lines[1].split(":")[1].strip())
    after_mean = float(perf_lines[2].split(":")[1].strip())
    after_std = float(perf_lines[3].split(":")[1].strip())
    improvement = (after_mean - before_mean) / before_mean * 100

    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "before_std": before_std,
        "after_std": after_std,
        "improvement": improvement,
    }


def get_number_of_patch_modified_lines(git_patch_text: str) -> int:
    """Count the number of modified lines in a git patch text."""
    num_lines = 0
    for line in git_patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            num_lines += 1
        if line.startswith("-") and not line.startswith("---"):
            num_lines += 1
    return num_lines


def evaluate_instance(
    instance: dict, gold_run: Path, pred_run: Path, use_correctness_files: bool = True
) -> Dict:
    """
    Evaluate a single instance comparing gold and prediction runs.

    Args:
        instance: Dataset instance with instance_id, PASS_TO_PASS, patch, repo
        gold_run: Path to gold run directory
        pred_run: Path to prediction run directory
        use_correctness_files: Whether to use pre-computed correctness files

    Returns:
        Dictionary with evaluation metrics for this instance
    """
    instance_id = instance["instance_id"]
    pass_to_pass = instance["PASS_TO_PASS"]

    gold_run_entry = gold_run / instance_id / "perf_summary.txt"
    pred_run_entry = pred_run / instance_id / "perf_summary.txt"

    # Compute prediction speedup ratio.
    if pred_run_entry.exists():
        pred_perf_info = parse_perf_summary(pred_run_entry.read_text())
        pred_speedup_ratio = (
            pred_perf_info["before_mean"] / pred_perf_info["after_mean"]
        )
    else:
        pred_speedup_ratio = 1.0  # No speedup if no prediction run exists

    # Compute gold speedup ratio.
    if gold_run_entry.exists():
        gold_perf_info = parse_perf_summary(gold_run_entry.read_text())
        gold_speedup_ratio = (
            gold_perf_info["before_mean"] / gold_perf_info["after_mean"]
        )
    else:
        gold_speedup_ratio = 1.0
        gold_perf_info = {"before_mean": 0.0}

    # Check that pass to pass tests are still passing.
    correctness_dir = pred_run / instance_id / "raw_correctness_output"
    num_modified_lines = get_number_of_patch_modified_lines(instance["patch"])

    if not correctness_dir.exists():
        return {
            "instance_id": instance_id,
            "raw_pred_speedup_ratio": 1.0,
            "pred_speedup_ratio": 1.0,
            "gold_speedup_ratio": gold_speedup_ratio,
            "human_speedup_ratio": (
                1.0 / gold_speedup_ratio if gold_speedup_ratio != 0 else 0
            ),
            "correctness": 0.0,
            "correctness_pct": 0.0,
            "pre_edit_runtime": gold_perf_info["before_mean"],
            "patch_length": num_modified_lines,
        }

    if not use_correctness_files:
        pred_statuses = {}
        for test_output in correctness_dir.glob("*.txt"):
            test_output_text = test_output.read_text()
            pred_statuses.update(
                MAP_REPO_TO_PARSER[instance["repo"]](test_output_text)
            )
    else:
        pred_statuses = json.loads(
            (pred_run / instance_id / "covering_test_status.json").read_text()
        )

    passed_tests = []
    for test in pass_to_pass:
        if "PASS" in pred_statuses.get(test, ""):
            passed_tests.append(test)

    passed_tests = set(passed_tests)
    correctness_pct = len(passed_tests) / len(pass_to_pass) if pass_to_pass else 1.0
    adjusted_pred_speedup_ratio = 1.0 if correctness_pct != 1.0 else pred_speedup_ratio

    return {
        "instance_id": instance_id,
        "raw_pred_speedup_ratio": pred_speedup_ratio,
        "pred_speedup_ratio": adjusted_pred_speedup_ratio,
        "gold_speedup_ratio": gold_speedup_ratio,
        "human_speedup_ratio": (
            adjusted_pred_speedup_ratio / gold_speedup_ratio
            if gold_speedup_ratio != 0
            else 0
        ),
        "correctness": 0.0 if correctness_pct != 1.0 else 1.0,
        "correctness_pct": correctness_pct,
        "pre_edit_runtime": gold_perf_info["before_mean"],
        "patch_length": len(instance["patch"].splitlines()),
    }


def compute_performance_breakdown(df: pd.DataFrame) -> Dict:
    """
    Compute performance breakdown metrics from evaluation results.

    Args:
        df: DataFrame with evaluation results

    Returns:
        Dictionary with breakdown metrics
    """
    total_instances = len(df)
    if total_instances == 0:
        return {
            "total_instances": 0,
            "overall_score": 0.0,
            "proportion_incorrect": 0.0,
            "proportion_correct_but_no_speedup": 0.0,
            "proportion_correct_with_speedup_but_human_no_speedup": 0.0,
            "proportion_human_speedup_or_better": 0.0,
        }

    # Compute harmonic mean of human speedup ratios (overall score)
    # Floor human_speedup_ratio at 0.001 (capping effective gold speedup at
    # 1000x) to prevent extreme outliers from dominating the benchmark score.
    floored_human_speedup = df["human_speedup_ratio"].clip(lower=0.001)
    overall_score = total_instances / (1 / floored_human_speedup).sum()

    # Proportion incorrect (correctness < 1.0)
    incorrect_instances = (df["correctness"] < 1.0).sum()
    proportion_incorrect = incorrect_instances / total_instances

    # Proportion correct but no speedup
    correct_but_no_speedup = (
        (df["correctness"] == 1.0) & (df["raw_pred_speedup_ratio"] < 1.0)
    ).sum()
    proportion_correct_but_no_speedup = correct_but_no_speedup / total_instances

    # Proportion correct with speedup but human no speedup
    correct_with_speedup_but_human_no_speedup = (
        (df["correctness"] == 1.0)
        & (df["raw_pred_speedup_ratio"] >= 1.0)
        & (df["human_speedup_ratio"] < 1.0)
    ).sum()
    proportion_correct_with_speedup_but_human_no_speedup = (
        correct_with_speedup_but_human_no_speedup / total_instances
    )

    # Proportion with human speedup or better
    human_speedup_or_better = (df["human_speedup_ratio"] >= 1.0).sum()
    proportion_human_speedup_or_better = human_speedup_or_better / total_instances

    return {
        "total_instances": total_instances,
        "overall_score": round(overall_score, 4),
        "proportion_incorrect": round(proportion_incorrect, 4),
        "proportion_correct_but_no_speedup": round(
            proportion_correct_but_no_speedup, 4
        ),
        "proportion_correct_with_speedup_but_human_no_speedup": round(
            proportion_correct_with_speedup_but_human_no_speedup, 4
        ),
        "proportion_human_speedup_or_better": round(
            proportion_human_speedup_or_better, 4
        ),
    }


def generate_report(
    gold_run: Path,
    pred_run: Path,
    output_dir: Path,
    num_workers: int = 4,
    dataset_name: str = "swefficiency/swefficiency",
) -> pd.DataFrame:
    """
    Generate evaluation report comparing gold and prediction runs.

    Args:
        gold_run: Path to gold run directory
        pred_run: Path to prediction run directory
        output_dir: Output directory for reports
        num_workers: Number of parallel workers
        dataset_name: HuggingFace dataset name

    Returns:
        DataFrame with evaluation results
    """
    from swefficiency.harness.utils import load_swefficiency_dataset
    ds = load_swefficiency_dataset(dataset_name, split="test")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = pred_run.name

    worker = partial(evaluate_instance, gold_run=gold_run, pred_run=pred_run)
    with multiprocessing.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker, ds, chunksize=1),
                total=len(ds),
                desc="Evaluating instances",
            )
        )

    results_df = pd.DataFrame(results)

    # Save CSV report
    csv_path = output_dir / f"eval_report_{report_name}.csv"
    results_df.to_csv(csv_path, index=False)

    # Compute and save JSON report with breakdown
    breakdown = compute_performance_breakdown(results_df)
    breakdown["report"] = csv_path.name
    json_path = output_dir / f"eval_report_{report_name}.json"
    with open(json_path, "w") as f:
        json.dump(breakdown, f, indent=2)

    return results_df, breakdown, csv_path, json_path
