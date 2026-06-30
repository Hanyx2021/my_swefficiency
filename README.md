
<div align="center">
  <img src="docs/assets/logos/swefficiency_banner_main.png" alt="SWE-fficiency Logo" width="500"/>
</div>


<p align="center">
  <a href="https://swefficiency.com">
    <img src="https://img.shields.io/badge/project-Home-b31b1b.svg" alt="home">
  </a>
  <a href="https://huggingface.co/datasets/swefficiency/swefficiency">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue" alt="Data">
  </a>
  <a href="https://arxiv.org/abs/2511.06090">
    <img src="https://img.shields.io/badge/arXiv-2511.06090-b31b1b.svg" alt="paper">
  </a>
</p>

---

# SWE-fficiency: Can Language Models Optimize Real World Repositories on Real World Workloads?

**TL;DR** — SWE-fficiency is a *repository-level* benchmark for **performance optimization** (not bug fixing). Each task ships:
- a full codebase,
- a targeted **performance workload** to speed up,
- and the subset of repo **correctness tests** that must remain green.

We evaluate patches by applying them, running the correctness suite, and measuring runtime speedups vs. the *expert (human) PR*, reporting **Speedup Ratio (SR)**.

---

## 🚀 What is SWE-fficiency?

SWE-fficiency evaluates *pass-to-pass* performance engineering: start from a codebase and a slow workload, improve runtime, and **don’t break behavior**. The focus is on **investigation** (profiling/localization) and **correctness-preserving** edits—mirroring how performance engineers work day-to-day.

### Highlights
- **Real repos, real workloads**: **498** tasks from **9** major Python libraries—**numpy, scipy, pandas, scikit-learn, matplotlib, xarray, sympy, dask, astropy**.
- **Correctness-preserving**: Edits must pass the repo’s own unit/integration tests *covering the changed code*.
- **Reproducible evaluation**: Prebuilt, containerized environments; per-task CPU/memory pinning recommended (**4 vCPUs, 16 GB RAM** per worker).
- **Metric**: **Speedup Ratio (SR)** = *(LM speedup) / (expert speedup)*; aggregate with a harmonic mean. SR>1.0 means you beat the human baseline.

### Why this matters
Performance improvements in widely used libraries have outsized impact. SWE-fficiency isolates the open-ended challenge: **find bottlenecks, propose safe optimizations, and prove correctness** against the repo’s own tests—at repository scope.

---

## 📦 Install & Environment

We recommend Python 3.12 and a Linux host. The benchmark is also installable via `pip` in editable mode.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Alternatively, you can install directly via pip.
pip install -e .
```

## Quick Start

Evaluating on SWE-fficiency is a multi-step process via our package's CLI

### Step 0: VM / Container Setup (highly recommended for reproducibility)

For faithful reproduction of paper results, use a large VM (for identical leaderboard setup, use GCP `n2-standard-64`) and run the setup scripts to configure Docker and CPU pinning. We recommend using `--num_workers 12` on this configuration, which allocates 4 vCPUs and 16 GB RAM per worker.

```bash
bash scripts/vm/setup_vm.sh

# IMPORTANT: This script pins the number of CPUs for the docker daemon
# hence why it must be run in sudo priveleges. This is so image building
# and pulling overhead does not interfere with evaluation.
sudo scripts/vm/setup_docker.sh MEM_MAX MEM_HIGH
```

### Step 1: Run gold baseline (establishes reference performance)

```bash
swefficiency eval --run_id my_eval --num_workers 12
swefficiency eval --run_id my_eval --num_workers 12 --dataset /home/hyx/swe_data/data
```

This runs the expert (human) patches to establish baseline performance metrics. Results are stored in `logs/run_evaluation/my_eval/gold/`.

### Step 2: Run your model predictions

```bash
swefficiency eval --run_id my_eval --num_workers 12 --prediction_path predictions.jsonl
```

Your predictions file should be JSONL with each line containing:
```json
{"instance_id": "<id>", "model_patch": "<patch_text>", "model_name_or_path": "<model_name>"}
```

Results are stored in `logs/run_evaluation/my_eval/<model_name>/`.

### Step 3: Generate evaluation report

```bash
swefficiency report \
    --gold_run logs/run_evaluation/my_eval/gold \
    --pred_run logs/run_evaluation/my_eval/<model_name>
```

This generates two output files in `eval_reports/`:
- `eval_report_<model_name>.csv` - Per-instance results
- `eval_report_<model_name>.json` - Summary metrics including:
  - `overall_score`: Harmonic mean of speedup ratios
  - `proportion_incorrect`: Instances that failed correctness tests
  - `proportion_correct_but_no_speedup`: Correct but slower than baseline
  - `proportion_human_speedup_or_better`: Matched or beat expert performance

You can also point to arbitrary paths if your evaluation results are stored elsewhere:
```bash
swefficiency report \
    --gold_run /path/to/gold/results \
    --pred_run /path/to/model/results \
    --report_output my_reports
```



---

## 🧰 Dataset

* **Location**: [Hugging Face — swefficiency/swefficiency](https://huggingface.co/datasets/swefficiency/swefficiency)
* **Task structure** (per instance):

  * Repo snapshot + diff metadata
  * A **performance workload** script that exhibits a measurable speedup under the expert patch
  * The **set of repo tests** whose coverage intersects the expert diff (the “guarding” tests)

> The workloads are **separate from correctness tests** (as in real projects). The benchmark rejects instances whose speedups are not statistically significant in a controlled environment.

---

## 📊 Evaluation

### Metric: Speedup Ratio (SR)

For each instance:

* Let `T_pre` be workload runtime pre-edit.
* Let `T_post_gold` be runtime after applying the **expert** patch.
* Let `T_post_lm` be runtime after applying your model’s patch.

**Expert speedup** = `T_pre / T_post_gold`
**Model speedup** = `T_pre / T_post_lm`
**Speedup Ratio (SR)** = `Model speedup / Expert speedup`.

* We aggregate SR across tasks with the **harmonic mean**.
* If a patch **fails correctness tests** or **doesn’t apply**, the instance is scored as if **no LM edit** were attempted (`T_pre / T_post_lm = 1`).

### Two-stage evaluation pipeline

1. **Run Patch Evaluation** — Apply predicted patches, run guarding correctness tests, run the performance workload; store logs and raw measurements.
2. **Check Evaluation** — Aggregate JSON/CSV artifacts into final metrics (SR, pass rates, etc.).

See the [Quick Start](#quick-start) section above for CLI usage, or `scripts/eval/README.md` for advanced options.

---

## 🛠️ Generation (Agents & Harness)

We provide integration points for popular SWE agent harnesses lie OpenHands and SWE-agent via already containerized docker containers.

We ship **prebuilt Docker images** for generation to match the evaluation environment and avoid dependency drift.

> Recommended per-task limits (matching paper setup): **3 hours** wall-clock, **100** max actions/turns; be generous with workload timeouts (since tests or workloads can be substantial).

Need a generalized way to prep instances, run your agent, and capture patches? See
`scripts/inference/README.md` for the `cursor.py` harness. It loads the
SWE-fficiency dataset directly from Hugging Face, runs prework/inference steps
defined in YAML specs (Cursor CLI example included), and writes git patches ready
for `swefficiency eval`.

---

## 🔬 Reproducibility Tips

* Use the provided **container images** (prebuilt for each instance).
* **Pin CPU and memory** per worker (4 vCPUs / 16 GB RAM). See `scripts/vm/` scripts for more details.
* Pre-built images include everything needed.

---

## 📈 Baseline Snapshot

We include reference results in the paper across several modern LMs using OpenHands/SWE-agent. Overall, agents today are **far from expert parity** (SR ≪ 1×) and frequently introduce correctness regressions when attempting optimizations. See paper for full tables and analysis.

---

## 🧭 Project Structure (high level)

```
.
├── scripts/
│   ├── eval/           # evaluation runner + aggregator
│   └── vm/             # docker & VM pinning helpers
├── swefficiency/       # python package (cli, utils, loaders)
├── assets/figures/     # logos, diagrams
└── README.md
```

---

## Acknowledgements

This codebase began as a fork from SWE-Gym's fork of SWE-bench (https://github.com/SWE-Gym/SWE-Bench-Fork). We updated repo specific dependencies in the constants files, extended the data pipeline to be able to filter performance specific commits (as per our paper), and updated the evaluation harness to validate our performance + correctness setting. We've also added several helper scripts and utilities to support evaluation and experiment analysis

## License

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).