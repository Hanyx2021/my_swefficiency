#!/usr/bin/env python3

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

"""Generalized inference harness for SWE-fficiency Docker instances.

This tool downloads instances from the Hugging Face dataset, launches
``swefficiency/swefficiency_images:<instance_id>`` containers, runs user-specified
prework/inference steps, and extracts git patches for downstream evaluation.

Design goals:
- No dependency on the swefficiency Python package – only standard deps (datasets,
  docker, yaml, jinja2).
- YAML-based specs describing prework templates, inference commands, resource
  limits, and artifact collection so Cursor CLI or other agent flows can plug in.
- CPU/memory pinning knobs similar to the evaluation harness.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import re
import shlex
import sys
import tarfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import datasets
import docker
import yaml
from dotenv import load_dotenv
from jinja2 import Template

from swefficiency.harness.cpu_assignment import SYS_CPU

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "run_inference"
DEFAULT_SPEC_TEMPLATE = "swefficiency/swefficiency_images:{instance_id}"
DEFAULT_PATCH_CONTAINER_PATH = "/tmp/model.patch"
DEFAULT_PATCH_COMMAND = (
    'BASE_COMMIT="{{ instance.base_commit | default("", true) }}"; '
    "cd /testbed && "
    "git add -N . >/dev/null 2>&1 || true; "
    'if [ -n "$BASE_COMMIT" ]; then '
    'git diff --binary "$BASE_COMMIT" > /tmp/model.patch; '
    "else "
    "git diff --binary HEAD > /tmp/model.patch; "
    "fi"
)

load_dotenv(ENV_PATH)


class HarnessError(Exception):
    """Raised when any per-instance processing step fails."""


@dataclass(slots=True)
class ScriptStep:
    """Represents either a prework script or any template-backed command."""

    name: str
    template: Path
    destination: str
    execute: bool = True
    command: Optional[str] = None
    shell: str = "/bin/bash"
    chmod_x: bool = True
    continue_on_error: bool = False
    env: Dict[str, str] = field(default_factory=dict)
    timeout_sec: Optional[int] = None


@dataclass(slots=True)
class InferenceCommand:
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    shell: str = "/bin/bash"
    timeout_sec: Optional[int] = None
    log_name: str = "inference.log"


@dataclass(slots=True)
class PatchSpec:
    command: str = DEFAULT_PATCH_COMMAND
    container_path: str = DEFAULT_PATCH_CONTAINER_PATH
    host_filename: str = "patch.diff"
    shell: str = "/bin/bash"
    timeout_sec: Optional[int] = None


@dataclass(slots=True)
class Spec:
    name: str
    description: str
    docker_workdir: str
    docker_user: str
    image_template: str
    scripts: List[ScriptStep]
    inference: InferenceCommand
    patch: PatchSpec
    artifacts: List[dict]
    variables: Dict[str, Any] = field(default_factory=dict)


def parse_kv_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def load_vars_file(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    raw = path.read_text()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return yaml.safe_load(raw)


def ensure_path(base: Path, maybe_path: str | None) -> Path:
    if not maybe_path:
        raise ValueError("Path value missing in spec")
    candidate = Path(maybe_path)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def render_template(template_path: Path, context: Dict[str, Any]) -> str:
    template_text = template_path.read_text()
    return Template(template_text).render(**context)


def render_inline(text: str, context: Dict[str, Any]) -> str:
    return Template(text).render(**context)


def render_env(env_map: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
    return {k: render_inline(str(v), context) for k, v in env_map.items()}


def _iter_dataset(
    dataset_name: str,
    split: str,
) -> Iterable[dict]:
    hf_kwargs = {}
    ds = datasets.load_dataset(dataset_name, split=split, **hf_kwargs)
    for row in ds:
        # datasets rows behave like dicts but static typing does not know that.
        yield dict(row)  # type: ignore[arg-type]


def filter_instances(
    instances: Iterable[dict],
    instance_ids: Optional[set[str]],
    regex: Optional[re.Pattern[str]],
    limit: Optional[int],
) -> List[dict]:
    selected: List[dict] = []
    for inst in instances:
        iid = inst.get("instance_id")
        if not iid:
            continue
        if instance_ids and iid not in instance_ids:
            continue
        if regex and not regex.search(iid):
            continue
        selected.append(inst)
        if limit and len(selected) >= limit:
            break
    return selected


def load_spec(path: Path) -> Spec:
    spec_dict = yaml.safe_load(path.read_text())
    if not isinstance(spec_dict, dict):
        raise ValueError("Spec file must be a mapping")

    docker_cfg = spec_dict.get("docker", {})
    scripts_cfg = spec_dict.get("prework", {}).get("scripts", [])
    inference_cfg = spec_dict.get("inference", {})
    patch_cfg = spec_dict.get("patch", {})

    resolved_scripts = []
    for raw_script in scripts_cfg:
        template_path = ensure_path(path.parent, raw_script["template"])
        resolved_scripts.append(
            ScriptStep(
                name=raw_script.get("name", template_path.stem),
                template=template_path,
                destination=raw_script.get(
                    "destination", f"/tmp/{template_path.stem}.sh"
                ),
                execute=raw_script.get("execute", True),
                command=raw_script.get("command"),
                shell=raw_script.get("shell", "/bin/bash"),
                chmod_x=raw_script.get("chmod_x", True),
                continue_on_error=raw_script.get("continue_on_error", False),
                env=raw_script.get("env", {}),
                timeout_sec=raw_script.get("timeout_sec"),
            )
        )

    inference_cmd = inference_cfg.get("command")
    if not inference_cmd:
        raise ValueError("Spec missing inference.command")

    inference = InferenceCommand(
        command=inference_cmd,
        env=inference_cfg.get("env", {}),
        workdir=inference_cfg.get("workdir"),
        shell=inference_cfg.get("shell", "/bin/bash"),
        timeout_sec=inference_cfg.get("timeout_sec"),
        log_name=inference_cfg.get("log_name", "inference.log"),
    )

    patch = PatchSpec(
        command=patch_cfg.get("command", DEFAULT_PATCH_COMMAND),
        container_path=patch_cfg.get("container_path", DEFAULT_PATCH_CONTAINER_PATH),
        host_filename=patch_cfg.get("host_filename", "patch.diff"),
        shell=patch_cfg.get("shell", "/bin/bash"),
        timeout_sec=patch_cfg.get("timeout_sec"),
    )

    artifacts = spec_dict.get("artifacts", [])

    return Spec(
        name=spec_dict.get("name", path.stem),
        description=spec_dict.get("description", ""),
        docker_workdir=docker_cfg.get("workdir", "/testbed"),
        docker_user=docker_cfg.get("user", "root"),
        image_template=docker_cfg.get("image_template", DEFAULT_SPEC_TEMPLATE),
        scripts=resolved_scripts,
        inference=inference,
        patch=patch,
        artifacts=artifacts,
        variables=spec_dict.get("variables", {}),
    )


### CPU assignment logic copied from swefficiency/harness/cpu_assignment.py

import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple

SYS_CPU = "/sys/devices/system/cpu"


def _parse_cpu_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def _cpu_to_node(cpu: int) -> int:
    d = f"{SYS_CPU}/cpu{cpu}"
    try:
        entries = [e for e in os.listdir(d) if e.startswith("node")]
        return int(entries[0].replace("node", "")) if entries else 0
    except FileNotFoundError:
        return 0


def discover_physical_cores() -> List[Tuple[List[int], int]]:
    """
    Returns a list of (siblings, numa_node) for each physical core.
    `siblings` is a sorted list of all logical CPUs that share that core (SMT threads).
    """
    seen = set()
    cores = []
    for entry in os.listdir(SYS_CPU):
        if not entry.startswith("cpu") or not entry[3:].isdigit():
            continue
        cpu = int(entry[3:])
        if cpu in seen:
            continue
        topo = f"{SYS_CPU}/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(topo) as f:
                sibs = sorted(set(_parse_cpu_list(f.read().strip())))
        except FileNotFoundError:
            sibs = [cpu]
        for c in sibs:
            seen.add(c)
        node = _cpu_to_node(sibs[0])
        cores.append((sibs, node))
    # stable ordering by (node, first-sib)
    cores.sort(key=lambda t: (t[1], t[0][0]))
    return cores


def allocate_whole_cores(
    num_workers: int,
    vcpus_per_worker: int = 4,
    threads_per_core: int = 2,
    reserve_cores: int = 0,  # optionally keep some cores unassigned (for OS/IRQs)
):
    """
    Allocate to each worker a set of vCPUs built from WHOLE physical cores.
    - No physical core is shared across workers.
    - If threads_per_core == 2, each assigned core contributes both SMT threads (e.g., 0,32).
      So vcpus_per_worker must be divisible by threads_per_core.
    - Returns: [{worker, cpuset_cpus, cpuset_mems, nano_cpus}]
    """
    if num_workers <= 0 or vcpus_per_worker <= 0:
        raise ValueError("num_workers and vcpus_per_worker must be > 0")
    if threads_per_core not in (1, 2):
        raise ValueError("threads_per_core must be 1 or 2")
    if vcpus_per_worker % threads_per_core != 0:
        raise ValueError("vcpus_per_worker must be divisible by threads_per_core")

    cores_needed_per_worker = vcpus_per_worker // threads_per_core

    cores = discover_physical_cores()
    if reserve_cores > 0:
        # drop the first N cores (ordered by (node, cpu)) to keep for the host
        cores = cores[reserve_cores:]

    total_cores = len(cores)
    need_cores = num_workers * cores_needed_per_worker
    if total_cores < need_cores:
        raise RuntimeError(
            f"Not enough physical cores: need {need_cores}, have {total_cores}"
        )

    # Bucket cores by NUMA node
    by_node: Dict[int, deque] = defaultdict(deque)
    for sibs, node in cores:
        by_node[node].append((sibs, node))

    plans = [{"worker": i, "cores": [], "nodes": set()} for i in range(num_workers)]

    # Phase 1: pack whole workers inside a single node when possible
    w = 0
    for node in sorted(by_node.keys()):
        q = by_node[node]
        while (
            len(q) >= cores_needed_per_worker
            and w < num_workers
            and len(plans[w]["cores"]) == 0
        ):
            take = [q.popleft()[0] for _ in range(cores_needed_per_worker)]
            plans[w]["cores"].extend(take)
            plans[w]["nodes"].add(node)
            w += 1
            if w >= num_workers:
                break

    # Phase 2: fill remaining workers round-robin from whatever cores remain (may span nodes)
    remaining = []
    for node in sorted(by_node.keys()):
        remaining.extend(list(by_node[node]))
    pending = [
        i
        for i in range(num_workers)
        if len(plans[i]["cores"]) < cores_needed_per_worker
    ]
    p = 0
    for sibs, node in remaining:
        if not pending:
            break
        i = pending[p]
        plans[i]["cores"].append(sibs)
        plans[i]["nodes"].add(node)
        if len(plans[i]["cores"]) == cores_needed_per_worker:
            pending.pop(p)
            if not pending:
                break
            p %= len(pending)
        else:
            p = (p + 1) % len(pending)

    # Finalize: expand each core into the desired number of threads (1 or 2)
    out = []
    for r in plans:
        cpus = []
        for core_sibs in r["cores"]:
            chosen = core_sibs[:threads_per_core]  # pick 1 or both siblings
            cpus.extend(chosen)
        cpus = sorted(cpus)
        mems = ",".join(str(n) for n in sorted(r["nodes"])) or "0"
        out.append(
            {
                "cpuset_cpus": ",".join(map(str, cpus)),
                "cpuset_mems": mems,
                "nano_cpus": int(1e9 * len(cpus)),  # optional: hard cap to vCPU count
            }
        )
    return out


def divide_cpus_among_workers(num_workers, cpus_per_worker=4):
    cpu_groups = []

    for i in range(num_workers):
        cpu_groups.append(list(range(i * cpus_per_worker, (i + 1) * cpus_per_worker)))

    return cpu_groups


def _tar_bytes(path: Path, data: bytes, mode: int = 0o644) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tarinfo = tarfile.TarInfo(name=path.name)
        tarinfo.size = len(data)
        tarinfo.mode = mode
        tarinfo.mtime = int(time.time())
        tar.addfile(tarinfo, io.BytesIO(data))
    buf.seek(0)
    return buf.read()


def copy_text_to_container(
    container, text: str, dest_path: str, mode: int = 0o644
) -> None:
    dest = Path(dest_path)
    container.exec_run(f"mkdir -p {shlex.quote(str(dest.parent))}")
    archive = _tar_bytes(dest, text.encode("utf-8"), mode=mode)
    container.put_archive(str(dest.parent), archive)


def copy_from_container(container, src_path: str, dest_path: Path) -> None:
    stream, _ = container.get_archive(src_path)
    buf = io.BytesIO()
    for chunk in stream:
        buf.write(chunk)
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        member = tar.next()
        if not member:
            raise HarnessError(f"No data returned when copying {src_path}")
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise HarnessError(f"Failed extracting {src_path} from container")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(file_obj.read())


def run_exec(
    container,
    command: str,
    *,
    shell: Optional[str],
    env: Optional[Dict[str, str]],
    workdir: Optional[str],
    user: str,
    log_file: Path,
    stream: bool,
    label: str,
    tee_log: Optional[Path] = None,
) -> int:
    if shell is None:
        final_command = command
    else:
        final_command = f"{shell} -lc {shlex.quote(command)}"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    extra_handles: List[io.TextIOBase] = []
    with contextlib.ExitStack() as stack:
        primary = stack.enter_context(log_file.open("a", encoding="utf-8"))
        extra_handles.append(primary)
        if tee_log:
            tee_log.parent.mkdir(parents=True, exist_ok=True)
            if tee_log != log_file:
                extra_handles.append(
                    stack.enter_context(tee_log.open("a", encoding="utf-8"))
                )

        def _write_all(text: str) -> None:
            for handle in extra_handles:
                handle.write(text)
                handle.flush()

        _write_all(f"$ {command}\n")

        def _emit(chunk: bytes, dest) -> None:
            text = chunk.decode("utf-8", errors="replace")
            _write_all(text)
            prefix = f"[{label}] " if label else ""
            dest.write(f"{prefix}{text}")
            dest.flush()

        exit_code: Optional[int]
        if stream:
            low_level = container.client.api
            exec_create = low_level.exec_create(
                container.id,
                cmd=final_command,
                environment=env,
                workdir=workdir,
                user=user,
            )
            exec_id = exec_create["Id"]
            output_stream = low_level.exec_start(exec_id, stream=True, demux=True)
            for stdout_chunk, stderr_chunk in output_stream:
                if stdout_chunk:
                    _emit(stdout_chunk, sys.stdout)
                if stderr_chunk:
                    _emit(stderr_chunk, sys.stderr)
            exit_code = low_level.exec_inspect(exec_id).get("ExitCode")
        else:
            result = container.exec_run(
                final_command,
                environment=env,
                workdir=workdir,
                user=user,
                demux=True,
            )
            stdout, stderr = result.output
            if stdout:
                _write_all(stdout.decode("utf-8", errors="replace"))
            if stderr:
                _write_all(stderr.decode("utf-8", errors="replace"))
            exit_code = result.exit_code

        _write_all(f"\nexit_code={exit_code}\n")
    return exit_code if exit_code is not None else -1


def ensure_clean_worktree(
    container,
    repo_path: str,
    *,
    user: str,
    log_file: Path,
    stream: bool,
    label: str,
    tee_log: Optional[Path] = None,
) -> None:
    quoted_repo = shlex.quote(repo_path)
    command = (
        f"cd {quoted_repo} && "
        "STATUS=$(git status --porcelain --untracked-files=all) && "
        'if [ -n "$STATUS" ]; then '
        "echo 'Dirty working tree detected before inference:'; "
        'echo "$STATUS"; '
        "exit 1; "
        "fi"
    )
    exit_code = run_exec(
        container,
        command,
        shell="/bin/bash",
        env=None,
        workdir=repo_path,
        user=user,
        log_file=log_file,
        stream=stream,
        label=label,
        tee_log=tee_log,
    )
    if exit_code != 0:
        raise HarnessError(
            "Testbed repository has uncommitted changes before execution; rebuild the image"
        )


def process_instance(
    *,
    instance: dict,
    spec: Spec,
    context_vars: Dict[str, Any],
    log_root: Path,
    docker_client,
    cpu_assignment: Optional[Dict[str, Any]],
    resource_limits: Dict[str, Any],
    pull_missing_images: bool,
    run_id: str,
    remove_container: bool,
    remove_image: bool,
    stream_logs: bool,
) -> dict:
    instance_id = instance["instance_id"]
    log_dir = log_root / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    aggregate_log = log_dir / "container.log"

    image_name = spec.image_template.format(instance_id=instance_id)
    if pull_missing_images:
        docker_client.images.pull(image_name)

    template_context = {
        "instance": instance,
        "instance_id": instance_id,
        "spec": spec,
        "vars": context_vars,
    }
    patch_host_name = render_inline(spec.patch.host_filename, template_context)
    patch_host_path = log_dir / patch_host_name
    if patch_host_path.exists():
        print(
            f"[SKIP] {instance_id}: existing patch at {patch_host_path}, skipping",
            flush=True,
        )
        return {
            "instance_id": instance_id,
            "status": "skipped",
            "patch": str(patch_host_path),
        }

    container_name = f"inference.{run_id}.{instance_id}"
    per_container_limits = dict(resource_limits)
    if cpu_assignment:
        for key in ("cpuset_cpus", "cpuset_mems", "nano_cpus"):
            value = cpu_assignment.get(key)
            if value is not None:
                per_container_limits[key] = value

    create_kwargs = {
        "name": container_name,
        "image": image_name,
        "user": spec.docker_user,
        "command": "tail -f /dev/null",
        "detach": True,
        "tty": False,
        "working_dir": spec.docker_workdir,
    }
    create_kwargs.update(per_container_limits)

    container = docker_client.containers.create(**create_kwargs)
    container.start()

    try:
        # ensure_clean_worktree(
        #     container,
        #     spec.docker_workdir,
        #     user=spec.docker_user,
        #     log_file=log_dir / "preflight.log",
        #     stream=stream_logs,
        #     label=f"{instance_id}:preflight",
        #     tee_log=aggregate_log,
        # )

        # Prework scripts
        for script_step in spec.scripts:
            rendered = render_template(script_step.template, template_context)
            mode = 0o755 if script_step.chmod_x else 0o644
            copy_text_to_container(container, rendered, script_step.destination, mode)
            if script_step.execute:
                if script_step.command:
                    command = render_inline(script_step.command, template_context)
                else:
                    command = f"{script_step.shell} {script_step.destination}"
                log_file = log_dir / f"{script_step.name}.log"
                step_env = render_env(script_step.env, template_context)
                exit_code = run_exec(
                    container,
                    command,
                    shell=None,
                    env=step_env,
                    workdir=spec.docker_workdir,
                    user=spec.docker_user,
                    log_file=log_file,
                    stream=stream_logs,
                    label=f"{instance_id}:{script_step.name}",
                    tee_log=aggregate_log,
                )
                if exit_code != 0 and not script_step.continue_on_error:
                    raise HarnessError(
                        f"Prework script {script_step.name} failed with code {exit_code}"
                    )

        # Inference command
        inference_log = log_dir / spec.inference.log_name
        inference_command = render_inline(spec.inference.command, template_context)
        inference_env = render_env(spec.inference.env, template_context)
        exit_code = run_exec(
            container,
            inference_command,
            shell=spec.inference.shell,
            env=inference_env,
            workdir=spec.inference.workdir or spec.docker_workdir,
            user=spec.docker_user,
            log_file=inference_log,
            stream=stream_logs,
            label=f"{instance_id}:inference",
            tee_log=aggregate_log,
        )
        if exit_code != 0:
            raise HarnessError(f"Inference command failed with code {exit_code}")

        # Patch extraction
        patch_command = render_inline(spec.patch.command, template_context)
        patch_log = log_dir / "patch.log"
        exit_code = run_exec(
            container,
            patch_command,
            shell=spec.patch.shell,
            env=None,
            workdir=spec.docker_workdir,
            user=spec.docker_user,
            log_file=patch_log,
            stream=stream_logs,
            label=f"{instance_id}:patch",
            tee_log=aggregate_log,
        )
        if exit_code != 0:
            raise HarnessError("Patch extraction command failed")

        patch_container_path = render_inline(
            spec.patch.container_path, template_context
        )
        copy_from_container(container, patch_container_path, patch_host_path)

        # Additional artifacts
        for artifact in spec.artifacts:
            container_path_template = artifact.get("container_path")
            if not container_path_template:
                raise HarnessError("Artifact entry missing container_path")
            container_path = render_inline(container_path_template, template_context)
            host_template = artifact.get("host_filename")
            host_name = (
                render_inline(host_template, template_context)
                if host_template
                else Path(container_path).name
            )
            copy_from_container(container, container_path, log_dir / host_name)

        return {
            "instance_id": instance_id,
            "status": "success",
            "patch": str(patch_host_path),
        }
    finally:
        must_remove_container = remove_container or remove_image

        if must_remove_container:
            with contextlib.suppress(Exception):
                container.stop(timeout=5)
            with contextlib.suppress(Exception):
                container.remove(force=True)

        if remove_image:
            with contextlib.suppress(Exception):
                docker_client.images.remove(image=image_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="swefficiency/swefficiency", help="HF dataset name"
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--instance-ids", nargs="*", help="Explicit instance IDs to run"
    )
    parser.add_argument("--instance-regex", help="Regex to filter instance IDs")
    parser.add_argument(
        "--max-instances", type=int, help="Optional cap on number of instances"
    )
    parser.add_argument(
        "--spec", type=Path, required=True, help="Path to YAML spec file"
    )
    parser.add_argument("--run-id", required=True, help="Run identifier (used in logs)")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_LOG_DIR, help="Where to store logs"
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Concurrent workers")
    parser.add_argument(
        "--cpus-per-worker", type=int, default=4, help="Logical CPUs per worker"
    )
    parser.add_argument(
        "--threads-per-core",
        type=int,
        default=2,
        help="SMT threads per physical core (1 for disabled SMT, 2 for Hyper-Threading)",
    )
    parser.add_argument(
        "--reserve-cores",
        type=int,
        default=0,
        help="Number of physical cores to keep free for the host",
    )
    parser.add_argument(
        "--disable-cpu-pinning",
        action="store_true",
        help="Skip allocate_whole_cores and let Docker schedule CPUs automatically",
    )
    parser.add_argument("--mem-limit", default="32g", help="Docker mem_limit")
    parser.add_argument(
        "--mem-reservation", default="16g", help="Docker mem_reservation"
    )
    parser.add_argument("--memswap", default="32g", help="Docker memswap limit")
    parser.add_argument("--nano-cpus", type=int, help="Docker nano_cpus value")
    parser.add_argument(
        "--pull-missing-images",
        dest="pull_missing_images",
        action="store_true",
        default=True,
        help="docker pull image tags before launching containers (default: on)",
    )
    parser.add_argument(
        "--no-pull",
        dest="pull_missing_images",
        action="store_false",
        help="skip docker pull (assume images are available locally)",
    )
    parser.add_argument(
        "--keep-containers",
        action="store_true",
        help="Leave inference containers running after completion (default: clean up)",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Do not remove instance images after completion (default: remove)",
    )
    parser.add_argument(
        "--stream-logs",
        action="store_true",
        help="Mirror container stdout/stderr to the console while writing log files",
    )
    parser.add_argument(
        "--vars-file", type=Path, help="YAML/JSON file with template variables"
    )
    parser.add_argument(
        "--var", action="append", default=[], help="Inline template vars KEY=VALUE"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List selected instances and exit"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)

    debug_mode = os.environ.get("DEBUG") == "1"

    user_vars = spec.variables.copy()
    user_vars.update(load_vars_file(args.vars_file))
    user_vars.update(parse_kv_pairs(args.var))
    if "cursor_api_key" not in user_vars:
        env_cursor_key = os.environ.get("CURSOR_API_KEY")
        if env_cursor_key:
            user_vars["cursor_api_key"] = env_cursor_key

    regex = re.compile(args.instance_regex) if args.instance_regex else None
    id_filter = set(args.instance_ids) if args.instance_ids else None

    instances = filter_instances(
        _iter_dataset(args.dataset, args.split),
        id_filter,
        regex,
        args.max_instances,
    )

    if not instances:
        parser.error("No instances matched the provided filters")

    if args.dry_run:
        print("Selected instances:")
        for inst in instances:
            print(inst["instance_id"])
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = args.output_dir / args.run_id / spec.name
    run_log_dir.mkdir(parents=True, exist_ok=True)

    cpu_assignments: List[Optional[Dict[str, Any]]] = [None] * args.num_workers
    if not args.disable_cpu_pinning:
        try:
            cpu_assignments = allocate_whole_cores(
                args.num_workers,
                vcpus_per_worker=args.cpus_per_worker,
                threads_per_core=args.threads_per_core,
                reserve_cores=args.reserve_cores,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[WARN] Falling back to no CPU pinning:",
                exc,
                file=sys.stderr,
            )
            cpu_assignments = [None] * args.num_workers
    resource_limits = {
        "mem_limit": args.mem_limit,
        "mem_reservation": args.mem_reservation,
        "memswap_limit": args.memswap,
    }
    if args.nano_cpus:
        resource_limits["nano_cpus"] = args.nano_cpus

    docker_client = docker.from_env(timeout=3600)

    remove_container = not args.keep_containers
    remove_image = not args.keep_images
    auto_stream_logs = debug_mode and args.num_workers == 1
    stream_logs = args.stream_logs or auto_stream_logs
    if auto_stream_logs and not args.stream_logs:
        print(
            "[DEBUG] DEBUG=1 detected with a single worker; streaming container logs."
        )

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        future_map = {}
        for idx, instance in enumerate(instances):
            cpu_assignment = (
                cpu_assignments[idx % len(cpu_assignments)] if cpu_assignments else None
            )
            future = executor.submit(
                process_instance,
                instance=instance,
                spec=spec,
                context_vars=user_vars,
                log_root=run_log_dir,
                docker_client=docker_client,
                cpu_assignment=cpu_assignment,
                resource_limits=resource_limits,
                pull_missing_images=args.pull_missing_images,
                run_id=args.run_id,
                remove_container=remove_container,
                remove_image=remove_image,
                stream_logs=stream_logs,
            )
            future_map[future] = instance["instance_id"]

        for future in concurrent.futures.as_completed(future_map):
            instance_id = future_map[future]
            try:
                result = future.result()
                print(f"[OK] {instance_id}: patch saved to {result['patch']}")
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR] {instance_id}: {exc}", file=sys.stderr)
                results.append(
                    {"instance_id": instance_id, "status": "error", "error": str(exc)}
                )

    summary_path = run_log_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
