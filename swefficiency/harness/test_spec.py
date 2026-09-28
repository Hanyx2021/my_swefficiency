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

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

from swefficiency.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    MAP_REPO_TO_INSTALL,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    USE_X86,
    SWEfficiencyInstance,
)
from swefficiency.harness.dockerfiles import (
    get_dockerfile_annotate_instance,
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)
from swefficiency.harness.utils import (
    get_environment_yml,
    get_requirements,
    get_test_directives,
)
from swefficiency.perf_filter import utils as perf_utils

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """

    instance_id: str
    repo: str
    version: str
    repo_script_list: list[str]
    eval_script_list: list[str]
    env_script_list: list[str]
    arch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    base_commit: str

    coverage_script_list: list[str]
    meaningful_edit_script_list: list[str]
    performance_script_list: list[str]
    performance_profiling_script_list: list[str]
    correctness_script_list: list[str]
    introspection_guard_script_list: list[str]

    workload: Optional[str]
    covering_tests: Optional[list[str]]
    single_thread_tests: Optional[list[str]]

    build_timeout: Optional[int]

    @property
    def setup_env_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.env_script_list)
            + "\n"
        )

    @property
    def eval_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -xo pipefail"] + self.eval_script_list)
            + "\n"
        )
        # Don't exit early because we need to revert tests at the end

    @property
    def install_repo_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.repo_script_list)
            + "\n"
        )

    @property
    def ast_meaningful_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -xo pipefail"] + self.meaningful_edit_script_list
            )
            + "\n"
        )

    @property
    def coverage_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -xo pipefail"] + self.coverage_script_list)
            + "\n"
        )

    @property
    def performance_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -exo pipefail"] + self.performance_script_list
            )
            + "\n"
        )

    @property
    def performance_profiling_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -exo pipefail"]
                + self.performance_profiling_script_list
            )
            + "\n"
        )

    @property
    def correctness_script(self):
        # TODO: Check whether we should add -e here?
        return (
            "\n".join(
                ["#!/bin/bash", "set -xo pipefail"] + self.correctness_script_list
            )
            + "\n"
        )

    @property
    def introspection_guard_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -exo pipefail"]
                + self.introspection_guard_script_list
            )
            + "\n"
        )

    @property
    def base_image_key(self):
        return f"sweb.base.{self.arch}:latest"

    @property
    def env_image_key(self):
        """
        The key for the environment image is based on the hash of the environment script list.
        If the environment script list changes, the image will be rebuilt automatically.

        Note that old images are not automatically deleted, so consider cleaning up old images periodically.
        """
        hash_object = hashlib.sha256()
        # Normalize env_command to "conda" for hash calculation to avoid hash changes
        # when switching between conda and mamba for acceleration
        normalized_script_list = str(self.env_script_list).replace("mamba", "conda")
        hash_object.update(normalized_script_list.encode("utf-8"))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 characters is still very likely to be unique
        return f"sweb.env.{self.arch}.{val}:latest"

    @property
    def instance_image_key(self):
        return f"sweb.eval.{self.arch}.{self.instance_id}:latest"

    @property
    def annotate_instance_image_key(self):
        return f"sweb.eval.{self.arch}.{self.instance_id}.annotate:latest"

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"sweb.eval.{self.instance_id}"
        return f"sweb.eval.{self.instance_id}.{run_id}"

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(self.platform, self.arch)

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(self.platform, self.arch)

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.env_image_key)

    @property
    def annotate_instance_dockerfile(self):
        return get_dockerfile_annotate_instance(self.platform, self.instance_image_key)

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")


def get_test_specs_from_dataset(
    dataset: Union[list[SWEfficiencyInstance], list[TestSpec]],
) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEfficiencyInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)

    test_specs = []
    for instance in dataset:
        try:
            test_specs.append(make_test_spec(instance))
        except NotImplementedError as e:
            continue
        except Exception as e:
            print(
                f"Error creating test spec for instance {instance[KEY_INSTANCE_ID]} for version {instance['version']}: {e}"
            )
            traceback.print_exc()

    return test_specs


# ---------------------------------------------------------------------------
# github.com mirror rotation
# ---------------------------------------------------------------------------
# Ordered mirror prefixes; each is prepended verbatim to "<owner>/<repo>", so
# both prefix styles work:
#     https://hk.gh-proxy.com/https://github.com/   (proxy style)
#     https://gitclone.com/github.com/              (gitclone style)
# Measured from this network on 2026-09-15 (real `git clone --single-branch` of
# pandas-dev/pandas, the heaviest repo in the suite, ~457MB):
#     gh-proxy.com     43s  ok, 38993 commits, base_commit + tags fetchable
#     ghproxy.net     300s+ stalls; fine for small repos (pytest 140s), not pandas
#     hk.gh-proxy.com 300s+ hangs after "redirecting to hk.gh-proxy.org/https:/
#                     github.com/..." (note the mangled single slash) or 503
#     gitclone.com    151s *exit 0* but degenerate: 1 commit, 0 tags, and the
#                     requested base_commit simply absent -- the rotation used to
#                     accept that clone and die later at `git reset --hard`
#     ghfast.top      dead since 2026-08 (403)
#     github.com      blocked (TCP connect times out after ~135s)
# Mirror health rotates on a scale of hours, so the list is ordered but never
# trusted: a clone only counts once base_commit is checked out (see the rotation
# below).  Override with SWEFFICIENCY_GITHUB_MIRRORS="<p1> <p2> ..." (space or
# comma separated).  The legacy single-value SWEFFICIENCY_GITHUB_MIRROR is still
# honoured as the primary mirror.  ROUNDS / TIMEOUT / BUDGET tune the rotation.
_DEFAULT_GITHUB_MIRRORS = [
    "https://gh-proxy.com/https://github.com/",
    "https://ghproxy.net/https://github.com/",
    "https://hk.gh-proxy.com/https://github.com/",
    "https://gitclone.com/github.com/",
]
_LEGACY_MIRROR = "https://ghfast.top/https://github.com/"


def get_github_mirrors() -> list[str]:
    """Ordered github.com mirror prefixes; entry 0 is the preferred mirror."""
    raw = os.environ.get("SWEFFICIENCY_GITHUB_MIRRORS", "").strip()
    if not raw:
        # Back-compat: the old single-value variable names the primary mirror.
        raw = os.environ.get("SWEFFICIENCY_GITHUB_MIRROR", "").strip()
    mirrors = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
    if not mirrors:
        mirrors = list(_DEFAULT_GITHUB_MIRRORS)
    # Tolerate prefixes written without a trailing slash.
    return [m if m.endswith("/") else m + "/" for m in mirrors]


def _env_nonneg_int(name: str, default: int) -> int:
    """int(os.environ[name]) when it is a non-negative int, else *default*."""
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


# Width of the maths-library thread pools during the correctness run.  1 makes
# the run deterministic; 0 leaves the environment alone (pre-2026-09-16
# behaviour, threads = nproc).
_DEFAULT_CORRECTNESS_NUM_THREADS = 1

# Honoured by the OpenMP runtime, NumPy/OpenBLAS, MKL and numexpr respectively.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def get_correctness_thread_env() -> str:
    """Shell env-assignment prefix that pins the maths thread pools, or "".

    The correctness script already parallelises across test *files* (``xargs
    -P``); letting every pytest process fan out to all cores on top of that
    only oversubscribes the machine, and it makes the run non-reproducible.
    Several suites assert on *exact* floating-point convergence, and parallel
    BLAS/OpenMP reductions accumulate in a thread-count-dependent order, so a
    256-core host can fail a test that a small host passes.  scikit-learn's
    ``test_kmeans_convergence`` is one: the KMeans E-step is an OpenMP
    ``prange`` reduction over ``n_samples // 256`` chunks, and the test asserts
    ``n_iter_ < max_iter`` with ``tol=0`` -- i.e. the centre shift must hit
    *exactly* 0.0.  That happens with 1 thread and never with 256.

    Override with ``SWEFFICIENCY_CORRECTNESS_NUM_THREADS`` (0 disables the
    pinning entirely).
    """
    n_threads = _env_nonneg_int(
        "SWEFFICIENCY_CORRECTNESS_NUM_THREADS", _DEFAULT_CORRECTNESS_NUM_THREADS
    )
    if n_threads == 0:
        return ""
    return " ".join(f"{var}={n_threads}" for var in _THREAD_ENV_VARS) + " "


def make_repo_script_list(
    specs, repo, repo_directory, base_commit, env_name, treesitter_env_name=None
):
    """
    Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.
    """
    # Clone via an ordered mirror list: github.com is unreachable from this
    # network and single mirrors rot (ghfast.top died 2026-08, gh-proxy.com died
    # 2026-09-15).  One full pass over the list is one "round"; first success wins.
    _clone_mirrors = get_github_mirrors()
    _clone_primary = _clone_mirrors[0]
    _clone_rounds = max(1, _env_nonneg_int("SWEFFICIENCY_GITHUB_MIRROR_ROUNDS", 2))
    _attempt_timeout = _env_nonneg_int("SWEFFICIENCY_GITHUB_MIRROR_TIMEOUT", 600)
    # scikit-learn's build is capped at 600s (BUILD_TIMEOUT_CONFIGS_OVERRIDES),
    # so the clone phase must never eat the whole build timeout.
    _build_timeout = BUILD_TIMEOUT_CONFIGS_OVERRIDES.get(repo)
    _clone_budget = _env_nonneg_int(
        "SWEFFICIENCY_GITHUB_MIRROR_BUDGET",
        (_build_timeout // 2) if _build_timeout else 1800,
    )
    # Post-clone fetches (--tags, base_commit) sit *outside* the clone rotation,
    # so a mirror that accepts the clone and then stalls on the follow-up fetch
    # is never killed.  Production hit exactly that on 2026-09-15: a
    # `git fetch origin --tags` still alive after 3h24m.  0 disables the wrapper.
    _fetch_timeout = _env_nonneg_int("SWEFFICIENCY_GITHUB_FETCH_TIMEOUT", 600)
    # Bash reference to the mirror that actually served the clone.  Repo specs
    # hardcode a *single* mirror prefix for `git remote add upstream`, submodule
    # url.<base>.insteadOf and wget; rewriting those to this variable means a
    # dead *primary* mirror can no longer poison a build whose clone already
    # succeeded on a secondary one -- the exact failure mode seen on 2026-09-15
    # (hk.gh-proxy.com returning 503 while ghproxy.net served fine).  Quoted so
    # it drops into `url.<ref>.insteadOf`, `git remote add` and `wget` alike.
    _mirror_ref = '"${_SWEFF_MIRROR_USED}"'
    _rounds_literal = " ".join(str(i) for i in range(1, _clone_rounds + 1))
    setup_commands = [
        "git config --global http.lowSpeedLimit 0",
        "git config --global http.postBuffer 524288000",
        # github.com is unreachable and single mirrors rot, so rotate over an
        # ordered mirror list; one pass over the list is one round.
        # NOTE: setup_repo.sh runs under `set -exo pipefail`, so every command
        # that may fail is kept in an `if` condition (which is exempt from -e).
        f"export _SWEFF_MIRRORS='{' '.join(_clone_mirrors)}'",
        # Mirror the clone ends up using.  Defaults to the primary so that any
        # command emitted *before* the loop still gets a well-formed URL; the
        # success branch below overwrites it with the mirror that really worked.
        # Exported so nested shells (submodule, pip) inherit it.
        '_SWEFF_MIRROR_USED="${_SWEFF_MIRRORS%% *}"',
        "export _SWEFF_MIRROR_USED",
        f"_SWEFF_ROUNDS={_clone_rounds}",
        f"_SWEFF_ATTEMPT_TIMEOUT={_attempt_timeout}",
        f"_SWEFF_BUDGET={_clone_budget}",
        "_SWEFF_OK=''",
        "# timeout(1) is coreutils and present in ubuntu:22.04; probe anyway so a",
        "# missing binary degrades to an untimed clone instead of a 127 per attempt.",
        "_SWEFF_TIMEOUT_BIN=''",
        "if command -v timeout >/dev/null 2>&1; then _SWEFF_TIMEOUT_BIN=timeout; fi",
        "_SWEFF_STARTED=$(date +%s 2>/dev/null || echo 0)",
        'if [ -z "$_SWEFF_MIRRORS" ]; then',
        '  echo "SWEFF: FATAL no github mirrors configured"',
        "  exit 128",
        "fi",
        # /testbed is this RUN's cwd (inherited WORKDIR); leave it before removing
        # it, otherwise getcwd() fails inside git.
        "cd /",
        f"for _SWEFF_R in {_rounds_literal}; do",
        "  for _SWEFF_M in $_SWEFF_MIRRORS; do",
        '    if [ "$_SWEFF_STARTED" -gt 0 ] && [ "$_SWEFF_BUDGET" -gt 0 ]; then',
        "      _SWEFF_ELAPSED=$(( $(date +%s 2>/dev/null || echo 0) - _SWEFF_STARTED ))",
        '      if [ "$_SWEFF_ELAPSED" -ge "$_SWEFF_BUDGET" ]; then',
        '        echo "SWEFF: clone budget ${_SWEFF_BUDGET}s exhausted; giving up"',
        "        break 2",
        "      fi",
        "    fi",
        "# Assemble the wrapper as ONE word list: an empty expansion must vanish",
        "# entirely (never leave the timeout value as a command name).",
        '    _SWEFF_RUN=""',
        '    if [ -n "$_SWEFF_TIMEOUT_BIN" ]; then _SWEFF_RUN="timeout $_SWEFF_ATTEMPT_TIMEOUT"; fi',
        '    echo "SWEFF: clone attempt round=$_SWEFF_R/$_SWEFF_ROUNDS mirror=$_SWEFF_M"',
        # A failed clone removes the directory it created, but a SIGKILLed one
        # (timeout/oom) cannot, so wipe the target every attempt.  `|| true`: a
        # failing rm must not abort the build; the clone then fails visibly with
        # "not an empty directory" and rotates to the next mirror.
        f"    rm -rf {repo_directory} || true",
        f'    if $_SWEFF_RUN git clone --single-branch -o origin "${{_SWEFF_M}}{repo}" {repo_directory}; then',
        # A mirror can hand back a *degenerate* clone and still exit 0.  Measured
        # on 2026-09-15: gitclone.com returned rc=0 in 151s for pandas with
        # commits=1, tags=0 and base_commit absent -- the rotation then declared
        # success, and the build died much later at `git reset --hard` with
        # `exit status 128` while `git fetch origin --tags` on that stub hung for
        # 3h24m.  So a clone is only "OK" once the commit we actually need is
        # reachable; otherwise this mirror is a failure and we rotate on.
        # `git -C` rather than cd keeps every command off the error path of a
        # failing cwd change under `set -e`.
        f"      for _SWEFF_T in 1 2 3; do $_SWEFF_RUN git -C {repo_directory} fetch origin {base_commit} 2>/dev/null && break; git -C {repo_directory} config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*' 2>/dev/null || true; $_SWEFF_RUN git -C {repo_directory} fetch origin 2>/dev/null || true; sleep 5; done",
        # Verify by performing the operation the build actually needs.  A commit
        # object can exist in a stunted clone while its tree/blobs do not, so
        # `cat-file -e` is not enough; a successful `reset --hard` proves the
        # whole path is present, and it is the exact step that used to fail with
        # `exit status 128` after a bogus "clone OK".
        f"      if git -C {repo_directory} reset --hard {base_commit} 2>/dev/null; then",
        "        _SWEFF_OK=yes",
        '        _SWEFF_MIRROR_USED="$_SWEFF_M"',
        '        echo "SWEFF: clone OK via $_SWEFF_M (base_commit checked out)"',
        "        break 2",
        "      fi",
        f'      echo "SWEFF: mirror $_SWEFF_M cloned but {base_commit} is not usable (degenerate clone); trying next"',
        "    else",
        '      echo "SWEFF: mirror $_SWEFF_M failed; trying next"',
        "    fi",
        "    sleep 5",
        "  done",
        "  sleep 15",
        "done",
        'if [ -z "$_SWEFF_OK" ]; then',
        f'  echo "SWEFF: FATAL could not clone {repo} from any of: $_SWEFF_MIRRORS"',
        "  exit 128",
        "fi",
        f"chmod -R 777 {repo_directory}",
        f"cd {repo_directory}",
        # `chmod -R 777` above rewrites the mode bits of every file in the
        # tree, so without this git reports the whole repo as modified
        # (`mode change 100644 => 100755`).  Those hunks then leak into every
        # later `git diff <base_commit>`: for scipy the runtime patch capture
        # produced a ~175 KB "patch" of nothing but mode changes, which blew
        # past the single-argument limit of the remote-write helper and
        # aborted the baseline correctness check (__INTROSPECTION_GUARD_FAILED__).
        # core.fileMode lives in .git/config, so it persists into the image and
        # also covers diffs taken at runtime.  Content changes are still
        # detected and still committed by the 'Fix environment' step below.
        "git config core.fileMode false",
        # Same one-word-list trick as _SWEFF_RUN above, but for the post-clone
        # fetches: an empty expansion must vanish rather than leave `600` as the
        # command name.  A mirror that stalls here used to hang the build for
        # hours because only the clone itself was wrapped in `timeout`.
        f"_SWEFF_FETCH_TIMEOUT={_fetch_timeout}",
        '_SWEFF_FETCH_RUN=""',
        'if [ -n "$_SWEFF_TIMEOUT_BIN" ] && [ "$_SWEFF_FETCH_TIMEOUT" -gt 0 ]; then'
        ' _SWEFF_FETCH_RUN="timeout $_SWEFF_FETCH_TIMEOUT"; fi',
        "$_SWEFF_FETCH_RUN git fetch origin --tags 2>/dev/null || true",
        # base_commit was already fetched and checked out inside the rotation
        # loop -- that is what makes a mirror's clone count as successful.
        "git remote remove origin",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]
    # Everything above is the clone/fetch preamble, whose `origin` URLs are
    # already mirror-correct.  The legacy-literal rewrite below must not touch
    # it: it would rewrite the mirror list itself if a legacy prefix ever
    # reappeared inside _SWEFF_MIRRORS.
    _preamble_len = len(setup_commands)

    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    # Run pre-install set up if provided
    if "pre_install" in specs:
        assert isinstance(
            specs["pre_install"], list
        ), "pre_install should be a list of commands"
        for pre_install in specs["pre_install"]:
            setup_commands.append(pre_install)

    # scipy checkouts from the 1.10.0.dev0 era (e.g. base_commit bd608529) ship
    # ``scipy/_build_utils/cythoner.py`` with a bare ``#!python3`` shebang.  The
    # kernel resolves a *relative* interpreter against the cwd of the exec'ing
    # process, and meson/ninja runs that helper with cwd set to the build
    # directory, where no ``python3`` exists: every ``.pyx`` -> ``.c`` step dies
    # with ``/bin/sh: 1: .../cythoner.py: not found`` (exit 127) and
    # ``pip install --no-build-isolation -e .`` aborts while "Preparing
    # metadata", so the instance image can never be built
    # (scipy__scipy-16599 / -16790 / -16840).  Rewrite only that shebang: the
    # file-content check makes the command a no-op for every other checkout, and
    # the 'Fix environment' commit below swallows the edit so that it can never
    # surface in an agent's diff.
    if repo == "scipy/scipy":
        setup_commands.append(
            "if [ -f /testbed/scipy/_build_utils/cythoner.py ] && "
            "head -1 /testbed/scipy/_build_utils/cythoner.py | grep -qx '#!python3'; then "
            "sed -i '1s|^#!python3$|#!/usr/bin/env python3|' "
            "/testbed/scipy/_build_utils/cythoner.py && "
            "chmod +x /testbed/scipy/_build_utils/cythoner.py; fi"
        )

    if "install" in specs:
        setup_commands.append(specs["install"])

    # SWE-fficiency: Add coverage, pytest-profiling, pytest-memray.
    setup_commands.append("python -m pip install --no-cache-dir coverage unidiff")

    # TODO: Add pytest-profiling, pytest-memray later.
    # Note: only sklearn requires interval tree, so we don't install it by default.
    setup_commands.append("python -m pip install --no-cache-dir jedi asttokens")

    # Tree-sitter is backwards compatible,
    if treesitter_env_name:
        # Ensure pip is available in the treesitter environment
        setup_commands.extend([
            f"conda activate {treesitter_env_name}",
            "python -m pip --version 2>/dev/null || conda install pip -y || true",  # Install pip if missing
            "python -m pip install tree-sitter tree_sitter_languages",
            f"conda activate {env_name}",
        ])

    # Add NumPy downgrade for astropy to avoid compatibility issues
    # NumPy 2.0 removed C API members (elsize, etc.) and changed Python-level
    # behaviour (copy=False). Old astropy source hits several of these.
    if "astropy" in repo:
        setup_commands.extend([
            "# Check if NumPy downgrade is needed for astropy",
            "if grep -q 'in1d' /testbed/astropy/units/quantity_helper/function_helpers.py"
            " || grep -rq 'elsize' /testbed/astropy/ --include='*.c' --include='*.h' 2>/dev/null"
            " || grep -rq 'copy=False' /testbed/astropy/ --include='*.py' 2>/dev/null; then",
            "    pip install 'numpy<2.0'",
            "fi",
        ])

    # There may be untracked changes, just try to commit all of them (ok if it fails).
    setup_commands.append(
        "git -c user.name='Automated Test' -c user.email='automated@test.com' commit -a -m 'Fix environment' || true"
    )

    # Normalise legacy hardcoded mirror literals.  MAP_REPO_TO_INSTALL /
    # specs["pre_install"] carry upstream-remote, submodule url.<base>.insteadOf
    # and wget literals that hardcode the once-default mirror prefix
    # (``https://ghfast.top/...``).  Those are appended verbatim above and bypass
    # the rotation list, so when ghfast.top 403s (dead since 2026-08) the build
    # dies at e.g. pandas' ``git fetch upstream --tags``.
    #
    # These sites are single-valued by nature (git config insteadOf, wget) and
    # cannot rotate, so they all point at $_SWEFF_MIRROR_USED -- the mirror that
    # actually served the clone, which may be any entry of the list.  Pointing
    # them at a fixed "primary" is what broke on 2026-09-15: a clone that
    # succeeded via ghproxy.net was then killed by `git fetch upstream --tags`
    # against a 503-ing hk.gh-proxy.com.
    #
    # The set/unset pairs inside _SUBMODULE_INIT_RETRY use the same literal, so a
    # blanket replace keeps them consistent.  No-op for a pure-legacy config.
    if _clone_primary != _LEGACY_MIRROR:
        _legacy_quoted = "'" + _LEGACY_MIRROR + "'"
        rewritten = []
        for cmd in setup_commands[_preamble_len:]:
            # Single-quoted sites (astropy's ``url.'<mirror>'.insteadOf``) would
            # keep a bare ${...} reference literal, so handle the quoted form
            # first: after this it is a double-quoted reference, which the second
            # replace below no longer matches.
            cmd = cmd.replace(_legacy_quoted, _mirror_ref)
            if cmd.strip() == "git fetch upstream --tags":
                # pandas' pre_install.  The only *unguarded* post-clone fetch in
                # the repo specs (its `--unshallow` sibling on the next line is
                # already `|| true`), and redundant with the rotation's own
                # `git fetch origin --tags` against the same mirror.  Unguarded
                # it turns one flaky mirror response into `exit status 128` that
                # throws away a perfectly good clone.
                cmd = (
                    f"$_SWEFF_FETCH_RUN {cmd}"
                    " || echo 'SWEFF: WARN upstream --tags fetch failed;"
                    " origin already supplied tags, continuing'"
                )
            rewritten.append(cmd.replace(_LEGACY_MIRROR, _mirror_ref))
        setup_commands = setup_commands[:_preamble_len] + rewritten
    return setup_commands


def replace_uninstallable_packages_requirements_txt(requirement_str: str) -> str:
    """Replaces certain packages in a requirements.txt-like string.
    For example, some packages have been yanked and we need to replace them with compatible alternatives.
    """
    replacements = {
        # See https://ghfast.top/https://github.com/princeton-nlp/SWE-bench/issues/199
        # This package was sinced yanked, so we need to force pip
        # to install it.
        # "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in requirement_str.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(
                f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)"
            )
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)
    return "\n".join(requirements_replaced) + "\n"


def make_env_script_list(
    instance: SWEfficiencyInstance, specs: dict, env_name: str, treesitter_env_name=None
) -> list[str]:
    """
    Creates the list of commands to set up the conda environment for testing.
    This is the setup script for the environment image.

    Returns:
        list[str]: List of commands to set up the conda environment
    """
    HEREDOC_DELIMITER = "EOF_59812759871"
    reqs_commands = [
        "source /opt/miniconda3/bin/activate",
    ]

    reqs_commands.extend(
        [
            # TODO: For safety, remove defaults?
            # "conda config --remove channels defaults",
            # "conda config --add channels conda-forge",
            # "conda config --set channel_priority strict",
            # "conda install -n base conda-libmamba-solver",
            # "conda config --set solver libmamba",
            # "conda config --set safety_checks disabled",
        ]
    )

    env_command = specs.get("env_command", "conda")

    # Create conda environment according to install instructinos
    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        # Create environment
        cmd = f"{env_command} create -n {env_name} python={specs['python']} pip -y"
        reqs_commands.append(cmd)

        # Install dependencies
        reqs = replace_uninstallable_packages_requirements_txt(
            get_requirements(instance)
        )
        path_to_reqs = f"$HOME/{pkgs}"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        cmd = f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        reqs_commands.append(cmd)
        reqs_commands.append(f"rm {path_to_reqs}")
    elif pkgs == "environment.yml":
        # Create environment from yml
        reqs = get_environment_yml(instance, env_name)
        path_to_reqs = "environment.yml"

        # Always remove defaults to avoid deprecated pkgs/free channel (404 errors)
        reqs_commands.append("conda config --remove channels defaults || true")

        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        if "no_use_env" in specs and specs["no_use_env"]:
            # `conda create` based installation
            cmd = f"{env_command} create -c conda-forge -n {env_name} python={specs['python']} -y"
            reqs_commands.append(cmd)

            # Install dependencies
            cmd = f"{env_command} env update -f {path_to_reqs}"
            reqs_commands.append(cmd)
        else:
            # `conda env create` based installation
            cmd = f"{env_command} env create -n {env_name} --file {path_to_reqs} -v"
            reqs_commands.append(cmd)

            if "python" in specs:
                cmd = f"conda activate {env_name} && conda install python={specs['python']} -y"
            else:
                cmd = f"conda activate {env_name}"
            reqs_commands.append(cmd)

        # Remove environment.yml
        reqs_commands.append(f"rm {path_to_reqs}")
    else:
        # Create environment + install dependencies
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        cmd = (
            f"{env_command} create -n {env_name} python={specs['python']} {pkgs} -y -v"
        )
        reqs_commands.append(cmd)

    reqs_commands.append("conda clean --all -y")  # Clean up conda cache to save space
    reqs_commands.append(f"conda activate {env_name}")
    
    # Ensure pip is installed in the environment before using it
    reqs_commands.append("python -m ensurepip --upgrade || mamba install pip -y")

    # Install additional packages if specified
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)

    if treesitter_env_name:
        # TODO: Just hardcode treesitter python version.
        reqs_commands.append(
            f"{env_command} create -n {treesitter_env_name} python=3.6 -y"
        )

    # Some additional cleaning in case.
    # reqs_commands.append("pip cache purge")
    return reqs_commands


def make_test_command(instance, without_directives=False, prefer_distributed=False):
    if instance["repo"] == "python/mypy":
        pattern = r"\[case ([^\]]+)\]"
        test_keys = re.findall(pattern, instance["test_patch"])
        test_keys_or = " or ".join(test_keys)
        test_command = (
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
            + " "
            + f'"{test_keys_or}"'
        )
        return test_command
    else:
        test_directives = [] if without_directives else get_test_directives(instance)
        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]

        # NOTE: Prefer distributed test command in the eval only setting, since tests can take a while.
        test_cmd = (
            specs.get("distributed_test_cmd", specs["test_cmd"])
            if prefer_distributed
            else specs["test_cmd"]
        )
        test_command = " ".join(
            [
                test_cmd,
                *test_directives,
            ]
        )
        return test_command


def _get_basic_eval_components(
    instance, specs, env_name, repo_directory, base_commit
) -> list[str]:
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]

    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])

    return eval_commands


def make_eval_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Assume that patch is already applied.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )

    test_command = make_test_command(
        instance, without_directives=True, prefer_distributed=True
    )
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,  # This should be empty, but let's just keep this.
    ]

    if "covering_tests" in instance:
        # For actual dataset, we know covering tests, so pull this from the instance.
        eval_commands += [test_command + " " + " ".join(instance["covering_tests"])]
    else:
        # This should probably never run on actual eval.
        eval_commands += [make_test_command(instance)]

    eval_commands += [
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


### BEGIN SWE-FFICIENCY EDITS

AST_START_TAG = "SWEPERF_AST_START"
AST_END_TAG = "SWEPERF_AST_END"


def make_ast_copy_preedit_commands(
    source_files: list[str], preedit_base_dir="/tmp/preedit/"
) -> list[str]:
    """Copy preedit state to a temp directory."""
    commands = [f"mkdir -p {preedit_base_dir}"]

    for file in source_files:
        # Assume file is in relative path form already.
        target_dir = os.path.join(preedit_base_dir, os.path.dirname(file))
        target_path = os.path.join(preedit_base_dir, file)

        commands.append(f"mkdir -p {target_dir}")
        commands.append(f"cp {file} {target_path}")

    return commands


ACCEPTABLE_EXTENIONS = [
    ".py",
    ".pyx",
    ".pyx.tp",
    ".pyi",
    ".pxi",
    ".pxd",
    ".pxd.tp",
]


def make_ast_run_tree_comparison_commands(
    source_files: list[str],
    preedit_base_dir="/tmp/preedit/",
    treesitter_file="/tresitter_compare.py",
) -> str:
    """For each source file, check if it exists in the `preedit_base_dir` using bash commands:
    if it exists, run the `treesitter_file`.
    """
    commands = []

    for file in source_files:
        # Construct the relative path in the preedit directory
        preedit_file = os.path.join(preedit_base_dir, file)
        postedit_file = file

        # Ignore anything that doesn't
        if all(ext not in file for ext in ACCEPTABLE_EXTENIONS):
            continue

        elif not file.endswith(".py"):
            commands.append(f"echo 'Warning: {file} is a non Python change.'")
            continue

        # Create the bash command to check if the file exists
        check_exists_command = f"if [ -f {postedit_file} ]; then python {treesitter_file} --preedit_file {preedit_file} --postedit_file {postedit_file}; else echo 'Warning: {file} not found in preedit directory'; fi"

        # Add the check-existence and run-comparison command to the list
        commands.append(check_exists_command)

    return commands


def make_meaningful_edit_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    This script list checks that the AST for pre and post edit files (ignoring comments) are different.
    We assume (safely) that performance related edits must be different.

    We also assume that patch (on source files is already)
    """

    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    source_files = re.findall(DIFF_MODIFIED_FILE_REGEX, instance["patch"])
    all_files = test_files + source_files
    # Reset entire repo state
    reset_command = f"git checkout {base_commit} {' '.join(all_files)}"

    filtered_source_files = [
        file
        for file in source_files
        if not perf_utils.is_doc_file(file)
        and not perf_utils.has_lock_file_change(file)
    ]

    # 1. Make a copy of all the source files to a temp directory.
    preedit_base_dir = "/tmp/preedit/"
    treesitter_file = "/tmp/treesitter_compare.py"

    copy_preedit_commands = make_ast_copy_preedit_commands(
        filtered_source_files, preedit_base_dir=preedit_base_dir
    )

    # 2. Apply patch diff (assuming it is copied at "/tmp/patch.diff"). Old versions don't have --allow-empty.
    apply_patch_command = "git apply -v /tmp/patch.diff"
    run_tree_comparison_commands = make_ast_run_tree_comparison_commands(
        filtered_source_files,
        preedit_base_dir=preedit_base_dir,
        treesitter_file=treesitter_file,
    )

    ast_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        reset_command,
    ]

    ast_commands.extend(copy_preedit_commands)
    ast_commands.append(apply_patch_command)
    ast_commands.append(f"echo {AST_START_TAG}")  # For easier parsing later.
    ast_commands.extend(run_tree_comparison_commands)

    ast_commands += [f"echo {AST_END_TAG}", reset_command]  # For easier parsing later.
    return ast_commands


def check_ast_result(raw_output):
    """Checks that AST parser says edit is reasonably meaningful."""
    relevant_lines = raw_output.splitlines()

    # Remove lines that start with "+"
    std_out_lines = [line for line in relevant_lines if not line.startswith("+")]

    # Get all the elements between "AST_START_TAG" and "AST_END_TAG".
    start_tag_index = std_out_lines.index(AST_START_TAG)
    end_tag_index = std_out_lines.index(AST_END_TAG)
    relevant_lines = std_out_lines[start_tag_index + 1 : end_tag_index]

    content = "\n".join(relevant_lines)

    warning_count = content.count("Warning")
    non_meaningful_count = content.count("NOOP")
    meaningful_count = content.count("MEANINGFUL")

    # Meaningful edits are Python edits that change the AST.
    has_meaningful = meaningful_count > 0

    # Warning just means that there are edits that we prob don't know.
    has_warning = warning_count > 0

    return has_meaningful or has_warning


COVERAGE_START_TAG = "SWEPERF_COVERAGE_START"
COVERAGE_END_TAG = "SWEPERF_COVERAGE_END"
PER_TEST_TAG = "SWEPERF_COVERAGE_REPORT"

COARSE_COVERAGE_AST_SCRIPT_LOCATION = "/tmp/coverage_ast.py"

CYTHON_COVERAGE_RC_FILE = """\
[run]
plugins = Cython.Coverage
"""
COVERAGE_RC_FILE_MAPPING = {
    "scikit-learn/scikit-learn": CYTHON_COVERAGE_RC_FILE,
    "pandas-dev/pandas": CYTHON_COVERAGE_RC_FILE,
    "numpy/numpy": CYTHON_COVERAGE_RC_FILE,
    "scipy/scipy": CYTHON_COVERAGE_RC_FILE,
    "explosion/spacy": CYTHON_COVERAGE_RC_FILE,
}

DEFAULT_COVERAGE_DATA_DIR = "/tmp/coverage_data/"
RAW_COVERAGE_OUTPUT_DIR = "/tmp/raw_coverage_data/"
DEFAULT_OUTFILE = "/tmp/all_tests.txt"

# Sklearn and dask are quite slow so use coarse AST methods.
REPOS_TO_IGNORE_COARSE_AST_COVERAGE = {
    "pandas-dev/pandas",  # Dependencies hard to track, lets just run full repo level coverage
    # "django/django", # Django"
    "numpy/numpy",  # Numpy tests run fast anyway.
    "pydata/xarray",  # Runs fast anyway.
    "matplotlib/matplotlib",  # Runs fast as well anyway.
    "explosion/spacy",
    "modin-project/modin",
    "pandas-dev/pandas",
}

# This identifies any tests in parent directories: many data science libraries obey this type of ruling.
identify_parent_dir_tests = 'while read file; do dir=$(dirname "$file"); prev=""; while [ "$dir" != "$prev" ]; do testdir="$dir/tests"; [ -d "$testdir" ] && find "$testdir" -type f; prev="$dir"; dir=$(dirname "$dir"); done; done | sort -u'

REPOS_TO_SPECIAL_TEST_COVERAGE = {
    "scipy/scipy": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "sympy/sympy": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "astropy/astropy": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "dask/dask": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "scikit-learn/scikit-learn": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
}

REPOS_TO_TEST_SEARCH_DEPTH = {
    # "pandas-dev/pandas": 1,
    # "modin-project/modin": 1,
    "scikit-learn/scikit-learn": 3,  # Sklearn tests are expensive
}

check_exit_code_pytest = 'status=$?; if [[ $status -ne 0 && $status -ne 1 ]]; then echo "Workload failed with exit code $status" >&2; exit $status; fi'


def make_coverage_commands_old(
    instance,
    repo_directory,
    source_files: list[str],
    parallel_coverage_workers: int = 4,
) -> list[str]:
    """
    This command enumerates all test files, and runs pytest coverage over all tests, printing the coverage report afterwards.
    """

    # TODO: Coverage run does not play well with pytest distributed, so we run it in a single process.
    test_cmd = make_test_command(
        instance, without_directives=True, prefer_distributed=False
    )
    testbed_source_files = [f"/testbed/{f}" for f in source_files]

    env_portion = ""

    if "pytest" in test_cmd or "tox" in test_cmd:
        module_cmd = f" --module {test_cmd}"
    else:
        if instance["repo"] == "sympy/sympy":
            env_portion = test_cmd.split("bin/test")[0].strip()
            module_cmd = test_cmd[len(env_portion) :]
        else:
            # Assume that we're just running a Python entrypoint.
            module_cmd = f" {test_cmd}"

    # First step is coarse pruning, using AST to identify collisions between test files and files edited in diff.
    # find_and_get_tests_as_single_line = f"find {repo_directory} -type f -name \"test_*.py\" -o -name \"*_test.py\" | xargs readlink -f | tr '\\n' ',' | sed '$s/,$//'"
    if instance["repo"] in REPOS_TO_IGNORE_COARSE_AST_COVERAGE:
        find_test_cmd = f'find {repo_directory} -type f -name "*test*.py"'
        run_ast_coverage_cmd_and_get_tests = find_test_cmd + f" > {DEFAULT_OUTFILE}"

    elif instance["repo"] in REPOS_TO_SPECIAL_TEST_COVERAGE:
        # assert False
        try:
            template = REPOS_TO_SPECIAL_TEST_COVERAGE[instance["repo"]]
            source_files_commas = ",".join(source_files)

            run_ast_coverage_cmd_and_get_tests = (
                template.format(source_files=source_files_commas)
                + f" > {DEFAULT_OUTFILE}"
            )
        except Exception as e:
            print(e)
            traceback.print_exc()

    else:
        test_recursion_depth = REPOS_TO_TEST_SEARCH_DEPTH.get(instance["repo"], 3)
        run_ast_coverage_cmd_and_get_tests = f"python {COARSE_COVERAGE_AST_SCRIPT_LOCATION} --source_files {','.join(testbed_source_files)} --outfile {DEFAULT_OUTFILE} --ignore_depth {test_recursion_depth} 2> /dev/null || exit 1"

    coverage_file_content = COVERAGE_RC_FILE_MAPPING.get(instance["repo"])
    coverage_write_script = ""
    coverage_rc_file_addition = ""
    if coverage_file_content:
        coverage_file = "/tmp/.coveragerc"
        coverage_write_script = f"""/bin/bash -c \'mkdir -p "$(dirname {coverage_file})" && cat <<EOF > "{coverage_file}"
{coverage_file_content}
EOF\'"""
        coverage_rc_file_addition = f"COVERAGE_RCFILE={coverage_file} "

    # remove_existing_coveragerc = "rm /testbed/.coveragerc"

    test_directive_symbol = "$line"
    if instance["repo"] == "django/django":
        # Django requires as modeule.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )
    elif instance["repo"] == "scipy/scipy" and instance["version"] in [
        "0.16",
        "0.17",
        "0.18",
        "0.19",
    ]:
        # Scipy early versions require you to provide as module file.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )

    # Next, up for matched tests, we then feed this into actual pytest with coverage to verify that the diff intersects with the actual edit.

    # coverage_command = (
    #     f"cat {DEFAULT_OUTFILE} |"
    #     f"xargs -n 1 -P {parallel_coverage_workers} -d\\\\n --verbose --no-run-if-empty -I % "
    #     f"/bin/bash -c 'tmpfile=$(mktemp); echo \"Processing: %\" >&2; {coverage_rc_file_addition} "
    #     f"COVERAGE_FILE={DEFAULT_COVERAGE_DATA_DIR}$(echo -n \"%\" | sha256sum | cut -d\" \" -f1) {env_portion} "
    #     f"coverage run -p --include={','.join(testbed_source_files)} --context=% {module_cmd} {test_directive_symbol}"
    # )

    # f"coverage run -p --include={','.join(testbed_source_files)} --context=\"$line\" {module_cmd} {test_directive_symbol} | tee \"$outfile\" > /dev/null'"
    coverage_command = (
        f"cat {DEFAULT_OUTFILE} | "
        f"shuf | "  # Shuffle to avoid bias.
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P {parallel_coverage_workers} -d\\\\n --verbose --no-run-if-empty -I % "
        f'/bin/bash -c \'idx=$(echo % | cut -d" " -f1); '
        f'line=$(echo % | cut -d" " -f2-); '
        f'outfile="{RAW_COVERAGE_OUTPUT_DIR}$idx.txt"; '
        f"mkdir -p {RAW_COVERAGE_OUTPUT_DIR}; "
        f"currenttime=$(date); "
        f'echo "[$currenttime] Processing [$idx]: $line" >&2; '
        f"{coverage_rc_file_addition} "
        f'COVERAGE_FILE={DEFAULT_COVERAGE_DATA_DIR}$(echo -n "$line" | sha256sum | cut -d" " -f1) {env_portion} '
        f"coverage run -p --include={','.join(testbed_source_files)} --context=\"$line\" {module_cmd} {test_directive_symbol} | tee \"$outfile\" > /dev/null; "
        f'echo "[$currenttime] Finished processing [$idx]: $line" >&2;\''
    )

    # Print all outputs, then delete them.
    # print_output_command = f"cat /tmp/coverage_raw_output/*.txt"

    echo_nproc = 'echo "Number of parallel workers: $(nproc)"'

    return [
        # remove_existing_coveragerc,
        echo_nproc,
        coverage_write_script,
        run_ast_coverage_cmd_and_get_tests,
        f"cat {DEFAULT_OUTFILE}",
        coverage_command,
        # print_output_command, # This makes container take literally forever to run, so we don't do this.
    ]


def make_coverage_commands(
    instance,
    repo_directory,
    source_files: list[str],
    parallel_coverage_workers: int = 4,
) -> list[str]:
    """
    Build shell snippets that:
      • compute the candidate-test list (AST coarse coverage or repo-specific)
      • create a helper script to run one test with coverage
      • launch an xargs pipeline that feeds tests to that helper
    """

    # ------------------------------------------------------------------ basics
    test_cmd = make_test_command(
        instance, without_directives=True, prefer_distributed=False
    )
    testbed_source_files = [f"/testbed/{f}" for f in source_files]

    # --- translate test_cmd into the form "coverage run … --module pytest …"
    env_portion = ""
    if "pytest" in test_cmd or "tox" in test_cmd:
        module_cmd = f" --module {test_cmd}"
    else:
        if instance["repo"] == "sympy/sympy":
            env_portion = test_cmd.split("bin/test")[0].strip()
            module_cmd = test_cmd[len(env_portion) :]
        else:
            module_cmd = f" {test_cmd}"

    # ------------------------------------------------------------------ coarse AST pruning
    if instance["repo"] in REPOS_TO_IGNORE_COARSE_AST_COVERAGE:
        find_tests = f'find {repo_directory} -type f -name "*test*.py"'
        run_ast_coverage_cmd_and_get_tests = f"{find_tests} > {DEFAULT_OUTFILE}"
    elif instance["repo"] in REPOS_TO_SPECIAL_TEST_COVERAGE:
        tmpl = REPOS_TO_SPECIAL_TEST_COVERAGE[instance["repo"]]
        run_ast_coverage_cmd_and_get_tests = (
            tmpl.format(source_files=",".join(source_files)) + f" > {DEFAULT_OUTFILE}"
        )
    else:
        depth = REPOS_TO_TEST_SEARCH_DEPTH.get(instance["repo"], 3)
        run_ast_coverage_cmd_and_get_tests = (
            f"python {COARSE_COVERAGE_AST_SCRIPT_LOCATION} "
            f"--source_files {','.join(testbed_source_files)} "
            f"--outfile {DEFAULT_OUTFILE} --ignore_depth {depth} 2>/dev/null || exit 1"
        )

    # ------------------------------------------------------------------ repo-specific test-directive symbol
    if instance["repo"] == "django/django":
        tdir = "$(echo \"$line\" | sed 's|.py||; s|/testbed/tests/||; s|/|.|g')"
    elif instance["repo"] == "scipy/scipy" and instance["version"] in [
        "0.16",
        "0.17",
        "0.18",
        "0.19",
    ]:
        tdir = "$(echo \"$line\" | sed 's|.py||; s|/testbed/tests/||; s|/|.|g')"
    else:
        tdir = "$line"

    # ------------------------------------------------------------------ optional .coveragerc support
    coverage_file_content = COVERAGE_RC_FILE_MAPPING.get(instance["repo"])
    coverage_write_script = ""
    coverage_rc_file_addition = ""
    if coverage_file_content:
        coverage_file = "/tmp/.coveragerc"
        coverage_write_script = f"""\
/bin/bash -c 'mkdir -p "$(dirname {coverage_file})" && cat <<"EOF" > {coverage_file}
{coverage_file_content}
EOF'"""
        coverage_rc_file_addition = f"COVERAGE_RCFILE={coverage_file} "

    # ------------------------------------------------------------------ helper script (written by the driver itself)
    helper_name = "./run_one_coverage_test.sh"
    helper_body = textwrap.dedent(
        f"""\
#!/usr/bin/env bash
set -euo pipefail

input=$1
read -r idx line <<<"$input"

outfile="{RAW_COVERAGE_OUTPUT_DIR}/${{idx}}.txt"
mkdir -p "{RAW_COVERAGE_OUTPUT_DIR}"

echo "[ $(date) ] Processing [${{idx}}]: $line" >&2

{coverage_rc_file_addition} \\
COVERAGE_FILE={DEFAULT_COVERAGE_DATA_DIR}$(printf "%s" "$line" | sha256sum | cut -d" " -f1) \\
{env_portion} coverage run -p \\
    --include={','.join(testbed_source_files)} \\
    --context="$line" {module_cmd} {tdir} \\
| tee "$outfile" > /dev/null"""
    ).rstrip()

    create_helper_cmd = f"""\
/bin/bash -c 'cat <<"EOF" > {helper_name}
{helper_body}
EOF'"""
    make_helper_executable = f"chmod +x {helper_name}"

    # ------------------------------------------------------------------ xargs pipeline
    coverage_command = (
        f"cat {DEFAULT_OUTFILE} | "
        f"shuf | "  # Shuffle to avoid bias.
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P {parallel_coverage_workers} -d\\\\n --verbose --no-run-if-empty -I % "
        f"{helper_name} %"
    )

    # ------------------------------------------------------------------ extras / bookkeeping
    echo_nproc = 'echo "Number of parallel workers: $(nproc)"'

    # ------------------------------------------------------------------ final command list
    return [
        echo_nproc,
        coverage_write_script,  # may be empty string
        run_ast_coverage_cmd_and_get_tests,
        f"cat {DEFAULT_OUTFILE}",  # optional diagnostic
        create_helper_cmd,
        make_helper_executable,
        f"cat {helper_name}",  # optional diagnostic
        coverage_command,
        f"rm {helper_name}",  # cleanup helper script
    ]


COVERAGE_ANALYSIS_SCRIPT_LOCATION = "/tmp/coverage_analysis.py"


def make_coverage_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Applies the test patch and runs each test one by one to determine coverage. Assumes
    coverage is alreayd installed in the environment.
    """

    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        (f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}")
        if test_patch
        else ""
    )

    apply_patch_command = "git apply -v /tmp/patch.diff"
    reset_command = f"git checkout {base_commit}"

    source_files = re.findall(DIFF_MODIFIED_FILE_REGEX, instance["patch"])

    num_parallel_workers = 8

    REPOS_TO_SLOW_COVERAGE = {
        "scikit-learn/scikit-learn",
        # "sympy/sympy",
    }

    if instance["repo"] in REPOS_TO_SLOW_COVERAGE:
        # Tests are expensive, so we run sequentially.
        num_parallel_workers = 1

    coverage_commands = make_coverage_commands(
        instance,
        repo_directory,
        source_files,
        parallel_coverage_workers=num_parallel_workers,
    )

    # Get base components first.
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    # coverage does not have a fun time when you don't run it from package root dir.
    cd_into_module_dir_cmd = "cd $(python -c 'import setuptools; packages = setuptools.find_packages(); print(next((p for p in packages if \".\" not in p), None))')"
    cd_back_to_repo = f"cd {repo_directory}"

    # Parse coverage analysis command: assumes that patch is at /tmp/patch.diff.
    parse_coverage_file_cmd = (
        f"python {COVERAGE_ANALYSIS_SCRIPT_LOCATION} --patch-file /tmp/patch.diff"
    )

    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        apply_patch_command,
    ]

    if "install" in specs:
        eval_commands.append(specs["install"])

    eval_commands += [
        # cd_into_module_dir_cmd, # At this point, we are in the module source directory.
        f"coverage --version",
        f"mkdir -p {DEFAULT_COVERAGE_DATA_DIR}",
        *coverage_commands,
        f"echo {COVERAGE_START_TAG}",
        parse_coverage_file_cmd,
        f"echo {COVERAGE_END_TAG}",
        # cd_back_to_repo,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
        reset_command,
    ]
    return eval_commands


def parse_coverage_report(raw_output):
    # Coverage output is sandwitched between COVERAGE_START_TAG and COVERAGE_END_TAGS.
    raw_stdout_lines = [l for l in raw_output.splitlines() if not l.startswith("+")]
    line_start_index = raw_stdout_lines.index("BEGIN_VALID_TESTS")
    line_end_index = raw_stdout_lines.index("END_VALID_TESTS")

    return raw_stdout_lines[line_start_index + 1 : line_end_index]


PERF_WORKLOAD_SCRIPT_LOCATION = "/tmp/workload.py"
PERF_CPROFILE_OUTPUT_LOCATION = "/tmp/workload_cprofile.prof"

perf_start_tag = "PERF_START:"
perf_end_tag = "PERF_END:"


def _get_california_housing_download_commands():
    """Download and cache California housing dataset for scikit-learn workloads."""
    return [
        "    echo 'Preparing California housing dataset...'",
        "    DATA_HOME=\"${HOME}/scikit_learn_data\"",
        "    mkdir -p \"${DATA_HOME}\"",
        "    if [ ! -s \"${DATA_HOME}/cal_housing.tgz\" ] || ! gzip -t \"${DATA_HOME}/cal_housing.tgz\" 2>/dev/null; then",
        "        rm -f /tmp/california_housing.tgz \"${DATA_HOME}/cal_housing.tgz\"",
        "        echo 'Downloading from primary URL...'",
        "        curl -L \\",
        "             --retry 10 --retry-delay 5 --max-time 120 \\",
        "             -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' \\",
        "             -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' \\",
        "             -o /tmp/california_housing.tgz \\",
        "             https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz",
        "        if [ -s /tmp/california_housing.tgz ] && gzip -t /tmp/california_housing.tgz 2>/dev/null; then",
        "            mv /tmp/california_housing.tgz \"${DATA_HOME}/cal_housing.tgz\"",
        "            echo 'Valid archive cached.'",
        "        else",
        "            echo 'FATAL: Could not download valid archive. Aborting.'",
        "            exit 1",
        "        fi",
        "    else",
        "        echo 'Archive already cached and valid.'",
        "    fi",
    ]


def make_performance_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Assumes we have already copied results to the directories specified above.
    """
    # Run with a few iterations of warmup first.
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    # Add scikit-learn patching logic for specific instances
    if instance[KEY_INSTANCE_ID] == "scikit-learn__scikit-learn-15834":
        eval_commands.extend([
            "if grep -q 'fetch_20newsgroups' /tmp/workload.py; then",
            "    echo 'Downloading 20news-bydate archive...'",
            "    wget -U \"Mozilla/5.0\" -c -t 10 --timeout=120 -O /tmp/20news.tar.gz http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz",
            "    echo 'Extracting...'",
            "    mkdir -p /tmp/20news_extracted /tmp/20news_all",
            "    tar -xzf /tmp/20news.tar.gz -C /tmp/20news_extracted",
            "    if [ -d /tmp/20news_extracted/20news-bydate-train ]; then",
            "        cp -r /tmp/20news_extracted/20news-bydate-train/* /tmp/20news_all/",
            "        cp -r /tmp/20news_extracted/20news-bydate-test/* /tmp/20news_all/",
            "    else",
            "        cp -r /tmp/20news_extracted/* /tmp/20news_all/ 2>/dev/null || true",
            "    fi",
            "    echo 'Patching /tmp/workload.py...'",
            "    sed -i 's/from sklearn.datasets import fetch_20newsgroups/from sklearn.datasets import load_files/g' /tmp/workload.py",
            "    sed -i \"s/dataset = fetch_20newsgroups(subset='all')/dataset = load_files('\\/tmp\\/20news_all', encoding='latin1')/g\" /tmp/workload.py",
            "    echo 'Cleanup...'",
            "    rm -rf /tmp/20news_extracted /tmp/20news.tar.gz",
            "    echo '20 Newsgroups ready – workload will now load local files.'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "scikit-learn__scikit-learn-29060":
        eval_commands.extend([
            "if grep -q 'fetch_california_housing' /tmp/workload.py; then",
            *_get_california_housing_download_commands(),

            "    echo 'Patching /tmp/workload.py...'",
            "    python3 -c \"",
            "import pathlib, re",
            "new_block = '''import tarfile, os",
            "import pandas as pd",
            "from sklearn.datasets import get_data_home",
            "from sklearn.impute import KNNImputer",
            "",
            "data_home = get_data_home()",
            "archive_path = os.path.join(data_home, 'cal_housing.tgz')",
            "with tarfile.open(mode='r:gz', name=archive_path) as f:",
            "    csv_file = None",
            "    for m in f.getmembers():",
            "        if m.isfile() and m.name.endswith('.csv'):",
            "            csv_file = m.name",
            "            break",
            "    if csv_file is None:",
            "        raise FileNotFoundError('No CSV found in archive')",
            "    df = pd.read_csv(f.extractfile(csv_file))",
            "",
            "df.columns = df.columns.str.lower().str.strip()",
            "df['averooms'] = df['total_rooms'] / df['households']",
            "df['avebedrms'] = df['total_bedrooms'] / df['households']",
            "df['aveoccup'] = df['population'] / df['households']",
            "rename_map = {",
            "    'median_income': 'medinc',",
            "    'housing_median_age': 'houseage',",
            "    'population': 'population',",
            "    'latitude': 'latitude',",
            "    'longitude': 'longitude',",
            "    'averooms': 'averooms',",
            "    'avebedrms': 'avebedrms',",
            "    'aveoccup': 'aveoccup'",
            "}",
            "df = df.rename(columns=rename_map)",
            "feature_names = ['medinc', 'houseage', 'averooms', 'avebedrms', 'population', 'aveoccup', 'latitude', 'longitude']",
            "X = df[feature_names]",
            "y = pd.Series(df['median_house_value'], name='house_value')",
            "'''",
            "code = pathlib.Path('/tmp/workload.py').read_text()",
            "pattern = r'from sklearn.datasets import fetch_california_housing.*?y = pd\\.Series\\(calhousing\\.target, name=.house_value.\\)'",
            "code, n = re.subn(pattern, new_block.strip(), code, flags=re.DOTALL)",
            "if n == 0:",
            "    raise RuntimeError('Pattern not found in workload.py')",
            "pathlib.Path('/tmp/workload_new.py').write_text(code)",
            "\"",
            "    mv /tmp/workload_new.py /tmp/workload.py",
            "    echo 'workload.py successfully patched (with all imports).'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "scikit-learn__scikit-learn-15615":
        eval_commands.extend([
            "if grep -q 'fetch_california_housing' /tmp/workload.py; then",
            *_get_california_housing_download_commands(),
            "    echo 'Patching /tmp/workload.py for local archive loading (nan_euclidean_distances version)...'",
            "    python3 -c \"",
            "import pathlib, re",
            "new_block = '''import tarfile, os",
            "from sklearn.datasets import get_data_home",
            "from sklearn.metrics.pairwise import nan_euclidean_distances",
            "",
            "data_home = get_data_home()",
            "archive_path = os.path.join(data_home, 'cal_housing.tgz')",
            "with tarfile.open(mode='r:gz', name=archive_path) as f:",
            "    csv_file = None",
            "    for m in f.getmembers():",
            "        if m.isfile() and m.name.endswith('.csv'):",
            "            csv_file = m.name",
            "            break",
            "    if csv_file is None:",
            "        raise FileNotFoundError('No CSV found in archive')",
            "    df = pd.read_csv(f.extractfile(csv_file))",
            "",
            "df.columns = df.columns.str.lower().str.strip()",
            "df['averooms'] = df['total_rooms'] / df['households']",
            "df['avebedrms'] = df['total_bedrooms'] / df['households']",
            "df['aveoccup'] = df['population'] / df['households']",
            "rename_map = {",
            "    'median_income': 'medinc',",
            "    'housing_median_age': 'houseage',",
            "    'population': 'population',",
            "    'latitude': 'latitude',",
            "    'longitude': 'longitude',",
            "    'averooms': 'averooms',",
            "    'avebedrms': 'avebedrms',",
            "    'aveoccup': 'aveoccup'",
            "}",
            "df = df.rename(columns=rename_map)",
            "feature_names = ['medinc', 'houseage', 'averooms', 'avebedrms', 'population', 'aveoccup', 'latitude', 'longitude']",
            "X = df[feature_names][:20000]",
            "'''",
            "code = pathlib.Path('/tmp/workload.py').read_text()",
            "pattern = r'from sklearn.datasets import fetch_california_housing.*?X = pd\\.DataFrame\\(calhousing\\.data, columns=calhousing\\.feature_names\\)\\[:20000\\]'",
            "code, n = re.subn(pattern, new_block.strip(), code, flags=re.DOTALL)",
            "if n == 0:",
            "    raise RuntimeError('Pattern not found in workload.py')",
            "pathlib.Path('/tmp/workload_new.py').write_text(code)",
            "\"",
            "    mv /tmp/workload_new.py /tmp/workload.py",
            "    echo 'workload.py successfully patched for nan_euclidean_distances workflow.'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "pandas-dev__pandas-52469":
        eval_commands.extend([
            "if grep -q 'dtype_backend' /tmp/workload.py; then",
            "    python3 -c \"",
            "import pathlib",
            "code = pathlib.Path('/tmp/workload.py').read_text()",
            "code = code.replace(",
            "    '''df_new = pd.read_csv(temp_path, engine='pyarrow', dtype_backend='pyarrow')''',",
            "    '''df_new = pd.read_csv(temp_path)''')",
            "pathlib.Path('/tmp/workload_patched.py').write_text(code)",
            "\"",
            "    mv /tmp/workload_patched.py /tmp/workload.py",
            "    echo 'workload.py patched: removed dtype_backend for older pandas'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "pandas-dev__pandas-52928":
        eval_commands.extend([
            "if grep -q 'dtype_backend' /tmp/workload.py; then",
            r"""    sed -i 's/df_new = pd\.read_csv(temp_path, engine="pyarrow", dtype_backend="pyarrow")/df_new = pd.read_csv(temp_path)/' /tmp/workload.py""",
            r"""    sed -i 's/arr = df_new\["v1"\]\.array\._pa_array/arr = df_new["v1"].to_numpy()/' /tmp/workload.py""",
            r"""    sed -i 's/pd\.Float64Dtype()\.__from_arrow__(arr)/pd.array(arr, dtype=pd.Float64Dtype())/' /tmp/workload.py""",
            "    echo 'workload.py patched: replaced dtype_backend and _pa_array for older pandas'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "pandas-dev__pandas-52548":
        eval_commands.extend([
            # ArrowDtype, dtype_backend='pyarrow', and engine='pyarrow' are
            # not supported in older pandas (pre-2.0).  The workload creates
            # ArrowDtype(pa.timestamp(unit='ns')) and calls read_csv with
            # engine='pyarrow' + dtype_backend='pyarrow' which triggers an
            # OverflowError in old pyarrow date-conversion code.
            "if grep -q \"dtype_backend='pyarrow'\" /tmp/workload.py; then",
            "    # Replace ArrowDtype constructor with plain numpy datetime64",
            "    sed -i \"s/pd\\.ArrowDtype(pa\\.timestamp(unit='ns'))/'datetime64[ns]'/\" /tmp/workload.py",
            "    # Drop pyarrow-specific kwargs (keep parse_dates intact)",
            "    sed -i \"s/, dtype_backend='pyarrow'//\" /tmp/workload.py",
            "    sed -i \"s/, engine='pyarrow'//\" /tmp/workload.py",
            "    echo 'workload.py patched: removed ArrowDtype and dtype_backend for pandas-52548'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] in ("pandas-dev__pandas-51549", "pandas-dev__pandas-52525"):
        eval_commands.extend([
            # pyarrow-backed dtypes cause IndexError/TypeError in older pandas
            "if grep -q '\\[pyarrow\\]' /tmp/workload.py; then",
            "    sed -i 's/\\[pyarrow\\]//g' /tmp/workload.py",
            "    # 'null' isn't a valid numpy dtype; use 'object' instead",
            "    sed -i \"s/dtype='null'/dtype='object'/\" /tmp/workload.py",
            "    echo 'workload.py patched: removed [pyarrow] dtype suffix for older pandas'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "pandas-dev__pandas-51439":
        # ArrowExtensionArray in older pandas lacks __array__(), so
        # np.asarray(self, dtype=dtype) in to_numpy() falls back to
        # element-by-element iteration — ~13 s/call for 10M elements.
        # The workload calls to_numpy() 5 000 times, which would take
        # ~18 h.  The underlying pyarrow ChunkedArray DOES have
        # __array__() (sub-millisecond).  Prepend a monkey-patch to
        # the workload file so __array__ delegates to _data.
        eval_commands.extend([
            # Write to stdout + redirect + mv to work around overlayfs
            # permission issues (chmod is a no-op on overlayfs).
            "python3 -c \"import pathlib, sys; c = pathlib.Path('/tmp/workload.py').read_text(); sys.stdout.write('import numpy as _np\\nfrom pandas.core.arrays.arrow.array import ArrowExtensionArray as _AEA\\n_AEA.__array__ = lambda self, dtype=None: _np.asarray(self._data, dtype=dtype)\\n' + c)\" > /tmp/workload_patched.py && mv /tmp/workload_patched.py /tmp/workload.py",
            "echo 'workload.py: prepended ArrowExtensionArray.__array__ monkey-patch for 51439'",
        ])
    elif instance[KEY_INSTANCE_ID].startswith("pandas-dev__pandas"):
        eval_commands.extend([
            "if grep -q 'makeStringIndex' /tmp/workload.py; then",
            "    # Remove try/except ImportError fallback blocks first (avoids SyntaxError)",
            "    sed -i '/^try:$/{N;N;N;/except ImportError:/d}' /tmp/workload.py",
            "    # Remove standalone import pandas._testing as tm",
            "    sed -i '/^import pandas\\._testing as tm$/d' /tmp/workload.py",
            "    # Replace tm.makeStringIndex(EXPR) with pd.Index equivalent",
            "    # cat+mv creates a new inode with default (writable) permissions,",
            "    # avoiding overlayfs issues that cause chmod to be a no-op.",
            "    cat /tmp/workload.py > /tmp/workload_tmp.py && \\",
            "    mv /tmp/workload_tmp.py /tmp/workload.py",
            "    python3 << 'PYEOF'",
            "import pathlib, re",
            "code = pathlib.Path('/tmp/workload.py').read_text()",
            "code = re.sub(r'tm\\.makeStringIndex\\(([^)]+)\\)', r'pd.Index([f\"str{i}\" for i in range(\\1)])', code)",
            "pathlib.Path('/tmp/workload.py').write_text(code)",
            "PYEOF",
            "    echo 'workload.py patched: replaced tm.makeStringIndex with pd.Index'",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "scikit-learn__scikit-learn-24856":
        eval_commands.extend([
            "if grep -q 'HIGGS.csv.gz' /tmp/workload.py; then",
            "    HIGGS_PATH=/tmp/HIGGS.csv.gz",
            "    if [ ! -s \"$HIGGS_PATH\" ] || ! gzip -t \"$HIGGS_PATH\" 2>/dev/null; then",
            "        echo 'FATAL: HIGGS.csv.gz not found or corrupted at $HIGGS_PATH.'",
            f"        echo 'Please place the file at "
            + os.environ.get(
                "SWEFFICIENCY_HIGGS_DATASET",
                # $HOME-relative, not the reference box's literal.
                os.path.expanduser("~/swefficiency/dataset/HIGGS.csv.gz"),
            )
            + " on the host.'",
            "        exit 1",
            "    else",
            "        echo 'HIGGS.csv.gz found and valid.'",
            "    fi",
            "fi",
        ])
    elif instance[KEY_INSTANCE_ID] == "pydata__xarray-4740":
        eval_commands.extend([
            "if grep -q 'da\\.array' /tmp/workload.py; then",
            r"""    sed -i 's/da\.array(\[0, 0\.5, 1\])/np.array([0.0, 0.5, 1.0])/' /tmp/workload.py""",
            r"""    sed -i 's/da\.array(\[0, 1\])/np.array([0, 1])/' /tmp/workload.py""",
            r"""    sed -i 's/da\.array(\[3, 4\])/np.array([3, 4])/' /tmp/workload.py""",
            "    echo 'workload.py patched: replaced dask arrays with numpy for older xarray'",
            "fi",
        ])

    # Compute the before and after.
    eval_commands.extend(
        [
            f"echo '{perf_start_tag}'",
            f"python {PERF_WORKLOAD_SCRIPT_LOCATION}",
            f"echo '{perf_end_tag}'",
        ]
    )

    return eval_commands


def make_performance_profiling_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Assumes we have already copied results to the directories specified above.
    """
    # Run with a few iterations of warmup first.
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    if instance[KEY_INSTANCE_ID] == "pandas-dev__pandas-51439":
        eval_commands.extend([
            "python3 -c \"import pathlib, sys; c = pathlib.Path('/tmp/workload.py').read_text(); sys.stdout.write('import numpy as _np\\nfrom pandas.core.arrays.arrow.array import ArrowExtensionArray as _AEA\\n_AEA.__array__ = lambda self, dtype=None: _np.asarray(self._data, dtype=dtype)\\n' + c)\" > /tmp/workload_patched.py && mv /tmp/workload_patched.py /tmp/workload.py",
            "echo 'workload.py: prepended ArrowExtensionArray.__array__ monkey-patch for 51439'",
        ])

    # Compute the before and after
    eval_commands.extend(
        [
            # Run the cProfile command to profile the workload script.
            f"python -m cProfile -o {PERF_CPROFILE_OUTPUT_LOCATION} {PERF_WORKLOAD_SCRIPT_LOCATION}",
        ]
    )

    return eval_commands


def parse_perf_output(raw_output):
    cleaned_perf_output = [l for l in raw_output.splitlines() if not l.startswith("+")]
    cleaned_per_output = "\n".join(cleaned_perf_output) + "\n"

    # Extract whats in between "START_AFTER_CHANGE:" and "END_AFTER_CHANGE:".
    start_index = cleaned_per_output.find(perf_start_tag)
    end_index = cleaned_per_output.find(perf_end_tag)

    if start_index == -1 or end_index == -1:
        raise ValueError("Perf tags not found in output.")

    # Get only text between the tags.
    perf_text = cleaned_per_output[start_index + len(perf_start_tag) : end_index]
    perf_text = cleaned_per_output

    pattern = r"(?:Mean|Std\s*Dev):\s*([\S]+)"
    matches = re.findall(pattern, perf_text)

    mean, std_dev = matches[0], matches[1]
    return float(mean), float(std_dev)


DEFAULT_COVERING_TESTS_LOCATION = "/tmp/covering_tests.txt"
DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION = "/tmp/single_thread_tests.txt"
DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION = "/tmp/raw_correctness_output"


def get_correctness_script_list(
    instance,
    specs,
    env_name,
    repo_directory,
    base_commit,
    test_patch,
    parallel_workers=4,
):
    """
    Assumes we have already copied results to the directories specified above.
    """
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    test_command = make_test_command(
        instance, without_directives=True, prefer_distributed=False
    )

    test_directive_symbol = "$line"
    if instance["repo"] == "django/django":
        # Django requires as modeule.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )
    elif instance["repo"] == "scipy/scipy" and instance["version"] in [
        "0.16",
        "0.17",
        "0.18",
        "0.19",
    ]:
        # Scipy early versions require you to provide as module file.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )

    # if instance['repo'] == "scikit-learn/scikit-learn":
    #     # Sklearn tests are quite expensive, so we run them single threaded.
    #     parallel_workers = 1

    NUM_RETRY = 2

    # Pin the maths thread pools so the run does not depend on the host's core
    # count (see get_correctness_thread_env).  The prefix must stay free of
    # single quotes: it is embedded in the single-quoted `bash -c '...'` below.
    thread_env = get_correctness_thread_env()

    correctness_command = (
        f"{{ cat {DEFAULT_COVERING_TESTS_LOCATION} | "
        f"grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} || true; }} | "
        f"shuf | "
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P {parallel_workers} -d\\\\n --verbose --no-run-if-empty -I % "
        f"/bin/bash -c '"
        f"set -o pipefail; "  # make PIPESTATUS meaningful
        f'idx=$(echo % | cut -d" " -f1); '
        f'line=$(echo % | cut -d" " -f2-); '
        f'outfile="{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$idx.txt"; '
        f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
        f'echo "[$(date)] Processing [$idx]: $line" >&2; '
        f"retries=0; rc=0; "
        f"while true; do "
        # capture both stdout+stderr; write to file; print nothing to terminal
        f'{thread_env}{test_command} {test_directive_symbol} | tee "$outfile" > /dev/null; '
        f"rc=${{PIPESTATUS[0]}}; "
        f'if [ "$rc" -le 1 ]; then break; fi; '
        f'if [ "$retries" -ge {NUM_RETRY} ]; then break; fi; '
        f"retries=$((retries+1)); "
        f'echo "[$(date)] Exit $rc (>1). Retrying [$idx] attempt $retries/{NUM_RETRY}: $line" >&2; '
        f"done; "
        f'echo "[$(date)] Finished processing [$idx] with exit $rc: $line" >&2; '
        f"exit $rc;"
        f"'"
    )

    # # We need to double quote test_command since
    # correctness_command = (
    #     f"cat {DEFAULT_COVERING_TESTS_LOCATION} | "
    #     f"grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | " # Remove single-threaded tests.
    #     f"shuf | " # Shuffle to avoid bias.
    #     "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
    #     f"awk '{{ print NR-1 \" \" $0 }}' | "
    #     f"xargs -P {parallel_workers} -d\\\\n --verbose --no-run-if-empty -I % "
    #     f"/bin/bash -c 'idx=$(echo % | cut -d\" \" -f1); "
    #     f"line=$(echo % | cut -d\" \" -f2-); "
    #     f"outfile=\"{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$idx.txt\"; "
    #     f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
    #     f"currenttime=$(date); "
    #     f"echo \"[$currenttime] Processing [$idx]: $line\" >&2; "
    #     f"{test_command} {test_directive_symbol} | tee \"$outfile\" > /dev/null; "
    #     f"echo \"[$currenttime] Finished processing [$idx]: $line\" >&2;'"
    # )

    single_threaded_correctness_cmd = (
        f"if [[ -s {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} ]]; then "
        f"cat {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | "
        f"shuf | "  # Shuffle to avoid bias.
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P 1 -d\\\\n --verbose --no-run-if-empty -I % "
        f"/bin/bash -c '"
        f'idx=$(echo % | cut -d" " -f1); '
        f"NUM_PARALLEL_TESTS=$(cat {DEFAULT_COVERING_TESTS_LOCATION} | grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | wc -l); "
        f'line=$(echo % | cut -d" " -f2-); '
        f'outfile="{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$((idx + NUM_PARALLEL_TESTS)).txt"; '
        f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
        f'echo "[$(date)] Processing [$((idx + NUM_PARALLEL_TESTS))]: $line" >&2; '
        f"retries=0; rc=0; "
        f"while true; do "
        f'PYTHONHASHSEED=0 {thread_env}{test_command} {test_directive_symbol} | tee "$outfile" > /dev/null; '
        f"rc=${{PIPESTATUS[0]}}; "
        f'if [ "$rc" -le 1 ]; then break; fi; '
        f'if [ "$retries" -ge {NUM_RETRY} ]; then break; fi; '
        f"retries=$((retries+1)); "
        f'echo "[$(date)] Exit $rc (>1). Retrying [$((idx + NUM_PARALLEL_TESTS))] attempt $retries/{NUM_RETRY}: $line" >&2; '
        f"done; "
        f'echo "[$(date)] Finished processing [$((idx + NUM_PARALLEL_TESTS))] with exit $rc: $line" >&2; '
        f"exit $rc;' "
        f"else echo 'Input file is empty, skipping.' >&2; fi"
    )

    # single_threaded_correctness_cmd = (
    #     f"if [[ -s {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} ]]; then "
    #     f"cat {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | "
    #     f"shuf | " # Shuffle to avoid bias.
    #     "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
    #     f"awk '{{ print NR-1 \" \" $0 }}' | "
    #     f"xargs -P 1 -d\\\\n --verbose --no-run-if-empty -I % "
    #     f"/bin/bash -c 'idx=$(echo % | cut -d\" \" -f1); "
    #     f"NUM_PARALLEL_TESTS=$(cat {DEFAULT_COVERING_TESTS_LOCATION} | grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | wc -l); "
    #     f"line=$(echo % | cut -d\" \" -f2-); "
    #     f"outfile=\"{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$((idx + NUM_PARALLEL_TESTS)).txt\"; "
    #     f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
    #     f"currenttime=$(date); "
    #     f"echo \"[$currenttime] Processing [$((idx + NUM_PARALLEL_TESTS))]: $line\" >&2; "
    #     f"{test_command} {test_directive_symbol} | tee \"$outfile\" > /dev/null'; "
    #     f"else echo 'Input file is empty, skipping.' >&2; fi"
    # )

    eval_commands.extend([correctness_command, single_threaded_correctness_cmd])

    return eval_commands


INTROSPECTION_GUARD_CMD_LOCATION = "/tmp/introspection_patch_check.py"


def get_introspection_guard_cmds(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Get commands to run introspection guard. Assumes that patch is already applied.
    """
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )
    check_introspection_cmd = (
        f"python {INTROSPECTION_GUARD_CMD_LOCATION} --patch-file /tmp/patch.diff"
    )
    eval_commands.append(check_introspection_cmd)

    return eval_commands


### END SWE-FFICIENCY EDITS

BUILD_TIMEOUT_CONFIGS_OVERRIDES = {
    "scikit-learn/scikit-learn": 600,
    "pandas-dev/pandas": None,
    "pydata/xarray": 3600,
}


def make_test_spec(instance: SWEfficiencyInstance, observed_versions=None) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance[KEY_INSTANCE_ID]
    # if there's capital letters in the repo name, convert to lowercase
    if instance_id != instance_id.lower():
        print(
            f"Instance ID {instance_id} contains capital letters. Converting to lowercase."
        )
        instance_id = instance_id.lower()
    repo = instance["repo"].lower()
    version = instance["version"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    hints_text = instance["hints_text"]  # Unused
    test_patch = instance["test_patch"]

    build_timeout = BUILD_TIMEOUT_CONFIGS_OVERRIDES.get(repo)

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    try:
        pass_to_pass = _from_json_or_obj(PASS_TO_PASS)
    except Exception as e:
        # print(f"Error parsing PASS_TO_PASS for instance {instance_id}: {e}. PASS_TO_PASS: {instance[PASS_TO_PASS]}")
        pass_to_pass = []

    try:
        fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)
    except Exception as e:
        # print(f"Error parsing FAIL_TO_PASS for instance {instance_id}: {e}. FAIL_TO_PASS: {instance[FAIL_TO_PASS]}")
        fail_to_pass = []

    try:
        covering_tests = _from_json_or_obj("covering_tests")
    except Exception as e:
        # print(f"Error parsing covering_tests for instance {instance_id}: {e}. covering_tests: {instance['covering_tests']}")
        covering_tests = []

    env_name = "testbed"
    treesitter_env_name = "treesitter"
    repo_directory = f"/{env_name}"

    if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
        raise NotImplementedError(f"Version {version} not implemented for repo {repo}")

    if observed_versions is not None:
        if version in observed_versions:
            raise RuntimeError(f"Version has already been observed: {version}")
        observed_versions.add(version)

    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    # --- Instance-level spec overrides (must be BEFORE make_repo_script_list) ---

    # Astropy 3.0/3.1/3.2 on env image 555bee985386fd03d06a6f:
    # The bare `pip` command resolves to the base env's Python 3.11 pip,
    # which installs packages to /opt/miniconda3/lib/python3.11/site-packages.
    # But the testbed conda env uses Python 3.7, whose `python` binary can ONLY
    # see /opt/miniconda3/envs/testbed/lib/python3.7/site-packages.  So any
    # package installed with bare `pip` is invisible to the actual runtime.
    #
    # Fix: use `python -m pip` (which uses the testbed env's Python 3.7 pip).
    # Python 3.7 still supports the setuptools "develop" egg-link mechanism,
    # so `pip install -e` works correctly (unlike on Python 3.11 where it
    # silently produces a non-importable package).  `--no-build-isolation` is
    # needed because ah_bootstrap.py (3.0/3.1) and extension_helpers/ (3.2) are
    # local source-tree modules that pip's build isolation cannot see.
    #
    # 3.2 additionally needs jinja2 (imported by astropy/modeling/setup_package.py
    # during metadata prep) and extension-helpers (setup.py line 67 imports it;
    # extension_helpers/ is a git submodule — an empty directory after clone —
    # so --no-build-isolation can't find it either).  Both are not in the SPECS
    # pip_packages for 3.2.
    _astropy_cfg = {
        "3.0": {},
        "3.1": {},
        "3.2": {"pre_pkgs": ["jinja2", "extension-helpers", "wheel"]},
    }
    if instance_id.startswith("astropy__astropy") and version in _astropy_cfg:
        specs = dict(specs)
        pkgs = _astropy_cfg[version].get("pre_pkgs", [])
        if pkgs:
            pre_installs = " && ".join(
                f"python -m pip install '{p}'" for p in pkgs
            )
            specs["install"] = (
                f"{pre_installs} && "
                "python -m pip install --no-build-isolation -e '.[dev_all]'"
            )
        else:
            specs["install"] = (
                "python -m pip install --no-build-isolation -e '.[dev_all]'"
            )

    # pandas-47781 (SPECS 1.4): env image c7de597aac5fcdac95013b (from Docker
    # cache) has a corrupted non-editable pandas install where the compiled
    # parsing.cpython-38-*.so is missing the format_is_iso symbol that the
    # corresponding Python source imports.  Switch to editable install (-e)
    # so the source tree's .so files (built by build_ext --inplace) are used
    # directly, avoiding the stale wheel in site-packages.
    if instance_id == "pandas-dev__pandas-47781":
        specs = dict(specs)
        specs["install"] = specs["install"].replace(
            "pip install --no-build-isolation . -c /tmp/constraints.txt",
            "pip install --no-build-isolation --editable . -c /tmp/constraints.txt",
        )

    # pandas-53731 (SPECS 2.0): the SPECS install uses cython<3 (0.29.x)
    # but the base commit's setup.py line 408 has a hard version check:
    #   RuntimeError: Cannot cythonize with old Cython version
    #                  (0.29.37 installed, needs 3.0.0)
    # Our previous approach (sed on pyproject.toml) only relaxed pip's
    # build requirement, not setup.py's runtime check.  Instead, upgrade
    # the cython constraint from <3 to >=3.0,<3.1 so both pip and setup.py
    # are satisfied.  The .pyx files in this PR are backward-compatible
    # with cython 3.x (the PR just fixes a function-signature warning).
    if instance_id == "pandas-dev__pandas-53731":
        specs = dict(specs)
        specs["install"] = specs["install"].replace(
            "cython<3",
            "cython>=3.0,<3.1"
        )

    # -------------------------------------------------------------------------

    repo_script_list = make_repo_script_list(
        specs,
        repo,
        repo_directory,
        base_commit,
        env_name,
        treesitter_env_name=treesitter_env_name,
    )
    env_script_list = make_env_script_list(
        instance, specs, env_name, treesitter_env_name=treesitter_env_name
    )
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    coverage_script_list = make_coverage_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    meaningful_edit_script_list = make_meaningful_edit_script_list(
        instance, specs, treesitter_env_name, repo_directory, base_commit, test_patch
    )

    # Note that this needs to be a function
    performance_script_list = make_performance_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    correctness_script_list = get_correctness_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )

    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    workload_text = instance.get("workload", "")
    if workload_text.strip() == "nan" or workload_text.strip() == "":
        workload_text = None

    # --- Filter network/DB-dependent tests from single_thread_tests ---
    # These test files need live connections to remote services (Google
    # BigQuery, AWS S3, GCS, SQL databases, etc.) that are unavailable in
    # offline evaluation containers.  In older pandas versions, the tests
    # inside these files are not consistently marked with
    # @pytest.mark.network, so the `-m "not network and not db"` pytest
    # filter in correctness.sh does not exclude them.  Left in the
    # single-thread list they hang indefinitely waiting for network
    # connections (especially problematic behind China's firewall).
    #
    # Removing them from single_thread_tests means they will run in the
    # parallel phase instead, where they are still subject to
    # `-m "not network and not db"` — if the markers work they are
    # skipped; if not, they fail fast (ConnectionError/Timeout) rather
    # than hanging because the parallel phase moves on.
    _NETWORK_DB_TEST_PATTERNS = (
        "test_gbq.py",       # Google BigQuery
        "test_s3.py",        # AWS S3
        "test_gcs.py",       # Google Cloud Storage
        "test_sql.py",       # SQL database connections
        "test_html.py",      # pd.read_html() fetches URLs
        "test_clipboard.py", # may access network resources
        "test_network.py",   # io/parser: explicit network tests
    )
    if instance["repo"] == "pandas-dev/pandas":
        raw_st = instance.get("single_thread_tests", [])
        # ``raw_st`` may be a numpy array (from parquet), whose truth value
        # is ambiguous when it has >1 element — must test length, not truthiness.
        if hasattr(raw_st, '__len__') and len(raw_st) > 0:
            single_thread_tests = [
                t for t in raw_st
                if not t.endswith(_NETWORK_DB_TEST_PATTERNS)
            ]
        else:
            single_thread_tests = []
    else:
        single_thread_tests = instance.get("single_thread_tests", [])

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        # SWE-fficiency changes.
        coverage_script_list=coverage_script_list,
        meaningful_edit_script_list=meaningful_edit_script_list,
        workload=workload_text,
        covering_tests=covering_tests,
        performance_script_list=performance_script_list,
        performance_profiling_script_list=make_performance_profiling_script_list(
            instance, specs, env_name, repo_directory, base_commit, test_patch
        ),
        correctness_script_list=correctness_script_list,
        introspection_guard_script_list=get_introspection_guard_cmds(
            instance, specs, env_name, repo_directory, base_commit, test_patch
        ),
        build_timeout=build_timeout,
        single_thread_tests=single_thread_tests,
        base_commit=base_commit,
    )
