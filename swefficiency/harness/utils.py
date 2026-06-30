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

import json
import os
import re
from argparse import ArgumentTypeError
from functools import cache
from pathlib import Path
from typing import cast

import requests
from datasets import Dataset
from dotenv import load_dotenv
import pyarrow.parquet as pq

from swefficiency.harness.constants import (
    KEY_INSTANCE_ID,
    MAP_REPO_TO_ENV_YML_PATHS,
    MAP_REPO_TO_REQS_PATHS,
    NON_TEST_EXTS,
    SWE_BENCH_URL_RAW,
    SWEfficiencyInstance,
)

load_dotenv()


def load_swefficiency_dataset(
    name="swefficiency/swefficiency", split="test", instance_ids=None, revision=None
) -> list[SWEfficiencyInstance]:
    """
    Load SWE-bench dataset from Hugging Face Datasets or local .json/.jsonl file or local directory
    """
    from datasets import load_dataset
    
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)
    # Load from local .json/.jsonl file
    if name.endswith(".json"):
        dataset = json.loads(Path(name).read_text())
        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    elif name.endswith(".jsonl"):
        dataset = [
            json.loads(instance) for instance in Path(name).read_text().splitlines()
        ]
        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    elif Path(name).is_dir():
        # Check if this is a directory with parquet files
        parquet_files = sorted(Path(name).glob("*.parquet"))
        if parquet_files:
            # Load from local parquet files directly using pyarrow
            all_data = []
            for parquet_file in parquet_files:
                table = pq.read_table(parquet_file)
                all_data.extend(table.to_pylist())
            dataset = all_data
            dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
        else:
            # Load from local directory (Hugging Face cache format)
            dataset = cast(Dataset, load_dataset(name, split=split, revision=revision))
            dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    else:
        # Load from Hugging Face Datasets
        if name.lower() in {"swefficiency"}:
            name = "swefficiency/swefficiency"
        elif name.lower() in {
            "swefficiency-lite",
        }:
            name = "swefficiency/swefficiency-lite"
        dataset = cast(Dataset, load_dataset(name, split=split, revision=revision))
        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    if instance_ids:
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [
            instance
            for instance in dataset
            if instance[KEY_INSTANCE_ID] in instance_ids
        ]
    return [cast(SWEfficiencyInstance, instance) for instance in dataset]


### MARK - Patch Correction
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


def get_first_idx(charlist):
    """Get index of first occurrence of "-" or "+" in charlist"""
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


def get_last_idx(charlist):
    """Get index of last occurrence of "-" or "+" in charlist"""
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


def strip_content(hunk):
    """Remove trailing non +/- lines and trailing whitespace per line per hunk"""
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    """Recalculate hunk start/end position and diff delta"""
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta


def extract_minimal_patch(model_patch):
    """
    Wrapper function that takes hunk and
    * Removes trailing non +/- lines and trailing whitespace per line per hunk
    * Recalculates hunk start/end position and diff delta
    * Returns new patch
    """
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def has_attribute_or_import_error(log_before):
    """
    Check to see if Attribute/Import-prefix is in log text

    Args:
        log_before (str): Validation log text before patch application
    """
    log_before = log_before.lower()

    if any([x in log_before for x in ["attribute", "import"]]):

        def get_lines_with_word(text, target_word):
            # Function to extract line(s) that contains target_word
            text, target_word = text.lower(), target_word.lower()
            lines, hits = text.split("\n")[::-1], []
            for line in lines:
                if target_word in line:
                    hits.append(line)
            return hits

        # Get line with Attribute/Import error
        lines_1 = get_lines_with_word(log_before, "attribute")
        lines_2 = get_lines_with_word(log_before, "import")
        lines_1 = " ".join(lines_1)
        lines_2 = " ".join(lines_2)

        if any([(x in lines_1 or x in lines_2) for x in ["error", "fail"]]):
            return True
    return False


ENV_YML_CACHE_DIR = Path("/home/hyx/swefficiency/cache/environment_yml")


@cache
def get_environment_yml_by_commit(repo: str, commit: str, env_name: str) -> str:
    ENV_YML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try disk cache first for each possible path
    for req_path in MAP_REPO_TO_ENV_YML_PATHS[repo]:
        cache_key = f"{repo.replace('/', '_')}_{commit}_{req_path.replace('/', '_')}"
        cache_file = ENV_YML_CACHE_DIR / cache_key
        if cache_file.exists():
            # print(f"[ENV CACHE] Using cached {req_path} for {repo}@{commit}")
            reqs_text = cache_file.read_text(encoding="utf-8")
            break
    else:
        # Cache miss: fetch from network
        session = requests.Session()
        retry = requests.adapters.Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504, 408, 429]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        reqs_text = None
        for req_path in MAP_REPO_TO_ENV_YML_PATHS[repo]:
            reqs_url = os.path.join(SWE_BENCH_URL_RAW, repo, commit, req_path)
            try:
                reqs = session.get(reqs_url, timeout=60)
                if reqs.status_code == 200:
                    reqs_text = reqs.text
                    # Save to disk cache
                    cache_key = f"{repo.replace('/', '_')}_{commit}_{req_path.replace('/', '_')}"
                    cache_file = ENV_YML_CACHE_DIR / cache_key
                    cache_file.write_text(reqs_text, encoding="utf-8")
                    print(f"[ENV CACHE] Saved {req_path} for {repo}@{commit} to disk cache")
                    break
            except Exception as e:
                print(f"[ENV CACHE] Failed to fetch {reqs_url}: {e}")
                continue

        if reqs_text is None:
            raise ValueError(
                f"Could not find environment.yml at paths {MAP_REPO_TO_ENV_YML_PATHS[repo]} for repo {repo} at commit {commit}"
            )

    # Parse YAML using ruamel.yaml for proper structure manipulation
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    env_data = yaml.load(reqs_text)
    env_data["name"] = env_name
    
    # Process dependencies to handle pip extras syntax
    if "dependencies" in env_data:
        pip_deps_with_extras = []
        new_dependencies = []
        
        for dep in env_data["dependencies"]:
            if isinstance(dep, str):
                # Check for pip extras syntax: package[extra]
                if "[" in dep and "]" in dep:
                    import re
                    match = re.match(r'^([^\[]+)\[([^\]]+)\]([=<>!~]+[^\s]*)?', dep)
                    if match:
                        package = match.group(1)
                        extras = match.group(2)
                        version_spec = match.group(3) if match.group(3) else ""
                        # Keep original syntax for pip: section
                        pip_deps_with_extras.append(f"{package}[{extras}]{version_spec}")
                        continue
            new_dependencies.append(dep)
        
        # If we have pip dependencies with extras, add them to pip: section
        if pip_deps_with_extras:
            # Check if pip: section already exists
            pip_section = None
            for i, dep in enumerate(new_dependencies):
                if isinstance(dep, dict) and "pip" in dep:
                    pip_section = dep
                    break
            
            if pip_section:
                # Add to existing pip: section
                for pip_dep in pip_deps_with_extras:
                    if pip_dep not in pip_section["pip"]:
                        pip_section["pip"].append(pip_dep)
            else:
                # Create new pip: section
                # Ensure pip is in dependencies
                if "pip" not in new_dependencies:
                    new_dependencies.insert(0, "pip")
                # Add pip: section
                new_dependencies.append({"pip": pip_deps_with_extras})
        
        env_data["dependencies"] = new_dependencies
    
    # Convert back to string
    from io import StringIO
    string_stream = StringIO()
    yaml.dump(env_data, string_stream)
    return string_stream.getvalue()


def get_environment_yml(instance: SWEfficiencyInstance, env_name: str) -> str:
    """
    Get environment.yml for given task instance

    Args:
        instance (dict): SWE Bench Task instance
        env_name (str): Rename retrieved environment.yml to this name
    Returns:
        environment.yml (str): Returns environment.yml as string
    """
    # Attempt to find environment.yml at each path based on task instance's repo

    commit = (
        instance["environment_setup_commit"]
        if "environment_setup_commit" in instance
        else instance["base_commit"]
    )

    env_yml = get_environment_yml_by_commit(instance["repo"], commit, env_name)

    return env_yml


@cache
def get_requirements_by_commit(repo: str, commit: str) -> str:
    # Configure retry session for network stability
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=10,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504, 408, 429]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    for req_path in MAP_REPO_TO_REQS_PATHS[repo]:
        reqs_url = os.path.join(SWE_BENCH_URL_RAW, repo, commit, req_path)
        reqs = session.get(reqs_url, timeout=300)  # Increased to 120 seconds
        if reqs.status_code == 200:
            break
    else:
        raise ValueError(
            f"Could not find requirements.txt at paths {MAP_REPO_TO_REQS_PATHS[repo]} for repo {repo} at commit {commit}"
        )

    lines = reqs.text
    original_req = []
    additional_reqs = []
    req_dir = "/".join(req_path.split("/")[:-1])
    exclude_line = lambda line: any(
        [line.strip().startswith(x) for x in ["-e .", "#", ".[test"]]
    )

    for line in lines.split("\n"):
        if line.strip().startswith("-r"):
            # Handle recursive requirements
            file_name = line[len("-r") :].strip()
            reqs_url = os.path.join(
                SWE_BENCH_URL_RAW,
                repo,
                commit,
                req_dir,
                file_name,
            )
            reqs = requests.get(reqs_url)
            if reqs.status_code == 200:
                for line_extra in reqs.text.split("\n"):
                    if not exclude_line(line_extra):
                        additional_reqs.append(line_extra)
        else:
            if not exclude_line(line):
                original_req.append(line)

    # Combine all requirements into single text body
    additional_reqs.append("\n".join(original_req))
    all_reqs = "\n".join(additional_reqs)

    return all_reqs


def get_requirements(instance: SWEfficiencyInstance) -> str:
    """
    Get requirements.txt for given task instance

    Args:
        instance (dict): task instance
    Returns:
        requirements.txt (str): Returns requirements.txt as string
    """
    # Attempt to find requirements.txt at each path based on task instance's repo
    commit = (
        instance["environment_setup_commit"]
        if "environment_setup_commit" in instance
        else instance["base_commit"]
    )

    return get_requirements_by_commit(instance["repo"], commit)


def get_test_directives(instance: SWEfficiencyInstance) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    if instance["repo"] == "swe-bench/humaneval":
        return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["test_patch"]
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance["repo"] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives


def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
