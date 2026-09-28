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

"""Move base/env images from the durable NFS store into the local (scratch) store.

The eval harness is pointed at the fast NVMe scratch store
(``SWE_PODMAN_STORE=local`` for the ``docker`` CLI shim +
``DOCKER_HOST=unix:///tmp/podman-scratch.sock`` for the SDK), which starts out
holding only the builder images.  The images that were built once and kept are
on the durable NFS store: ``sweb.base.<arch>:latest`` plus the 181
``sweb.env.<arch>.<hash>:latest`` the 498-instance dataset resolves to.

Without this module the harness falls through to *building* whatever it cannot
find -- and building ``sweb.base`` starts with ``FROM ubuntu:22.04``, which
requires docker hub.  This box has no route to ``registry-1.docker.io``, so the
run dies with::

    Error sweb.base.x86_64:latest: creating build container: initializing
    source docker://ubuntu:22.04: pinging container registry
    registry-1.docker.io: dial tcp 103.252.114.61:443: i/o timeout

This is the eval-side twin of ``ensure_env_in_local_store`` in
``OpenHandsBenchmarks/benchmarks/swefficiency/run_infer.py``; the transport is
identical, so both pipelines move images the same way::

    docker save <tag>   # SWE_PODMAN_STORE stripped -> default (NFS) store
      | docker load     # inherited SWE_PODMAN_STORE=local -> scratch store

**It is lazy, exactly like the infer side**: the shared ``sweb.base`` is moved
in up front (3GB, needed by any build), but an instance's ``sweb.env`` is moved
in only when that instance is about to start (``ensure_env_local``, called from
``run_validation.run_instance``) and dropped again when its last user finishes
(``release_env_for_instance``).  Moving all 181 envs before the first instance
would hold ~700GB of real (1260GB nominal) scratch space for hours and delay the
first result by the whole transfer; per-instance streaming keeps the resident
set at roughly the worker count instead.  The 498-instance dataset needs 211
*blocks* of consecutive same-env instances in dataset order, so laziness costs
almost no extra transfers (211 vs 181) while a resumed run only pays for the
instances it actually has left to do.

Everything is gated on ``SWEFFICIENCY_ENV_AUTO_STREAM=1`` *and* on the store
routing actually being the split NFS/scratch one, so an unset value keeps the
old build-from-scratch behaviour byte for byte.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker

# NFS store tags, keyed by their normalised (prefix-stripped) form.  Listing the
# NFS store is a CLI call (~1.4s); the env keys of a 498-instance dataset are
# looked up one by one, so cache the listing instead of shelling out per image.
_nfs_cache_lock = threading.Lock()
_nfs_cache: dict[str, str] | None = None

# Caps concurrent NFS->local streams.  A 6-8GB stream is NFS-bandwidth bound
# (measured ~45MB/s); past a couple of parallel streams the CLI only adds
# store-lock contention against the scratch store.  Same knob and default as the
# infer side (2).
STREAM_CONCURRENCY = max(
    1, int(os.environ.get("SWEFFICIENCY_ENV_STREAM_CONCURRENCY", "2"))
)
_stream_sem = threading.BoundedSemaphore(STREAM_CONCURRENCY)

# One lock per image tag, so two workers that need the *same* env at the same
# time queue behind each other instead of both racing a save|load of it (which
# would double NFS traffic and let one of the two loads lose).
_tag_locks_guard = threading.Lock()
_tag_locks: dict[str, threading.Lock] = {}


def _tag_lock(tag: str) -> threading.Lock:
    with _tag_locks_guard:
        return _tag_locks.setdefault(tag, threading.Lock())


# Reference counting for eviction: instance_id -> env image key, for the
# instances that are *in flight right now*.  Registered when an instance starts
# (register_env_user) and dropped when it ends; the env is removed only once no
# live instance needs it.  Same model as run_infer.py, and the reason lazy
# streaming does not thrash: instances that share an env sit next to each other
# in dataset order, so they are usually in flight together.
_workload_lock = threading.Lock()
_workload: dict[str, str] = {}


def stream_enabled() -> bool:
    """True when images should be moved NFS -> local instead of built.

    Requires the split-store routing, otherwise "the local store" *is* the NFS
    store and there is nothing to move.
    """
    return (
        os.environ.get("SWEFFICIENCY_ENV_AUTO_STREAM") == "1"
        and os.environ.get("SWE_PODMAN_STORE") == "local"
    )


def evict_enabled() -> bool:
    """True when a streamed image should be dropped once nothing needs it.

    Same flag the infer side uses (``SWEFFICIENCY_ENV_EVICT=1``), and only
    meaningful together with streaming: evicting envs the harness expects to be
    local, without being able to stream them back, would just be a slower way
    of failing.
    """
    return (
        os.environ.get("SWEFFICIENCY_ENV_EVICT") == "1" and stream_enabled()
    )


def repo_tag(image_key: str) -> str:
    """Normalise an image key to a repo:tag usable by docker save/images."""
    return image_key if ":" in image_key else f"{image_key}:latest"


def _local_has(client: docker.DockerClient, tag: str) -> bool:
    try:
        client.images.get(tag)
        return True
    except Exception:
        return False


def normalise(tag: str) -> str:
    """Strip the ``localhost/`` prefix podman tags locally-built images with.

    The harness refers to ``sweb.base.x86_64:latest`` / ``sweb.env.x86_64.<hash>:latest``
    while the store lists them as ``localhost/sweb.base...``; the docker-compat
    API resolves the short form, but a plain string comparison does not.
    """
    for prefix in ("localhost/", "docker.io/library/"):
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return tag


def _nfs_image_map() -> dict[str, str]:
    """NFS-store tags (default channel: no SWE_PODMAN_STORE), keyed normalised."""
    global _nfs_cache
    with _nfs_cache_lock:
        if _nfs_cache is not None:
            return _nfs_cache
    env = {k: v for k, v in os.environ.items() if k != "SWE_PODMAN_STORE"}
    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        print(f"[image stream] WARNING: cannot list the NFS store: {e}", flush=True)
        return {}
    tags = {}
    for line in proc.stdout.splitlines():
        tag = line.strip()
        if tag and "<none>" not in tag:
            tags[normalise(tag)] = tag
    with _nfs_cache_lock:
        _nfs_cache = tags
    return tags


def _stream_from_nfs(tag: str) -> tuple[bool, int, int, str, str]:
    """``docker save tag | docker load`` NFS -> local.

    Source = NFS store: the docker CLI reads SWE_PODMAN_STORE, so strip it for
    the save side.  Destination = local store: keep it for the load (inherited
    env).  Returns (ok, save_rc, load_rc, save_detail, load_detail) with the
    full stderr so failures are diagnosable.
    """
    save_env = {k: v for k, v in os.environ.items() if k != "SWE_PODMAN_STORE"}
    save = subprocess.Popen(
        ["docker", "save", tag],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=save_env,
    )
    assert save.stdout is not None
    load = subprocess.Popen(
        ["docker", "load"],
        stdin=save.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    save.stdout.close()
    _, save_err = save.communicate()
    _, load_err = load.communicate()
    save_detail = (save_err or b"").decode(errors="replace")
    load_detail = (load_err or b"").decode(errors="replace")
    return (
        save.returncode == 0 and load.returncode == 0,
        save.returncode,
        load.returncode,
        save_detail,
        load_detail,
    )


def _stream_one(client: docker.DockerClient, tag: str, position: str) -> bool:
    """Stream a single tag.  Returns True when ``client`` can see it afterwards."""
    with _stream_sem:
        # Re-check under the semaphore: another thread may have moved it while
        # we waited.
        if _local_has(client, tag):
            return True
        for attempt_no in (1, 2):
            print(
                f"[image stream] {position} NFS -> local: {tag}"
                + ("" if attempt_no == 1 else f" (retry {attempt_no}/2)"),
                flush=True,
            )
            started = time.time()
            ok, save_rc, load_rc, save_detail, load_detail = _stream_from_nfs(tag)
            if ok:
                # The load writes straight into the graphroot from a CLI
                # process; the SDK service can briefly lag before the freshly
                # committed image shows up.  Poll for it.
                for _ in range(10):
                    if _local_has(client, tag):
                        print(
                            f"[image stream]     done in {time.time() - started:.0f}s",
                            flush=True,
                        )
                        return True
                    time.sleep(0.5)
                print(
                    f"[image stream] WARNING: {tag} loaded but not visible to the "
                    "SDK after 5s — retrying",
                    flush=True,
                )
                continue
            print(
                f"[image stream] WARNING: stream failed for {tag} "
                f"(docker save rc={save_rc}, docker load rc={load_rc})",
                flush=True,
            )
            for detail in (save_detail, load_detail):
                if detail.strip():
                    print(f"[image stream]   {detail.strip()[-800:]}", flush=True)
        return False


def ensure_images_local(client: docker.DockerClient, image_keys) -> tuple[int, list]:
    """Make sure every image in ``image_keys`` is in the store ``client`` sees.

    Missing locally + present on NFS  -> streamed (this is the whole point).
    Missing both + streaming disabled -> left alone; the caller still hits the
      normal BuildImageError / "Building ..." path exactly as before.
    Already present locally           -> untouched no-op.

    Returns (number streamed, keys that are nowhere to be found).
    """
    keys = sorted({repo_tag(k) for k in image_keys})
    if not stream_enabled():
        return 0, []

    nfs_tags = _nfs_image_map()
    todo = [k for k in keys if not _local_has(client, k)]
    if not todo:
        print(f"[image stream] all {len(keys)} required images already local.")
        return 0, []

    # Stream the tag as the NFS store spells it (``localhost/...``); the load
    # keeps that name and the SDK resolves the harness' short form to it.
    streamable = [nfs_tags[normalise(k)] for k in todo if normalise(k) in nfs_tags]
    absent = [k for k in todo if normalise(k) not in nfs_tags]
    print(
        f"[image stream] {len(todo)}/{len(keys)} required images missing locally; "
        f"{len(streamable)} available on the NFS store.",
        flush=True,
    )
    for k in absent:
        print(
            f"[image stream] WARNING: {k} is on neither store — the normal "
            "build path will handle it.",
            flush=True,
        )

    # Streams run concurrently but bounded by the semaphore.  In practice this
    # path only ever carries the single shared base image now (the envs are
    # streamed lazily per instance), but keeping it generic costs nothing.
    streamed = 0
    with ThreadPoolExecutor(max_workers=STREAM_CONCURRENCY) as executor:
        futures = [
            executor.submit(_stream_one, client, tag, f"({i}/{len(streamable)})")
            for i, tag in enumerate(streamable, 1)
        ]
        for future in as_completed(futures):
            if future.result():
                streamed += 1
    print(f"[image stream] streamed {streamed}/{len(streamable)} images.", flush=True)
    return streamed, absent


def streamable_keys(image_keys) -> set[str]:
    """Which of ``image_keys`` the NFS store can supply (prefix-insensitive).

    Used by ``build_env_images`` to decide which env images must *not* be built:
    they are streamed later, per instance, instead.
    """
    if not stream_enabled():
        return set()
    nfs_tags = _nfs_image_map()
    return {k for k in image_keys if normalise(repo_tag(k)) in nfs_tags}


def ensure_env_local(
    client: docker.DockerClient, instance_id: str, env_key: str
) -> None:
    """Lazy per-instance half: make ``env_key`` local for ``instance_id``.

    No-op when the image is already there -- which is the common case for the
    2nd..Nth instance of the same env.  Otherwise the caller *blocks* until the
    transfer finishes: the instance image is built ``FROM`` this env, so there is
    nothing else useful it could be doing.  A transfer that cannot be completed
    falls through to the original failure mode (``client.images.get`` raising) so
    the instance is reported as an error rather than silently mis-measured.
    """
    register_env_user(instance_id, env_key)
    if not stream_enabled():
        return

    tag = repo_tag(env_key)
    if _local_has(client, tag):
        return

    with _tag_lock(tag):
        # Re-check under the per-tag lock: a worker that needed the same env
        # may have just finished streaming it.
        if _local_has(client, tag):
            return
        source = _nfs_image_map().get(normalise(tag))
        if source is None:
            print(
                f"[image stream] WARNING: {tag} is on neither store — the normal "
                "build path will handle it.",
                flush=True,
            )
            return
        if not _stream_one(client, source, f"[{instance_id}]"):
            print(
                f"[image stream] WARNING: could not stream {tag} for {instance_id}",
                flush=True,
            )
            # Surface it the way a missing image has always been surfaced.
            client.images.get(tag)


def register_env_user(instance_id: str, env_key: str) -> None:
    """Record that ``instance_id`` is using ``env_key`` right now."""
    if not evict_enabled():
        return
    with _workload_lock:
        _workload[instance_id] = env_key


def release_env_for_instance(instance_id: str, client: docker.DockerClient) -> None:
    """Drop one instance's claim; rmi its env image once nothing needs it.

    The authoritative copy on the NFS store is never touched -- this only undoes
    the NFS -> local move, so the scratch store does not accumulate a dataset's
    worth of env images (which is what the infer side's
    ``release_env_for_instance`` does).  Removing an env image that a built
    ``sweb.eval`` image descends from is safe: the child keeps its own layers.
    An env whose next user has not started yet is simply streamed again when it
    does; in dataset order those users are adjacent, so this is rare (211 blocks
    for 181 envs over the whole 498).
    """
    if not evict_enabled():
        return
    with _workload_lock:
        env_key = _workload.pop(instance_id, None)
        still_needed = env_key is not None and env_key in _workload.values()
    if env_key is None or still_needed:
        return

    tag = repo_tag(env_key)
    last: Exception | None = None
    for candidate in (tag, f"localhost/{tag}"):
        try:
            client.images.remove(candidate, force=True)
        except Exception as e:  # noqa: BLE001 - never fatal, just kept around
            last = e
            continue
        print(
            f"[image stream] evicted from local store: {candidate} "
            f"(last user {instance_id})",
            flush=True,
        )
        return
    print(
        f"[image stream] eviction skipped for {tag}: {last}",
        flush=True,
    )
