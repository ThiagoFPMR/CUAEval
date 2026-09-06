#!/usr/bin/env python3
"""Move directories between Backblaze B2 and local disk, idempotently.

Backblaze B2 is the only object store CUAEval speaks to, in both directions:

  * DOWNLOAD pulls model weights from B2 and is used in two places — locally,
    imported, to stage weights before a LOCAL docker serve; and on a vast.ai
    instance, where CUAEval rsyncs this file (as part of the deploy bundle) and
    runs it with the instance's python to stage weights before a REMOTE process
    serve.

  * UPLOAD is used by results.py to mirror the OSWorld results tree off the
    host's EBS volume while a run is still in progress — benchmark output
    otherwise lives on exactly one un-backed-up disk.

B2 is reached through its S3-compatible API, so a bucket is addressed as a
b2://bucket/prefix URI plus an endpoint (B2_S3_ENDPOINT, or passed explicitly),
and credentials come from B2_ACCESS_KEY_ID / B2_SECRET_ACCESS_KEY — never baked
in. One credential set covers both halves.

Both directions are size-compared and resumable: an object already present at
the destination with the same byte size is skipped, so re-running after an
interruption (or syncing the same tree every 10 minutes) only moves what's new.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

# Backblaze B2 (S3-compatible API) settings. The endpoint can be passed
# explicitly (results_sync.endpoint_url, --endpoint-url) or come from the env.
B2_ENDPOINT_ENV = "B2_S3_ENDPOINT"        # e.g. https://s3.us-west-004.backblazeb2.com
B2_KEY_ID_ENV = "B2_ACCESS_KEY_ID"        # Backblaze application keyID
B2_APP_KEY_ENV = "B2_SECRET_ACCESS_KEY"   # Backblaze applicationKey
B2_REGION_ENV = "B2_REGION"               # override the region parsed from the endpoint

# Env var names that must reach a vast.ai instance for it to stage weights itself
# (see backends/remote.py, which writes them to a 0600 b2.env on the box).
B2_ENV_VARS = (B2_KEY_ID_ENV, B2_APP_KEY_ENV, B2_ENDPOINT_ENV, B2_REGION_ENV)


def parse_b2_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("b2://"):
        raise ValueError(f"Expected b2:// URI, got {uri!r}")
    bucket, _, key = uri[len("b2://"):].partition("/")
    return bucket, key.rstrip("/")


def _transfer_config():
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )


def _list_objects(b2, bucket: str, prefix: str) -> list[tuple[str, int]]:
    objs: list[tuple[str, int]] = []
    paginator = b2.get_paginator("list_objects_v2")
    # An empty prefix means the bucket root; "/" would match nothing.
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/" if prefix else ""):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # skip pseudo-directory markers
                continue
            objs.append((key, obj["Size"]))
    return sorted(objs)


def _b2_region(endpoint_url: str) -> str:
    """Backblaze endpoints look like s3.<region>.backblazeb2.com; SigV4 still needs
    a region name. Prefer an explicit override, else parse it out of the host."""
    if os.environ.get(B2_REGION_ENV):
        return os.environ[B2_REGION_ENV]
    host = urlparse(endpoint_url).hostname or ""
    parts = host.split(".")
    if len(parts) >= 4 and parts[0] == "s3":
        return parts[1]
    return "us-east-1"


def _b2_client(workers: int, endpoint_url: str | None = None):
    """Backblaze B2 client, via B2's S3-compatible API.

    Requires an endpoint (from the argument or B2_S3_ENDPOINT) and the B2_*
    credentials. Used by both the download and the upload half.
    """
    import boto3
    from botocore.config import Config

    endpoint_url = endpoint_url or os.environ.get(B2_ENDPOINT_ENV)
    if not endpoint_url:
        raise RuntimeError(
            "Backblaze endpoint is not set; pass results_sync.endpoint_url / "
            f"--endpoint-url or export {B2_ENDPOINT_ENV} "
            "(e.g. https://s3.us-west-004.backblazeb2.com)"
        )
    key_id = os.environ.get(B2_KEY_ID_ENV)
    app_key = os.environ.get(B2_APP_KEY_ENV)
    if not key_id or not app_key:
        raise RuntimeError(
            f"Backblaze credentials are not set; export {B2_KEY_ID_ENV} and "
            f"{B2_APP_KEY_ENV} (the application keyID and applicationKey)"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=_b2_region(endpoint_url),
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(
            max_pool_connections=max(workers * 16, 16),
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )


def download_model(b2_src: str, model_dir: str | Path, workers: int = 4,
                   overwrite: bool = False, endpoint_url: str | None = None) -> None:
    """Download every object under b2_src into model_dir. Raises on any failure."""
    bucket, prefix = parse_b2_uri(b2_src)
    model_dir = Path(model_dir)

    b2 = _b2_client(workers, endpoint_url)

    objs = _list_objects(b2, bucket, prefix)
    if not objs:
        raise RuntimeError(f"No objects under b2://{bucket}/{prefix}")
    total = sum(sz for _, sz in objs)
    print(f"[b2] SRC=b2://{bucket}/{prefix} DEST={model_dir} "
          f"{len(objs)} files, {total / 1e9:.1f} GB, workers={workers}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = _transfer_config()
    done = 0
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []

    def _rel(key: str) -> str:
        # An empty prefix means the bucket root, where keys are already relative.
        return key[len(prefix) + 1:] if prefix else key

    def _one(item: tuple[str, int]) -> None:
        nonlocal done
        key, size = item
        rel = _rel(key)
        dest = model_dir / rel
        if not overwrite:
            try:
                if dest.stat().st_size == size:
                    with lock:
                        done += 1
                        print(f"[skip] {done}/{len(objs)} {rel}", flush=True)
                    return
            except OSError:
                pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        b2.download_file(bucket, key, str(tmp), Config=cfg)
        os.replace(tmp, dest)
        with lock:
            done += 1
            print(f"[down] {done}/{len(objs)} {rel} ({size / 1e6:.1f} MB)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it): it for it in objs}
        for fut in as_completed(futs):
            key, _ = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                rel = _rel(key)
                failures.append((rel, str(exc)))
                print(f"[fail] {rel}: {exc}", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} of {len(objs)} files failed to download")
    print(f"[b2] done: {len(objs)} files -> {model_dir}", flush=True)


def _excluded(rel: str, patterns: Sequence[str]) -> bool:
    """Match a glob against either the path relative to the sync root or the bare
    filename, so both 'recording.mp4' and '*/screenshots/*' behave as expected."""
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch(rel, pat) or fnmatch(name, pat) for pat in patterns)


def _iter_local_files(root: Path, exclude: Sequence[str]) -> list[tuple[Path, str, int]]:
    """(path, key-relative path, size) for every file under root, sorted."""
    out: list[tuple[Path, str, int]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.endswith(".part") or _excluded(rel, exclude):
            continue
        try:
            out.append((p, rel, p.stat().st_size))
        except OSError:      # vanished mid-walk (the runner is writing underneath us)
            continue
    return out


def upload_dir(local_dir: str | Path, b2_dest: str, *, workers: int = 8,
               exclude: Sequence[str] = (), overwrite: bool = False,
               quiet: bool = False, endpoint_url: str | None = None) -> tuple[int, int]:
    """Mirror local_dir into b2_dest (a b2://bucket/prefix on Backblaze B2).
    Returns (uploaded, skipped).

    Objects already at the destination with a matching byte size are skipped, so
    this is cheap to call repeatedly against a tree that only grows — which is
    exactly what the OSWorld runner produces (one directory per finished example,
    written once). Raises on any failed upload.
    """
    bucket, prefix = parse_b2_uri(b2_dest)
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"nothing to upload: {local_dir} does not exist")

    files = _iter_local_files(local_dir, exclude)
    if not files:
        return (0, 0)

    b2 = _b2_client(workers, endpoint_url)
    remote = {} if overwrite else {k: sz for k, sz in _list_objects(b2, bucket, prefix)}

    def _key(rel: str) -> str:
        return f"{prefix}/{rel}" if prefix else rel

    pending = [(p, rel, sz) for p, rel, sz in files if remote.get(_key(rel)) != sz]
    skipped = len(files) - len(pending)
    if not pending:
        return (0, skipped)

    total = sum(sz for _, _, sz in pending)
    if not quiet:
        print(f"[b2] SRC={local_dir} DEST=b2://{bucket}/{prefix} "
              f"{len(pending)} new/changed files ({total / 1e6:.1f} MB), "
              f"{skipped} unchanged, workers={workers}", flush=True)

    cfg = _transfer_config()
    done = 0
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []

    def _one(item: tuple[Path, str, int]) -> None:
        nonlocal done
        path, rel, size = item
        try:
            b2.upload_file(str(path), bucket, _key(rel), Config=cfg)
        except FileNotFoundError:
            return   # deleted between the walk and the upload; nothing to save
        with lock:
            done += 1
            if not quiet:
                print(f"[up] {done}/{len(pending)} {rel} ({size / 1e6:.1f} MB)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it): it for it in pending}
        for fut in as_completed(futs):
            _, rel, _ = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((rel, str(exc)))
                print(f"[fail] {rel}: {exc}", flush=True)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(pending)} files failed to upload "
            f"(first: {failures[0][0]}: {failures[0][1]})"
        )
    return (done, skipped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b2-src", required=True, help="b2://bucket/prefix to download")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("B2_WORKERS", "4")))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--endpoint-url", default=None,
                    help=f"Backblaze S3 endpoint (default: ${B2_ENDPOINT_ENV})")
    args = ap.parse_args()
    try:
        download_model(args.b2_src, args.model_dir, args.workers, args.overwrite,
                       args.endpoint_url)
    except Exception as exc:  # noqa: BLE001
        print(f"[b2] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
