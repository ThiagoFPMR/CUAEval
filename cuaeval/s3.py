#!/usr/bin/env python3
"""Download a model directory from S3, preserving layout, idempotently.

Vendored/adapted from DomainPruning/download_pruned_s3.py so CUAEval is
self-contained. This file is used in two places:

  * locally, imported, to stage weights before a LOCAL docker serve; and
  * on a vast.ai instance, where CUAEval rsyncs it (as part of the deploy
    bundle) and runs it with the instance's python to stage weights before a
    REMOTE process serve.

Each object under <s3-src>/<rel> lands at <model-dir>/<rel>. Big safetensors
shards come down as parallel multipart transfers; a local file already present
with the same byte size is skipped, so an interrupted run resumes cleanly.

Credentials come from the standard AWS env vars (AWS_ACCESS_KEY_ID / _SECRET /
_DEFAULT_REGION) — never baked in.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key.rstrip("/")


def _transfer_config():
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )


def _list_objects(s3, bucket: str, prefix: str) -> list[tuple[str, int]]:
    objs: list[tuple[str, int]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # skip pseudo-directory markers
                continue
            objs.append((key, obj["Size"]))
    return sorted(objs)


def download_model(s3_src: str, model_dir: str | Path, workers: int = 4,
                   overwrite: bool = False) -> None:
    """Download every object under s3_src into model_dir. Raises on any failure."""
    bucket, prefix = parse_s3_uri(s3_src)
    model_dir = Path(model_dir)

    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", config=Config(
        max_pool_connections=max(workers * 16, 16),
        retries={"max_attempts": 10, "mode": "adaptive"},
    ))

    objs = _list_objects(s3, bucket, prefix)
    if not objs:
        raise RuntimeError(f"No objects under s3://{bucket}/{prefix}")
    total = sum(sz for _, sz in objs)
    print(f"[s3] SRC=s3://{bucket}/{prefix} DEST={model_dir} "
          f"{len(objs)} files, {total / 1e9:.1f} GB, workers={workers}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = _transfer_config()
    done = 0
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []

    def _one(item: tuple[str, int]) -> None:
        nonlocal done
        key, size = item
        rel = key[len(prefix) + 1:]
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
        s3.download_file(bucket, key, str(tmp), Config=cfg)
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
                rel = key[len(prefix) + 1:]
                failures.append((rel, str(exc)))
                print(f"[fail] {rel}: {exc}", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} of {len(objs)} files failed to download")
    print(f"[s3] done: {len(objs)} files -> {model_dir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-src", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("S3_WORKERS", "4")))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        download_model(args.s3_src, args.model_dir, args.workers, args.overwrite)
    except Exception as exc:  # noqa: BLE001
        print(f"[s3] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
