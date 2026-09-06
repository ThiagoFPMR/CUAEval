# Custom OSWorld evaluation metas

An OSWorld *meta* is a JSON map of `{domain: [example_id, ...]}` selecting which
tasks a run executes. Upstream OSWorld ships several (`test_all.json`,
`test_small.json`, ...) inside its own `evaluation_examples/`, but the ones **we**
author don't come with a fresh clone — so we keep them here, version-controlled,
and `cuaeval bootstrap` copies every file in this directory into the checkout's
`evaluation_examples/` (see `cuaeval/bootstrap.py::_vendor_metas`). Only custom
metas live here, so vendoring never clobbers an upstream file.

A plan references one by its path *inside the checkout*, e.g.:

```yaml
defaults:
  meta: evaluation_examples/test_nogdrive.json
```

## Files

- **`test_nogdrive.json`** — `test_all.json` minus the 8 Google-Drive `multi_apps`
  tasks (361 tasks across 10 domains). Google Drive tasks need live credentials /
  network state that makes them flaky-to-impossible in a headless VM, so they're
  dropped from the default run.

To add another subset, drop a `{domain: [ids]}` JSON file here and point a plan's
`meta:` at `evaluation_examples/<your-file>.json`. To restrict a single run to a
subset of a meta's domains without a new file, use the plan job's `domains:` key.
