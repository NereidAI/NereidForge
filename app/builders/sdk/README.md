<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# NereidForge Trainer SDK

The SDK implements the normative integration contract between a training
component and the platform. In-tree vision trainers are its first consumers;
customer and partner trainers use the same contract in their own repositories
and images.

```bash
pip install ./app/builders/sdk
nf-trainer init my-trainer --task detection
```

## Trainer contract

### Capability profile

Each training image ships a `trainer-profile.json` using schema
`nf.trainer-profile/1`, normally at `/app/trainer-profile.json`. The read-only
profile entrypoint prints the same document:

```bash
docker run --rm <image> python /app/nf_profile.py
# Images with the shared SCRIPT dispatcher also accept SCRIPT=profile.
```

A profile declares the component name, framework, supported devices, tasks and
architectures, entrypoints, environment inputs, artifacts, configuration schema,
and code/weights license declarations. `tasks[].default_arch` must name an
architecture implemented by that task. Device lists must describe what the
specific image supports; different accelerator vendors use separate images.
Validate a profile before registering it:

```bash
nf-trainer validate trainer-profile.json
```

### Runtime and artifacts

The platform starts the profile's training entrypoint with these inputs:

| Variable | Meaning | SDK entry point |
|---|---|---|
| `INPUT_URI` | Input dataset prefix, normally an `s3://` URI | `download_dataset()` |
| `OUTPUT_DIR` | Job-owned artifact destination prefix | `publish_file()` |
| `CONFIG_JSON` | Complete inline JSON configuration or a JSON file path | `nereidforge_trainer.read_config()` |
| `ARTIFACT_ID` | Stable artifact basename, normally the job ID | `build_manifest()` |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` | Object-store access when remote artifacts are used | S3 artifact helpers |

A successful run exits zero only after publishing its declared model artifacts
and final metrics. The canonical files are `$OUTPUT_DIR/$ARTIFACT_ID.metrics.json`
and, when declared by the profile, `$ARTIFACT_ID.metadata.json`. Live
`progress.json` is overwritten during execution through `Progress`; it does not
replace final metrics. Failed runs exit non-zero and may publish diagnostics.
Canonical input layouts are listed in the
[builders README](../README.md#canonical-dataset-layouts).

### Final metrics

Every new or republished trainer emits `nf.training-metrics/1` on success:

```json
{
  "schema": "nf.training-metrics/1",
  "task": "detection",
  "architecture": "example-detector",
  "train_time_s": 3465.0,
  "mAP50": 0.84553
}
```

`schema` and `train_time_s` are required. Training duration is a finite,
positive wall-clock measurement around the framework's training work, including
validation/checkpoint work owned by that call and excluding queueing, dataset
download, export, upload, and platform registration. Distributed jobs use the
earliest training start through the latest finish, not the sum of rank times.
Other task-specific metric keys and artifact URIs remain open.

Use `build_training_metrics()` to construct the document and validate it before
building a delivery image:

```bash
nf-trainer validate-metrics /path/to/<id>.metrics.json
```

Augmentation presets are resolved through `resolve_augmentation()`. Nothing
outside this contract is guaranteed: the platform never imports trainer code,
and trainer code never imports the platform.

## License boundary

The SDK is Apache-2.0. Customer code written against it remains customer code
in a customer-owned image. A customer-selected framework and weights retain
their own licenses and obligations; user-supplied L2 components must not be
copied into this repository or its default L1 images. See
[LICENSE-POLICY.md](../../../LICENSE-POLICY.md).

## Versioning

`nereidforge_trainer.__version__` uses the rolling `0.0.0.dev0` development
default, the PEP 440 spelling of the `0.0.0-dev` image channel. Tagged image
versions derive from the signed git tag; the repository has no VERSION file.
Profile and final-metrics documents carry independent schema versions and
change only when their document shapes change.
