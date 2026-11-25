<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Model Registry

## Flow
1. Registration: training jobs register automatically; manual upload accepts single files, related file groups, and .zip/.tar.gz/.tgz archives.
2. Version management: every model supports multiple versions; optimized versions record their target device and speedup.
3. Compare and select: the version table shows target device and speedup so the best version can be chosen.
4. Deployment: select a version, configure the runtime (OpenVINO/Triton), GPU, and domain; the KServe service and Ingress are created automatically.

## Prerequisites
- Manual upload accepts only model files listed in the artifact catalog; an OpenVINO IR must include its same-named .xml/.bin pair, and ambiguous archives require an explicit primary artifact path.
- The deployment domain must resolve in DNS (the default generated name is `<model>-<version>.inference.<platform-domain>`).

## Artifact Bundles and Capabilities
- Multi-file uploads preserve every file; archives pass path, symlink, duplicate, file-count, and extracted-size validation before registration, without executing model code.
- Expanded version lists support nested paths so individual files can be downloaded; trials also use the exact relative paths from that list.
- The capability catalog distinguishes registration, download, trial, optimization, and deployment. TensorRT engines and Ascend OM currently support registration and download only; successful storage does not imply deployability.
- Compiled artifacts can carry a runtime-requirement JSON (toolchain, driver, compute capability, and similar); these fields record constraints only and are not a compatibility proof.

## Image Trials
- In a specific version's artifact list, choose **Test with image** and either pick a dataset image or upload a local one.
- **Auto** uses only verified built-in decoders; a built-in decoder can also be selected manually. When the output cannot be recognized, choose **Custom mapping** and configure input preprocessing, output tensors, and field semantics from the model's documentation.
- Detection draws boxes, instance segmentation draws instance contours, semantic segmentation overlays class masks, and classification shows Top-K. When the output is unrecognized and no mapping is configured, the platform never guesses coordinates or returns empty detections.
- The decoder chosen after a successful trial is saved per artifact and reused automatically next time; other artifacts of the same version are unaffected.
- At upload time, Model Source is only a model-family hint and Input Shape is only a preprocessing/export-size hint; neither proves the output format.

## Deleting Registry Entries
- Only the enabled local built-in `admin` in the `default` tenant with the current `admin` role can delete a model or one version; tenant admins, operators, and federated namesake accounts cannot.
- Deleting a model removes all of its registry versions. Hub records are archived and NF records are removed, while S3 artifact bytes are retained.
- Deployed, deploying, terminating, or failed-undeploy versions whose service still exists must be undeployed successfully first; a 409 response means no partial deletion occurred.

## FAQ
- **How do I access a deployment?** The deployment response returns the domain; the Ingress is created automatically and DNS points to the cluster IP.
- **Which runtime should I choose?** OpenVINO (CPU, verified) or Triton (reserved for GPU scenarios).
- **Why does it say "decoder mapping required"?** No built-in decoder reliably matches the current graph. Select a verified decoder manually, or configure a mapping from the actual output tensors.

## Next Steps
After deployment the inference endpoint is ready to use; alternatively, continue optimizing or evaluating other versions.
