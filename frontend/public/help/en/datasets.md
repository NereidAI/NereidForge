<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Dataset Management

## Flow
1. Create a dataset: enter a name and description, and choose a format (COCO / YOLO).
2. Upload data: individual images or a ZIP archive are accepted. Archives are extracted and imported asynchronously; progress is visible on the Dataset Versions page.
3. Dataset versions: every upload creates a new version. Historical versions are retained and can be switched between.

## Prerequisites
- Supported image formats: jpg / jpeg / png.
- Images inside a ZIP may sit at the archive root or in an images/ subdirectory.

## FAQ
- **How do I prepare data for annotation and training?** Finish the annotations in the annotation workbench, then use Export to generate a YOLO-format dataset. Training jobs pick it up automatically.
- **How are version numbers incremented?** A new version is created automatically on every image upload.

## Next Steps
Annotation then training: open the Annotations page and create an annotation project from this dataset, or go straight to the Training page and start a job with the dataset selected.
