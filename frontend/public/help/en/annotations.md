<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Annotation Management

## Flow
1. Create an annotation project: select a dataset and an annotation task type (detection, segmentation, classification, and so on).
2. Open the annotation workbench: Label Studio is embedded for drawing bounding boxes, polygons, and other shapes image by image.
3. Semi-automatic annotation: click Semi-automatic annotation and optionally select an ML Backend model to pre-annotate unlabeled tasks.
4. Review and export: finished annotations can be exported as COCO, YOLO, Label Studio JSON, or CSV.

## Prerequisites
- A dataset with uploaded images is required.
- Semi-automatic annotation needs a healthy ML Backend (registered on the ML Backend page, or the default service).

## FAQ
- **Why did pre-annotation return nothing?** Confirm the selected model's class set matches the annotation configuration. A weak model producing low-confidence predictions is expected.
- **Where does the export go?** Export creates a new dataset (YOLO format) that can be selected directly on the Training page.

## Next Steps
After export, open the Training page and create a training job; the data loop closes automatically.
