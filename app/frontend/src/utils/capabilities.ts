// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

/**
 * Helpers over component-announced capabilities (`nf.trainer-profile/1`).
 *
 * The platform never hardcodes what a trainer can do: a registered image
 * announces its tasks, architectures and devices, and `/capabilities`
 * returns them with per-node lock states against this deployment's
 * entitlements (ADR-015). These helpers keep the UI mapping in one place.
 */
import type {
  AnnouncedArchitecture,
  AnnouncedTask,
  CapabilityImage,
} from '@api/endpoints';

/** UI task vocabulary -> announced task ids that satisfy it. */
const UI_TASK_IDS: Record<string, string[]> = {
  detection: ['detection'],
  segmentation: ['instance_seg', 'semantic_seg'],
  classification: ['classification', 'multilabel'],
  pose: ['pose'],
  llm: ['llm'],
  vlm: ['vlm'],
};

export const announcedTaskIds = (uiTaskType: string): string[] =>
  UI_TASK_IDS[uiTaskType] ?? [uiTaskType];

/** Accelerator selector token -> device token used in announcements. */
export const acceleratorDevice = (accelerator: string): string => {
  if (accelerator === 'nvidia') return 'cuda';
  if (accelerator === 'amd') return 'rocm';
  if (accelerator === 'intel') return 'xpu';
  return 'cpu';
};

/** Repo-name fallback for catalog rows without an announcement. */
const repoFamily = (ref: string): { family: string; device: string } | null => {
  const base = ref.replace(/:[^:/]+$/, '').split('/').pop() ?? '';
  const match = base.match(/^(?:trainer-)?(torch|paddle|llm)-(cpu|cu|rocm|xpu)$/);
  if (!match) return null;
  return { family: match[1], device: match[2] === 'cu' ? 'cuda' : match[2] };
};

export const imageSupportsTask = (image: CapabilityImage, uiTaskType: string): boolean => {
  const wanted = new Set(announcedTaskIds(uiTaskType));
  if (image.announced) {
    return image.tasks.some((task) => wanted.has(task.id));
  }
  const ids = wanted;
  // Unannounced rows still carry the legacy task_types array.
  const legacy = image.task_types;
  return Array.isArray(legacy) && legacy.some((id) => ids.has(id));
};

export const imageSupportsDevice = (image: CapabilityImage, accelerator: string): boolean => {
  const device = acceleratorDevice(accelerator);
  if (image.announced && image.devices.length > 0) {
    return image.devices.includes(device);
  }
  const family = repoFamily(image.full_ref);
  return !family || family.device === device;
};

/** Images that can run this UI task on this vendor, announced-first. */
export const matchingImages = (
  images: CapabilityImage[],
  uiTaskType: string,
  accelerator: string,
): CapabilityImage[] =>
  images.filter(
    (image) => imageSupportsTask(image, uiTaskType) && imageSupportsDevice(image, accelerator),
  );

/** Find a specific image entry by its resolved full ref (tag-insensitive). */
export const findImage = (images: CapabilityImage[], ref: string): CapabilityImage | undefined => {
  const wanted = (ref || '').trim();
  if (!wanted) return undefined;
  const noTag = wanted.replace(/:[^:/]+$/, '');
  return images.find(
    (image) => image.full_ref === wanted || image.full_ref.replace(/:[^:/]+$/, '') === noTag,
  );
};

const taskOf = (image: CapabilityImage, uiTaskType: string): AnnouncedTask | undefined => {
  const wanted = announcedTaskIds(uiTaskType);
  return image.tasks.find((task) => wanted.includes(task.id));
};

/**
 * Architecture options announced for the selected image (or, when no image is
 * selected yet, the union over every matching image). Locked architectures
 * stay visible but disabled — the upgrade hint lists the missing entitlement.
 */
export const announcedArchOptions = (
  images: CapabilityImage[],
  uiTaskType: string,
  selectedImageRef: string,
): Array<AnnouncedArchitecture & { image: string }> => {
  const sources = selectedImageRef
    ? [findImage(images, selectedImageRef)].filter(Boolean) as CapabilityImage[]
    : matchingImages(images, uiTaskType, 'cpu').length
      ? matchingImages(images, uiTaskType, 'cpu')
      : images;
  const seen = new Map<string, AnnouncedArchitecture & { image: string }>();
  for (const image of sources) {
    const task = taskOf(image, uiTaskType);
    for (const arch of task?.architectures ?? []) {
      if (!seen.has(arch.id)) {
        seen.set(arch.id, { ...arch, image: image.full_ref });
      }
    }
  }
  return [...seen.values()];
};

/** Default architecture for a task as announced by the selected image. */
export const announcedDefaultArch = (
  images: CapabilityImage[],
  uiTaskType: string,
  selectedImageRef: string,
): string => {
  const image = findImage(images, selectedImageRef);
  if (!image) return '';
  return taskOf(image, uiTaskType)?.default_arch ?? '';
};

/** Lock state for the task the job asks of the selected image. */
export const taskLock = (
  image: CapabilityImage | undefined,
  uiTaskType: string,
): { locked: boolean; missing: string[] } => {
  const task = image && taskOf(image, uiTaskType);
  if (!task) return { locked: false, missing: [] };
  return { locked: Boolean(task.locked), missing: task.missing ?? [] };
};

/** First announced capability matching an id (e.g. optimization.tensorrt). */
export const capabilityLock = (
  images: CapabilityImage[],
  capabilityId: string,
): { locked: boolean; missing: string[] } => {
  for (const image of images) {
    const cap = image.capabilities.find((c) => c.id === capabilityId);
    if (cap && cap.locked) return { locked: true, missing: cap.missing ?? [] };
  }
  return { locked: false, missing: [] };
};
