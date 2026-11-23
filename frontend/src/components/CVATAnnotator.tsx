// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Button, Space, Spin, Empty, Select, Slider, Tooltip, Tag, List, Badge, message,
} from 'antd';
import {
  CaretRightOutlined, PauseOutlined, StepBackwardOutlined, StepForwardOutlined,
  SaveOutlined, DeleteOutlined, PlusSquareOutlined, EyeInvisibleOutlined,
  AimOutlined, CheckOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { annotationApi, samApi, modelApi } from '@api/endpoints';
import type { CVATAnnotationDetail, CVATEditorTrack, CVATTrackKf, CVATTrackInput, Model, ModelVersion } from '@nftypes/api';

// Frames are served one-by-one by CVAT (a video chunk is ~36 frames and is
// decoded server-side on demand). We must stay more than one chunk ahead of
// the playhead, otherwise crossing a cold chunk boundary stalls playback.
// 48 buffered frames + aggressive background prefetch hide that latency.
const PLAY_BUFFER = 48;
// Max number of frame requests in flight at once during background prefetch.
// CVAT's nginx rate-limits aggressive fetching (HTTP 429); keep this modest and
// rely on the backend's Retry-After backoff for the residual bursts.
const PREFETCH_CONCURRENCY = 4;
// Max decoded frames retained (each ~25KB PNG; 240 is a few MB).
const CACHE_MAX = 240;
// Frames nearest the playhead are always retained during eviction.
const CACHE_KEEP = 180;

const LABEL_COLORS = [
  '#e60049', '#0bb4ff', '#50e991', '#e6d800', '#9b19f5',
  '#ffa300', '#dc0ab4', '#b3d4ff', '#00b464', '#ff6bac',
];

interface LocalTrack {
  id: number;
  label_id: number;
  keyframes: CVATTrackKf[];
}

interface DragState {
  kind: 'new' | 'move' | 'resize';
  trackId: number | null;
  start: [number, number]; // natural coords at pointer-down
  cur: [number, number];
  origPoints: number[]; // box being moved/resized at pointer-down
}

/** Interpolate a track's rectangle at frame f (CVAT semantics: visible from a
 *  non-outside keyframe until an outside keyframe; linear between keyframes). */
export function trackBoxAt(track: LocalTrack, f: number): number[] | null {
  const kfs = [...track.keyframes].sort((a, b) => a.frame - b.frame);
  let prev: CVATTrackKf | null = null;
  let next: CVATTrackKf | null = null;
  for (const k of kfs) {
    if (k.frame <= f) prev = k;
    else { next = k; break; }
  }
  if (!prev || prev.outside) return null;
  if (next && !next.outside) {
    const span = next.frame - prev.frame || 1;
    const t = (f - prev.frame) / span;
    return [0, 1, 2, 3].map((i) => prev.points[i] + (next.points[i] - prev.points[i]) * t);
  }
  return [...prev.points];
}

interface Props {
  projectId: string;
  taskId: number;
  onSaved?: () => void;
  selectedMLBackendId?: string;
  // Increment to force a reload of the task's annotations/frames from CVAT
  // (the parent's header refresh button bumps this; props alone don't refetch).
  reloadNonce?: number;
}

/** NF-native video annotation editor for a CVAT task.

 * Frames are proxied through the NF backend (authenticated XHR -> blob).
 * Boxes are drawn/edited on a canvas overlay; every edit creates or updates
 * a keyframe at the current frame and CVAT interpolates between keyframes.
 * Saving replaces all annotations of the task with the editor's tracks.
 */
export const CVATAnnotator: React.FC<Props> = ({ projectId, taskId, onSaved, selectedMLBackendId, reloadNonce = 0 }) => {
  const { t } = useTranslation();

  const [detail, setDetail] = useState<CVATAnnotationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [tracks, setTracks] = useState<LocalTrack[]>([]);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);
  const [drawLabelId, setDrawLabelId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [samMode, setSamMode] = useState(false);
  const [samPreview, setSamPreview] = useState<number[][] | null>(null);
  const [samBusy, setSamBusy] = useState(false);
  const [samPoints, setSamPoints] = useState<number[][]>([]);
  // User-selected SAM model (interactive-segmentation weights from S3). The
  // image ships no SAM model, so a selection is required before click-to-segment.
  const [samModels, setSamModels] = useState<Model[]>([]);
  const [samModelName, setSamModelName] = useState<string | undefined>(undefined);
  const [samVersions, setSamVersions] = useState<ModelVersion[]>([]);
  const [samVersion, setSamVersion] = useState<string | undefined>(undefined);
  const [frameUrl, setFrameUrl] = useState<string>('');
  const [frameLoading, setFrameLoading] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const blobCacheRef = useRef<Map<number, string>>(new Map());
  // Coalesces concurrent requests for the same frame (current-frame load +
  // background prefetch + play loop) so each frame is fetched at most once.
  const inflightRef = useRef<Map<number, Promise<string | null>>>(new Map());
  const nextTrackIdRef = useRef(-1);
  const [dragTick, setDragTick] = useState(0); // forces canvas redraw during drag

  const frameCount = Math.max(1, detail?.task.frame_count || detail?.task.size || 1);
  const labels = useMemo(() => detail?.labels || [], [detail?.labels]);
  const labelName = useCallback((lid: number) => labels.find((l) => l.id === lid)?.name || `#${lid}`, [labels]);
  const labelColor = useCallback((lid: number) => LABEL_COLORS[lid % LABEL_COLORS.length], []);

  // -- data loading ---------------------------------------------------------

  const loadDetail = useCallback(async () => {
    setLoading(true);
    try {
      const res = await annotationApi.getCVATTaskAnnotation(projectId, taskId);
      setDetail(res.data);
      setTracks(res.data.tracks.map((tr: CVATEditorTrack) => ({
        id: tr.id,
        label_id: tr.label_id,
        keyframes: tr.keyframes.map((k) => ({ ...k, points: [...k.points] })),
      })));
      setFrame(0);
      setSelectedTrack(null);
      if (!drawLabelId && res.data.labels.length) setDrawLabelId(res.data.labels[0].id);
    } catch (e) {
      console.error('load annotation failed:', e);
      message.error(String((e as any)?.response?.data?.detail || e));
    } finally {
      setLoading(false);
    }
  }, [projectId, taskId, drawLabelId]);

  useEffect(() => {
    // release cached frame blobs when switching tasks
    blobCacheRef.current.forEach((url) => URL.revokeObjectURL(url));
    blobCacheRef.current = new Map();
    setTracks([]);
    setDetail(null);
    loadDetail();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, taskId]);

  // Explicit refresh from the parent (header refresh button): refetch the
  // task annotations without changing the task identity / clearing frames.
  useEffect(() => {
    if (reloadNonce > 0) loadDetail();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadNonce]);

  useEffect(() => () => {
    blobCacheRef.current.forEach((url) => URL.revokeObjectURL(url));
    blobCacheRef.current = new Map();
  }, []);

  // Load SAM-capable models (task_type=sam) for the interactive segmenter.
  useEffect(() => {
    modelApi.list({ page_size: 100, task_type: 'sam' })
      .then((r) => setSamModels(r.data.items || []))
      .catch(() => setSamModels([]));
  }, []);

  // Default the SAM model picker to the first available SAM model.
  useEffect(() => {
    if (!samModelName && samModels.length > 0) setSamModelName(samModels[0].name);
  }, [samModels, samModelName]);

  // Load versions of the chosen SAM model; default to its first ready version.
  useEffect(() => {
    if (!samModelName) { setSamVersions([]); setSamVersion(undefined); return; }
    const m = samModels.find((mm) => mm.name === samModelName);
    if (!m) { setSamVersions([]); setSamVersion(undefined); return; }
    let cancelled = false;
    modelApi.getVersions(m.id, { page_size: 100 })
      .then((r) => {
        if (cancelled) return;
        const vs = r.data.items || [];
        setSamVersions(vs);
        setSamVersion((cur) => cur || vs[0]?.version);
      })
      .catch(() => { if (!cancelled) { setSamVersions([]); setSamVersion(undefined); } });
    return () => { cancelled = true; };
  }, [samModelName, samModels]);

  const fetchFrameBlob = useCallback(async (n: number, signal?: AbortSignal): Promise<string | null> => {
    const cache = blobCacheRef.current;
    const hit = cache.get(n);
    if (hit) return hit;
    // Coalesce an identical in-flight request instead of issuing a duplicate.
    const flying = inflightRef.current.get(n);
    if (flying) return flying;
    const task = (async (): Promise<string | null> => {
      try {
        const res = await annotationApi.getCVATFrame(projectId, taskId, n, 'compressed', signal);
        const bytes = new Uint8Array(res.data);
        // CVAT video frames are PNG even at quality=compressed; image sets are
        // JPEG. Sniff the magic bytes so the browser decodes the right type.
        const isPng = bytes.length >= 8
          && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
        const url = URL.createObjectURL(new Blob([res.data], { type: isPng ? 'image/png' : 'image/jpeg' }));
        cache.set(n, url);
        // Warm the browser's image cache NOW so assigning this blob URL to the
        // visible <img> during playback decodes instantly (otherwise the image
        // paints one frame after the overlaid box, making boxes "jump").
        const pre = new Image();
        pre.src = url;
        // keep the cache bounded: drop frames distant from the playhead
        if (cache.size > CACHE_MAX) {
          const keys = [...cache.keys()].sort((a, b) => Math.abs(a - n) - Math.abs(b - n));
          for (const k of keys.slice(CACHE_KEEP)) {
            const u = cache.get(k);
            if (u) URL.revokeObjectURL(u);
            cache.delete(k);
          }
        }
        return url;
      } catch (e: unknown) {
        // aborted requests are expected when scrubbing fast; not an error
        if ((e as { name?: string })?.name === 'CanceledError' || (e as { code?: string })?.code === 'ERR_CANCELED') {
          return null;
        }
        console.error('frame fetch failed:', n, e);
        return null;
      } finally {
        inflightRef.current.delete(n);
      }
    })();
    inflightRef.current.set(n, task);
    return task;
  }, [projectId, taskId]);

  // load current frame (aborting any stale request so scrubbing stays
  // responsive) + prefetch neighbours once the position settles
  const frameAbortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!detail) return;
    let cancelled = false;
    frameAbortRef.current?.abort();
    const ac = new AbortController();
    frameAbortRef.current = ac;
    setFrameLoading(true);
    // A cold video chunk can briefly answer 429 while CVAT encodes it. Retry
    // for a few seconds (the loading overlay stays up) instead of failing the
    // whole detail view; rapid scrubbing still aborts via the AbortController.
    const loadWithRetry = async () => {
      for (let i = 0; i < 10 && !cancelled && !ac.signal.aborted; i += 1) {

        const url = await fetchFrameBlob(frame, i === 0 ? ac.signal : undefined);
        if (url) {
          if (!cancelled && !ac.signal.aborted) setFrameUrl(url);
          if (!cancelled) setFrameLoading(false);
          return;
        }

        await new Promise((r) => setTimeout(r, 300));
      }
      if (!cancelled) setFrameLoading(false);
    };
    loadWithRetry();
    // Prefetch only after the user stops scrubbing (250ms debounce); firing
    // prefetches on every intermediate frame floods the connection pool and
    // starves the frame the user is actually looking at.
    const timer = setTimeout(() => {
      for (const d of [1, -1, 2, 3]) {
        const n = frame + d;
        if (n >= 0 && n < frameCount) fetchFrameBlob(n);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [detail, frame, frameCount, fetchFrameBlob]);

  // playback loop: fully preload-driven. On play we first warm a buffer
  // (next N frames), then advance ONLY when the next frame blob is cached —
  // the <img> swap is instant and playback never outruns loading.
  const frameRef = useRef(frame);
  useEffect(() => { frameRef.current = frame; }, [frame]);
  const [playBuffering, setPlayBuffering] = useState(false);

  // Fetch a frame with short retries. A single transient failure (CVAT
  // decoding a cold chunk, a proxy hiccup) must not abort playback forever;
  // only a genuinely unreachable frame gives up after a few attempts.
  const fetchFrameRetry = useCallback(
    async (n: number, tries = 3): Promise<string | null> => {
      for (let i = 0; i < tries; i += 1) {
        const url = await fetchFrameBlob(n);
        if (url) return url;
        // the failed attempt may have cached nothing; wait briefly before retry

        await new Promise((r) => setTimeout(r, 150 * (i + 1)));
      }
      return null;
    },
    [fetchFrameBlob],
  );

  // Background prefetch: keep the next PLAY_BUFFER frames warm, fetched with a
  // bounded concurrency and fully decoupled from the playback timer. This is
  // what hides cold-chunk decode latency in CVAT's per-frame server endpoint.
  // fetchFrameBlob coalesces duplicate requests, so the play loop and the
  // current-frame loader never trigger a second request for the same frame.
  useEffect(() => {
    if (!detail) return undefined;
    let cancelled = false;
    const queue: number[] = [];
    let cursor = 0;

    const buildQueue = () => {
      queue.length = 0;
      const start = frameRef.current;
      const end = Math.min(frameCount - 1, start + PLAY_BUFFER);
      const cache = blobCacheRef.current;
      for (let n = start + 1; n <= end; n += 1) {
        if (!cache.has(n) && !inflightRef.current.has(n)) queue.push(n);
      }
      cursor = 0;
    };

    const worker = async () => {
      while (!cancelled) {
        if (cursor >= queue.length) {
          // queue drained: rebuild against the new playhead after a short idle
          buildQueue();
          if (cursor >= queue.length) {

            await new Promise((r) => setTimeout(r, 120));
            continue;
          }
        }
        const n = queue[cursor];
        cursor += 1;

        await fetchFrameBlob(n);
      }
    };

    buildQueue();
    const workers = Array.from({ length: PREFETCH_CONCURRENCY }, () => worker());
    return () => {
      cancelled = true;
      Promise.allSettled(workers);
    };
  }, [detail, frameCount, playing, fetchFrameBlob]);

  useEffect(() => {
    if (!playing || !detail) return;
    let stopped = false;
    (async () => {
      // initial buffer: warm the CURRENT frame and the next PLAY_BUFFER frames
      // concurrently (bounded) before the first advance, so starting playback
      // never stalls on a cold chunk.
      const start = frameRef.current;
      const end = Math.min(frameCount - 1, start + PLAY_BUFFER);
      setPlayBuffering(true);
      const wanted: number[] = [];
      for (let n = start; n <= end; n += 1) wanted.push(n);
      for (let i = 0; i < wanted.length; i += PREFETCH_CONCURRENCY) {
        if (stopped) return;
        const batch = wanted.slice(i, i + PREFETCH_CONCURRENCY);

        const urls = await Promise.all(batch.map((n) => fetchFrameRetry(n)));
        if (urls.some((u) => !u)) { setPlaying(false); return; }
      }
      if (stopped) return;
      setPlayBuffering(false);

      while (!stopped) {
        // fixed playback cadence; the <img> swap is instant because the
        // background prefetch keeps upcoming frames resident.
        await new Promise((r) => setTimeout(r, 120));
        if (stopped) break;
        const next = frameRef.current + 1;
        if (next >= frameCount) {
          setPlaying(false);
          break;
        }
        // Rarely the buffer may not have reached this frame yet (cold chunk);
        // wait for it (with retries) instead of advancing onto a blank frame.
        const url = await fetchFrameRetry(next);
        if (stopped) break;
        if (!url) {
          // frame truly unreachable after retries: stop playback
          setPlaying(false);
          break;
        }
        // Atomic advance: set the decoded blob URL and the frame index in the
        // same React batch so the image and its overlaid boxes always describe
        // the SAME frame (setting frame alone lets the canvas draw the new
        // frame's box on the previous frame's image for one tick = "jumping").
        // React 18 auto-batches these two setState calls into one commit.
        setFrameUrl(url);
        setFrame(next);
      }
    })();
    return () => { stopped = true; setPlayBuffering(false); };
  }, [playing, detail, frameCount, fetchFrameRetry]);

  // keyboard frame stepping
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') setFrame((f) => Math.max(0, f - 1));
      if (e.key === 'ArrowRight') setFrame((f) => Math.min(frameCount - 1, f + 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [frameCount]);

  // -- track editing helpers --------------------------------------------------

  const setKeyframe = useCallback((trackId: number, f: number, points: number[], outside = false) => {
    setTracks((prev) => prev.map((tr) => {
      if (tr.id !== trackId) return tr;
      const kfs = tr.keyframes.filter((k) => k.frame !== f);
      kfs.push({ frame: f, points: [...points], outside, occluded: false });
      kfs.sort((a, b) => a.frame - b.frame);
      return { ...tr, keyframes: kfs };
    }));
  }, []);

  const deleteKeyframe = useCallback((trackId: number, f: number) => {
    setTracks((prev) => prev
      .map((tr) => (tr.id === trackId
        ? { ...tr, keyframes: tr.keyframes.filter((k) => k.frame !== f) }
        : tr))
      .filter((tr) => tr.keyframes.length > 0));
    setSelectedTrack((s) => (s === trackId ? null : s));
  }, []);

  const toggleOutside = useCallback((trackId: number, f: number) => {
    setTracks((prev) => prev.map((tr) => {
      if (tr.id !== trackId) return tr;
      return {
        ...tr,
        keyframes: tr.keyframes.map((k) => (k.frame === f ? { ...k, outside: !k.outside } : k)),
      };
    }));
  }, []);

  const deleteTrack = useCallback((trackId: number) => {
    setTracks((prev) => prev.filter((tr) => tr.id !== trackId));
    setSelectedTrack((s) => (s === trackId ? null : s));
  }, []);

  const addKeyframeFromCurrent = useCallback(() => {
    if (selectedTrack == null) return;
    const tr = tracks.find((x) => x.id === selectedTrack);
    if (!tr) return;
    const box = trackBoxAt(tr, frame);
    if (!box) {
      message.warning(t('pages.annotations.studio.cvatNotVisible'));
      return;
    }
    setKeyframe(tr.id, frame, box);
  }, [selectedTrack, tracks, frame, setKeyframe, t]);

  // -- canvas rendering -------------------------------------------------------

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const w = img.naturalWidth || 640;
    const h = img.naturalHeight || 480;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    const drag = dragRef.current;
    const drawBox = (pts: number[], color: string, selected: boolean, dashed = false) => {
      const [x1, y1, x2, y2] = pts;
      ctx.strokeStyle = color;
      ctx.lineWidth = selected ? 3 : 2;
      if (dashed) ctx.setLineDash([6, 4]);
      else ctx.setLineDash([]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (selected) {
        ctx.fillStyle = color;
        const hs = 8;
        for (const [cx, cy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
          ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
        }
      }
    };

    for (const tr of tracks) {
      const box = trackBoxAt(tr, frame);
      if (!box) continue;
      const selected = tr.id === selectedTrack;
      const kfHere = tr.keyframes.find((k) => k.frame === frame);
      if (kfHere?.shape === 'polygon' && kfHere.poly && kfHere.poly.length >= 6) {
        // draw the polygon outline (+ bbox frame when selected)
        ctx.strokeStyle = labelColor(tr.label_id);
        ctx.lineWidth = selected ? 3 : 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(kfHere.poly[0], kfHere.poly[1]);
        for (let i = 2; i < kfHere.poly.length; i += 2) ctx.lineTo(kfHere.poly[i], kfHere.poly[i + 1]);
        ctx.closePath();
        ctx.stroke();
        ctx.fillStyle = labelColor(tr.label_id) + '22';
        ctx.fill();
        if (selected) drawBox(box, labelColor(tr.label_id), true);
      } else {
        drawBox(box, labelColor(tr.label_id), selected);
      }
      // label caption
      const tag = labelName(tr.label_id) + (tr.keyframes.some((k) => k.frame === frame) ? ' ◆' : '');
      ctx.font = '14px sans-serif';
      ctx.fillStyle = labelColor(tr.label_id);
      ctx.fillText(tag, box[0], Math.max(12, box[1] - 4));
    }

    // SAM preview polygon + prompt points
    if (samMode) {
      if (samPreview && samPreview.length >= 3) {
        ctx.strokeStyle = '#faad14';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(samPreview[0][0], samPreview[0][1]);
        for (const [px, py] of samPreview.slice(1)) ctx.lineTo(px, py);
        ctx.closePath();
        ctx.stroke();
        ctx.fillStyle = 'rgba(250, 173, 20, 0.12)';
        ctx.fill();
      }
      ctx.setLineDash([]);
      for (const [px, py] of samPoints) {
        ctx.fillStyle = '#52c41a';
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    // new-box preview
    if (drag?.kind === 'new') {
      const x1 = Math.min(drag.start[0], drag.cur[0]);
      const y1 = Math.min(drag.start[1], drag.cur[1]);
      const x2 = Math.max(drag.start[0], drag.cur[0]);
      const y2 = Math.max(drag.start[1], drag.cur[1]);
      drawBox([x1, y1, x2, y2], '#faad14', false, true);
    }
    // move/resize live preview (dashed) replaces the tracked box
    if (drag?.kind === 'move' || drag?.kind === 'resize') {
      const dx = drag.cur[0] - drag.start[0];
      const dy = drag.cur[1] - drag.start[1];
      let pts: number[];
      if (drag.kind === 'move') {
        pts = [
          drag.origPoints[0] + dx, drag.origPoints[1] + dy,
          drag.origPoints[2] + dx, drag.origPoints[3] + dy,
        ];
      } else {
        // resize: anchor corner is the opposite one (already baked into origPoints)
        const o = drag.origPoints;
        pts = [
          Math.min(o[0], drag.cur[0]), Math.min(o[1], drag.cur[1]),
          Math.max(o[0], drag.cur[0]), Math.max(o[1], drag.cur[1]),
        ];
      }
      drawBox(pts, '#faad14', false, true);
    }
  }, [tracks, frame, selectedTrack, labelColor, labelName, samMode, samPreview, samPoints]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas, frameUrl, dragTick]);

  // -- pointer interaction ------------------------------------------------------

  const toNatural = useCallback((e: React.PointerEvent): [number, number] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
  }, []);

  const hitTest = useCallback((x: number, y: number): { trackId: number; corner: boolean } | null => {
    const tol = 12; // display-px-ish tolerance in natural coords (scaled later)
    for (let i = tracks.length - 1; i >= 0; i--) {
      const box = trackBoxAt(tracks[i], frame);
      if (!box) continue;
      const [x1, y1, x2, y2] = box;
      const inside = x >= x1 - 2 && x <= x2 + 2 && y >= y1 - 2 && y <= y2 + 2;
      if (!inside) continue;
      const nearCorner = (
        Math.abs(x - x1) < tol || Math.abs(x - x2) < tol
      ) && (Math.abs(y - y1) < tol || Math.abs(y - y2) < tol);
      return { trackId: tracks[i].id, corner: nearCorner };
    }
    return null;
  }, [tracks, frame]);


  // SAM click-to-segment: collect points; on each click refine, on shift-click
  // add negative point, double-click / button commits the polygon as a track.
  const runSam = useCallback(async (pts: number[][], negs: number[][]) => {
    if (!projectId || samBusy) return;
    if (!samModelName || !samVersion) {
      message.warning(t('pages.annotations.studio.samSelectModel'));
      return;
    }
    setSamBusy(true);
    try {
      // frame already rendered -> reuse its blob as base64 (no URL fetch)
      const cur = blobCacheRef.current.get(frame);
      let b64 = '';
      if (cur) {
        const blob = await fetch(cur).then((r) => r.blob());
        b64 = await new Promise<string>((resolve) => {
          const fr = new FileReader();
          fr.onload = () => resolve(String(fr.result).split(',')[1] || '');
          fr.readAsDataURL(blob);
        });
      }
      const res = await samApi.segment({
        image_base64: b64 || undefined,
        image_url: b64 ? undefined : `/api/v1/projects/${projectId}/cvat-tasks/${taskId}/frames/${frame}`,
        points: pts,
        negative_points: negs,
        ml_backend_id: selectedMLBackendId || undefined,
        model_name: samModelName,
        model_version: samVersion,
      });
      const poly = res.data.polygon;
      if (poly && poly.length >= 3) {
        setSamPreview(poly);
      } else {
        setSamPreview(null);
        message.warning(res.data.message || t('pages.annotations.studio.samNoMask'));
      }
    } catch (e) {
      console.error('sam failed:', e);
      message.error(String((e as any)?.response?.data?.detail || e));
    } finally {
      setSamBusy(false);
    }
  }, [projectId, taskId, frame, samBusy, selectedMLBackendId, samModelName, samVersion, t]);

  const commitSamPolygon = useCallback(() => {
    if (!samPreview || samPreview.length < 3 || !drawLabelId) return;
    const poly = samPreview;
    const xs = poly.map((p) => p[0]);
    const ys = poly.map((p) => p[1]);
    const bbox: number[] = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    const flat: number[] = [];
    for (const [px, py] of poly) flat.push(px, py);
    const tid = nextTrackIdRef.current--;
    setTracks((prev) => [...prev, {
      id: tid,
      label_id: drawLabelId,
      keyframes: [{
        frame, points: bbox, outside: false, occluded: false,
        shape: 'polygon', poly: flat,
      }],
    }]);
    setSelectedTrack(tid);
    setSamPoints([]);
    setSamPreview(null);
    message.success(t('pages.annotations.studio.samCommitted'));
  }, [samPreview, drawLabelId, frame, t]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (playing) return;
    const [x, y] = toNatural(e);
    if (samMode) {
      // click-to-segment: collect prompt points and refine incrementally
      const neg = e.shiftKey;
      const nextPts = neg ? samPoints : [...samPoints, [x, y]];
      const nextNegs: number[][] = [];
      if (!neg) setSamPoints(nextPts);
      runSam(neg ? samPoints : nextPts, neg ? [[x, y]] : nextNegs);
      return;
    }
    const hit = hitTest(x, y);
    if (hit) {
      setSelectedTrack(hit.trackId);
      const tr = tracks.find((t2) => t2.id === hit.trackId)!;
      const box = trackBoxAt(tr, frame)!;
      dragRef.current = {
        kind: hit.corner ? 'resize' : 'move',
        trackId: hit.trackId,
        start: [x, y],
        cur: [x, y],
        origPoints: [...box],
      };
      // resize anchors on the opposite corner: adjust orig so cur defines the far corner
      if (hit.corner) {
        const o = dragRef.current.origPoints;
        const anchor = [
          Math.abs(x - o[0]) < Math.abs(x - o[2]) ? o[0] : o[2],
          Math.abs(y - o[1]) < Math.abs(y - o[3]) ? o[1] : o[3],
        ];
        dragRef.current.origPoints = [anchor[0], anchor[1], anchor[0], anchor[1]];
      }
      return;
    }
    setSelectedTrack(null);
    if (drawLabelId == null) return;
    dragRef.current = {
      kind: 'new', trackId: null, start: [x, y], cur: [x, y], origPoints: [],
    };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    drag.cur = toNatural(e);
    setDragTick((n) => n + 1);
  };

  const onPointerUp = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    const [x, y] = drag.cur;
    if (drag.kind === 'new') {
      const x1 = Math.min(drag.start[0], x);
      const y1 = Math.min(drag.start[1], y);
      const x2 = Math.max(drag.start[0], x);
      const y2 = Math.max(drag.start[1], y);
      if (x2 - x1 < 4 || y2 - y1 < 4) { setDragTick((n) => n + 1); return; }
      const id = nextTrackIdRef.current--;
      setTracks((prev) => [...prev, {
        id,
        label_id: drawLabelId!,
        keyframes: [{ frame, points: [x1, y1, x2, y2], outside: false, occluded: false }],
      }]);
      setSelectedTrack(id);
    } else if (drag.kind === 'move' && drag.trackId != null) {
      const dx = x - drag.start[0];
      const dy = y - drag.start[1];
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1) { setDragTick((n) => n + 1); return; }
      const pts = [
        drag.origPoints[0] + dx, drag.origPoints[1] + dy,
        drag.origPoints[2] + dx, drag.origPoints[3] + dy,
      ];
      setKeyframe(drag.trackId, frame, pts);
    } else if (drag.kind === 'resize' && drag.trackId != null) {
      const o = drag.origPoints;
      const pts = [
        Math.min(o[0], x), Math.min(o[1], y),
        Math.max(o[0], x), Math.max(o[1], y),
      ];
      if (pts[2] - pts[0] < 4 || pts[3] - pts[1] < 4) { setDragTick((n) => n + 1); return; }
      setKeyframe(drag.trackId, frame, pts);
    }
    setDragTick((n) => n + 1);
  };

  // -- save ----------------------------------------------------------------------

  const handleSave = async () => {
    setSaving(true);
    try {
      const payload: { tracks: CVATTrackInput[] } = {
        tracks: tracks.map((tr) => ({
          id: tr.id > 0 ? tr.id : undefined,
          label_id: tr.label_id,
          keyframes: tr.keyframes,
        })),
      };
      await annotationApi.replaceCVATTracks(projectId, taskId, payload);
      message.success(t('pages.annotations.studio.cvatSaved'));
      onSaved?.();
      await loadDetail();
    } catch (e) {
      console.error('save failed:', e);
      message.error(String((e as any)?.response?.data?.detail || e));
    } finally {
      setSaving(false);
    }
  };

  // -- render ----------------------------------------------------------------------

  if (loading && !detail) {
    return <Spin style={{ display: 'block', margin: '80px auto' }} />;
  }
  if (!detail) {
    return <Empty description={t('pages.annotations.studio.cvatNoTaskDetail')} />;
  }
  if (labels.length === 0) {
    return <Empty description={t('pages.annotations.studio.cvatNoLabels')} />;
  }

  const selected = tracks.find((tr) => tr.id === selectedTrack) || null;
  const kfAtCurrent = selected?.keyframes.find((k) => k.frame === frame) || null;

  return (
    <div data-testid="cvat-annotator">
      {/* toolbar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexWrap: 'wrap', gap: 8 }}>
        <Space wrap>
          <Select
            size="small"
            style={{ minWidth: 140 }}
            placeholder={t('pages.annotations.studio.cvatDrawLabel')}
            value={drawLabelId ?? undefined}
            onChange={(v) => setDrawLabelId(v)}
            options={labels.map((l) => ({ value: l.id, label: l.name }))}
          />
          <Tooltip title={samModels.length ? t('pages.annotations.studio.samModeHint') : t('pages.annotations.studio.samNoModelHint')}>
            <Button
              size="small"
              type={samMode ? 'primary' : 'default'}
              icon={<AimOutlined />}
              loading={samBusy}
              disabled={samModels.length === 0}
              onClick={() => {
                setSamMode((v) => !v);
                setSamPoints([]);
                setSamPreview(null);
              }}
            >
              {t('pages.annotations.studio.samMode')}
            </Button>
          </Tooltip>
          {samMode && (
            <>
              <Select
                size="small"
                style={{ minWidth: 150 }}
                placeholder={t('pages.annotations.studio.samSelectModel')}
                value={samModelName}
                onChange={(v) => { setSamModelName(v); setSamVersion(undefined); }}
                options={samModels.map((m) => ({ value: m.name, label: m.name }))}
              />
              <Select
                size="small"
                style={{ minWidth: 90 }}
                value={samVersion}
                onChange={setSamVersion}
                options={samVersions.map((v) => ({ value: v.version, label: v.version }))}
              />
              <Button size="small" icon={<CheckOutlined />} disabled={!samPreview} onClick={commitSamPolygon}>
                {t('pages.annotations.studio.samCommit')}
              </Button>
            </>
          )}
          <Tooltip title={t('pages.annotations.studio.cvatAddKfHint')}>
            <Button size="small" icon={<PlusSquareOutlined />} disabled={!selected} onClick={addKeyframeFromCurrent}>
              {t('pages.annotations.studio.cvatAddKf')}
            </Button>
          </Tooltip>
          <Tooltip title={t('pages.annotations.studio.cvatOutsideHint')}>
            <Button
              size="small"
              icon={<EyeInvisibleOutlined />}
              disabled={!kfAtCurrent}
              type={kfAtCurrent?.outside ? 'primary' : 'default'}
              onClick={() => selected && toggleOutside(selected.id, frame)}
            >
              {t('pages.annotations.studio.cvatOutside')}
            </Button>
          </Tooltip>
          <Tooltip title={t('pages.annotations.studio.cvatDelKfHint')}>
            <Button size="small" icon={<DeleteOutlined />} disabled={!kfAtCurrent} danger onClick={() => selected && deleteKeyframe(selected.id, frame)}>
              {t('pages.annotations.studio.cvatDelKf')}
            </Button>
          </Tooltip>
          <Tooltip title={t('pages.annotations.studio.cvatDelTrackHint')}>
            <Button size="small" icon={<DeleteOutlined />} disabled={!selected} danger onClick={() => selected && deleteTrack(selected.id)}>
              {t('pages.annotations.studio.cvatDelTrack')}
            </Button>
          </Tooltip>
        </Space>
        <Button size="small" type="primary" icon={<SaveOutlined />} loading={saving} onClick={handleSave}>
          {t('pages.annotations.studio.cvatSaveTracks')}
        </Button>
      </div>

      {/* frame viewer */}
      <div style={{ position: 'relative', background: '#000', minHeight: 280, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {frameLoading && !frameUrl ? (
          <Spin style={{ margin: '60px auto' }} />
        ) : frameUrl ? (
          <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
            <img
              ref={imgRef}
              src={frameUrl}
              alt={`frame ${frame}`}
              draggable={false}
              style={{
                maxWidth: '100%',
                maxHeight: '56vh',
                display: 'block',
                userSelect: 'none',
                opacity: frameLoading ? 0.55 : 1,
                transition: 'opacity 120ms',
              }}
              onLoad={drawCanvas}
            />
            <canvas
              ref={canvasRef}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={onPointerUp}
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: '100%',
                height: '100%',
                cursor: playing ? 'default' : 'crosshair',
                touchAction: 'none',
              }}
            />
          </div>
        ) : (
          <Empty description={t('pages.annotations.studio.cvatFrameLoadFail')} style={{ color: '#999' }} />
        )}
      </div>

      {/* timeline */}
      <div style={{ marginTop: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
          <Button size="small" icon={<StepBackwardOutlined />} onClick={() => setFrame((f) => Math.max(0, f - 1))} />
          <Tooltip title={playBuffering ? t('pages.annotations.studio.cvatBuffering') : ''}>
            <Button
              size="small"
              loading={playBuffering}
              icon={!playBuffering && (playing ? <PauseOutlined /> : <CaretRightOutlined />)}
              onClick={() => setPlaying((p) => !p)}
            />
          </Tooltip>
          <Button size="small" icon={<StepForwardOutlined />} onClick={() => setFrame((f) => Math.min(frameCount - 1, f + 1))} />
          <span style={{ fontSize: 12, color: '#888', minWidth: 90, whiteSpace: 'nowrap' }}>
            {frame + 1} / {frameCount}
          </span>
          <div style={{ flex: 1, minWidth: 0, position: 'relative', margin: '0 2px' }}>
            {/* key remounts rc-slider once real metadata (frameCount) is
             * loaded; rc-slider measures the rail at mount and renders a
             * broken (zero-width) track if mounted while width is unknown. */}
            <Slider
              key={`slider-${frameCount}`}
              min={0}
              max={frameCount - 1}
              value={Math.min(frame, frameCount - 1)}
              onChange={(v) => { setPlaying(false); setFrame(v as number); }}
            />
            {/* keyframe markers: all tracks gray, selected track colored */}
            <div style={{ position: 'absolute', top: 0, left: 6, right: 6, height: 4, pointerEvents: 'none' }}>
              {tracks.flatMap((tr) =>
                tr.keyframes.map((k) => (
                  <span
                    key={`${tr.id}-${k.frame}`}
                    style={{
                      position: 'absolute',
                      left: `${(k.frame / Math.max(1, frameCount - 1)) * 100}%`,
                      top: tr.id === selectedTrack ? -2 : 1,
                      width: tr.id === selectedTrack ? 6 : 4,
                      height: tr.id === selectedTrack ? 6 : 4,
                      borderRadius: 2,
                      background: tr.id === selectedTrack ? labelColor(tr.label_id) : '#bbb',
                    }}
                  />
                )),
              )}
            </div>
          </div>
        </div>
      </div>

      {/* track list */}
      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>
          {t('pages.annotations.studio.cvatTracks')} ({tracks.length})
        </div>
        <List
          size="small"
          style={{ maxHeight: 160, overflow: 'auto' }}
          dataSource={tracks}
          locale={{ emptyText: t('pages.annotations.studio.cvatNoTracks') }}
          renderItem={(tr) => (
            <List.Item
              onClick={() => { setSelectedTrack(tr.id); const first = tr.keyframes[0]?.frame ?? 0; setFrame(first); }}
              style={{
                cursor: 'pointer',
                padding: '4px 8px',
                background: tr.id === selectedTrack ? '#e6f4ff' : undefined,
              }}
            >
              <Space size={6}>
                <Badge color={labelColor(tr.label_id)} text={labelName(tr.label_id)} />
                <Tag style={{ fontSize: 11 }}>
                  {t('pages.annotations.studio.cvatKfCount')}: {tr.keyframes.length}
                </Tag>
                <Tag style={{ fontSize: 11 }}>
                  {tr.keyframes[0]?.frame}–{tr.keyframes[tr.keyframes.length - 1]?.frame}
                </Tag>
                <Button
                  size="small" type="text" danger icon={<DeleteOutlined />}
                  onClick={(e) => { e.stopPropagation(); deleteTrack(tr.id); }}
                />
              </Space>
            </List.Item>
          )}
        />
      </div>
    </div>
  );
};

export default CVATAnnotator;
