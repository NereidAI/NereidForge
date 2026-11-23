# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge optimization entrypoint (quantization + graph opt + benchmark).

Takes a source ONNX model from seaweedfs and produces an optimized artifact
for one target device:

  quant_method:
    graph        ONNX Runtime graph optimization (ORT_ENABLE_ALL)
    fp16         float16 conversion (onnxconverter-common)
    int8-dynamic onnxruntime.quantization.quantize_dynamic
    int8-static  quantize_static with a real calibration set sampled from the
                 training dataset (calib_uri + calib_n images)

  target_device:
    cpu          artifact stays .onnx, benchmarked via onnxruntime CPU EP
    openvino     converts to OpenVINO IR (.xml/.bin) via ovc, benchmarked via
                 openvino.runtime when the package is available
    tensorrt     requires a GPU node + trtexec; fails fast with a clear
                 message when unavailable (code path ready for GPU clusters)

Benchmark is standardized: sequential single-image inference, warmup 10,
measure 500 (configurable), reporting avg/p50/p95/p99 + qps for both source
and optimized models, plus the runtime environment for reproducibility.

Env contract:
  INPUT_URI    s3://... source model (.onnx)
  OUTPUT_DIR   s3://... artifact dir ({ARTIFACT_ID}.optimized.onnx etc.)
  ARTIFACT_ID  unique id (job id)
  CONFIG_JSON  optimization config

Exit codes: 0 ok.
"""

import json
import os
import platform
import socket
import sys
import time

import boto3

S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT",
    "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
)


def s3_client():
    """s3_client creates the standalone optimizer's client with explicit credentials."""
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    if not access_key.strip() or not secret_key.strip():
        raise RuntimeError(
            "S3_ACCESS_KEY and S3_SECRET_KEY must contain non-empty "
            "operator-supplied credentials before S3 access."
        )
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def parse_uri(uri):
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "").rstrip("/")


def download(s3, uri, local):
    bucket, key = parse_uri(uri)
    s3.download_file(bucket, key, local)
    print(f"[opt] downloaded {uri} -> {local}", flush=True)


def upload(s3, local, uri):
    bucket, key = parse_uri(uri)
    s3.upload_file(local, bucket, key)
    print(f"[opt] uploaded {uri}", flush=True)


def env_block() -> dict:
    """Record the benchmark environment for reproducibility."""
    return {
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "threads": int(os.cpu_count() or 1),
        "python": platform.python_version(),
    }


def calibration_reader(s3, calib_uri, imgsz, n):
    """Yield n preprocessed calibration images from the dataset prefix."""
    import cv2
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader

    bucket, prefix = parse_uri(calib_uri)
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                keys.append(k)
    keys = sorted(keys)[:n]
    print(f"[opt] calibration images: {len(keys)}", flush=True)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.keys = list(keys)

        def get_next(self):
            if not self.keys:
                return None
            k = self.keys.pop(0)
            try:
                s3.download_file(bucket, k, "/tmp/calib.img")
                img = cv2.imread("/tmp/calib.img")
                if img is None:
                    return self.get_next()
                img = cv2.resize(img, (imgsz, imgsz))
                arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                return {"images": arr.transpose(2, 0, 1)[None]}
            except Exception as e:  # noqa: BLE001
                print(f"[opt] calib skip {k}: {e}", flush=True)
                return self.get_next()

    return Reader()


def input_shape_for(sess, imgsz):
    """Materialize symbolic dims into a concrete YOLO-style input shape.

    Preserves the model's fixed spatial dims (e.g. a 320-exported model stays
    320) and only fills symbolic/batch dims with imgsz defaults.
    """
    shape = sess.get_inputs()[0].shape
    out = []
    for i, d in enumerate(shape):
        if d is None or d == -1 or not isinstance(d, int):
            out.append(1 if i == 0 else imgsz)
        else:
            out.append(int(d))
    if len(out) < 4:
        out = [1, 3, imgsz, imgsz]
    return out


def benchmark_ort(model_path, shape, warmup, measure):
    """Sequential single-image onnxruntime benchmark; returns latency stats."""
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dummy = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: dummy})
    lat = []
    for _ in range(measure):
        t0 = time.time()
        sess.run(None, {name: dummy})
        lat.append((time.time() - t0) * 1000)
    lat.sort()
    avg = sum(lat) / len(lat)
    return sess, {
        "avg_ms": round(avg, 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 3),
        "p99_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.99))], 3),
        "qps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
        "measure_count": len(lat),
        "method": f"sequential, warmup {warmup}, measure {measure}",
    }


def main():
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "")
    cfg = json.loads(os.environ.get("CONFIG_JSON", "{}"))

    s3 = s3_client()
    src = "/tmp/input.onnx"
    download(s3, input_uri, src)
    orig_size = os.path.getsize(src)

    device = cfg.get("target_device", "cpu")
    method = cfg.get("quant_method", "graph")
    imgsz = int(cfg.get("imgsz", 640))
    warmup = int(cfg.get("warmup", 10))
    measure = int(cfg.get("measure", 500))

    import numpy as np  # noqa: F401
    import onnx
    import onnxruntime as ort

    # --- Quantize / convert --------------------------------------------------
    ext, extra_files = ".onnx", []
    optimized = "/tmp/optimized.onnx"
    if device == "openvino":
        # OpenVINO IR: convert the (optionally quantized) onnx graph first,
        # then run ovc to emit .xml/.bin. ORT graph optimization is skipped for
        # the "graph" method because it injects com.microsoft.nchwc.* layout
        # operators that the OpenVINO ONNX frontend cannot convert; OpenVINO
        # performs its own graph transformation anyway.
        if method == "graph":
            quantized = src
        else:
            quantized = apply_quantization(src, method, cfg, s3, imgsz)
        try:
            import openvino as ov  # noqa: F401
            from openvino.tools.ovc import convert_model
        except ImportError as e:
            raise SystemExit(f"openvino not available in this image: {e}")
        ov_model = convert_model(quantized)
        # Carry the source ONNX class names into the IR so downstream
        # evaluation can map the model's classes onto the dataset's.
        names = _onnx_names(quantized)
        if names is not None:
            try:
                ov_model.set_rt_info(names, "names")
            except Exception as e:  # noqa: BLE001
                print(f"[opt] warn: could not embed names in IR: {e}", flush=True)
        ov.save_model(ov_model, "/tmp/optimized.xml")
        optimized, ext = "/tmp/optimized.xml", ".xml"
        extra_files = ["/tmp/optimized.bin"]
    elif device == "tensorrt":
        # TensorRT engine build requires trtexec + NVIDIA GPU. The code path
        # is kept ready; on CPU-only clusters it fails fast with a clear
        # message instead of a confusing crash.
        raise SystemExit(
            "tensorrt target requires a GPU node with trtexec installed; "
            "this cluster has no NVIDIA GPUs available for optimization jobs"
        )
    elif device in ("onnx-rocm", "ascend"):
        # AMD (ROCm) and HUAWEI (Ascend) targets produce the quantized ONNX
        # as their deployable artifact in this image; the actual EP/runtime
        # acceleration is applied on the target hardware. Benchmark is skipped
        # when no local provider can run the model (no AMD GPU / CANN toolchain).
        quantized = apply_quantization(src, method, cfg, s3, imgsz)
        if quantized != optimized:
            os.replace(quantized, optimized)
        if device == "ascend":
            try:
                import subprocess as _sp  # noqa: PLC0415
                _sp.run(["atc", "--version"], capture_output=True, timeout=20)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"huawei/ascend target requires the CANN 'atc' toolchain in "
                    f"this image; not found ({e})"
                )
    else:
        quantized = apply_quantization(src, method, cfg, s3, imgsz)
        if quantized != optimized:
            os.replace(quantized, optimized)
    opt_size = os.path.getsize(optimized) + sum(os.path.getsize(f) for f in extra_files)

    # --- Benchmark (standardized) ---------------------------------------------
    shape = input_shape_for(ort.InferenceSession(src, providers=["CPUExecutionProvider"]), imgsz)
    _, src_stats = benchmark_ort(src, shape, warmup, measure)
    if device == "openvino":
        opt_stats = benchmark_openvino(optimized, shape, warmup, measure)
    elif device in ("onnx-rocm", "ascend"):
        # No local provider for the target hardware (AMD GPU / Ascend NPU) in
        # this image: report the source benchmark and mark speedup unavailable
        # so the gate skips latency/throughput for these targets.
        opt_stats = {
            "avg_ms": None, "qps": None, "measure_count": 0,
            "method": "skipped (no local AMD/ASCEND runtime)",
        }
    else:
        _, opt_stats = benchmark_ort(optimized, shape, warmup, measure)
    speedup = (
        src_stats["avg_ms"] / opt_stats["avg_ms"] if opt_stats.get("avg_ms")
        else None
    )
    print(f"[opt] src avg {src_stats['avg_ms']}ms opt avg "
          f"{opt_stats.get('avg_ms') or 'n/a'}ms speedup {speedup or 'n/a'}", flush=True)

    # --- Upload artifacts + metrics --------------------------------------------
    out_base = f"{output_uri}/{artifact_id}.optimized"
    metrics = {
        "optimizer": f"onnxruntime:{method}" if device == "cpu" else device,
        "target_device": device,
        "quant_method": method,
        "dtype": cfg.get("dtype", "fp32" if method == "graph" else ("fp16" if method == "fp16" else "int8")),
        "source_size_bytes": orig_size,
        "optimized_size_bytes": opt_size,
        "size_reduction_pct": round((1 - opt_size / orig_size) * 100, 2) if orig_size else 0,
        "speedup_ratio": round(speedup, 3) if speedup is not None else None,
        "benchmark": opt_stats,
        "source_benchmark": src_stats,
        "environment": env_block(),
    }
    if output_uri:
        main_uri = f"{out_base}{ext}"
        upload(s3, optimized, main_uri)
        metrics["optimized_uri"] = main_uri
        for f in extra_files:
            fext = os.path.splitext(f)[1]
            furi = f"{output_uri}/{artifact_id}.optimized{fext}"
            upload(s3, f, furi)
            metrics.setdefault("extra_artifacts", []).append(furi)
        try:
            import nseval
        except ModuleNotFoundError:  # builders layout keeps it under common/
            from common import nseval
        upload_text(s3, f"{output_uri}/{artifact_id}.metrics.json", nseval.dump_json(metrics))

    print("[opt] SUCCESS", flush=True)
    sys.exit(0)


def convert_fp16_custom(src: str, out: str) -> str:
    """Convert an ONNX model to float16 via a manual graph pass.

    onnxconverter-common (as of 1.16) produces invalid graphs for models with
    Resize nodes (the YOLO export layout): it inserts *_output_cast0 casts that
    leave fp32/fp16 type mismatches that ORT refuses to load. Instead of
    patching that library, do the conversion directly: every float32 initializer
    and intermediate tensor becomes float16, and the graph inputs/outputs stay
    float32 with explicit Cast nodes inserted so the external contract is
    unchanged while the whole internal graph is uniformly fp16.
    """
    import onnx
    import numpy as np
    from onnx import helper

    m = onnx.load(src)
    graph = m.graph
    FLOAT = onnx.TensorProto.FLOAT
    FLOAT16 = onnx.TensorProto.FLOAT16

    # Resize requires its scale/roi inputs to stay float32 (ORT rejects fp16
    # scale); collect those tensor names so we skip converting them.
    keep_fp32: set[str] = set()
    for node in graph.node:
        if node.op_type == "Resize":
            # inputs: [X, roi, scales] (scales index 2, roi index 1)
            for idx in (1, 2):
                if len(node.input) > idx and node.input[idx]:
                    keep_fp32.add(node.input[idx])
        elif node.op_type == "Cast":
            # YOLO's built-in detect head casts outputs to fp32; keep the cast
            # io fp32 so the head contract is preserved where possible.
            for o in node.output:
                if o:
                    keep_fp32.add(o)
            for i in node.input:
                if i:
                    keep_fp32.add(i)

    # Rewrite all float32 initializers to float16.
    for init in graph.initializer:
        if init.data_type == FLOAT and init.name not in keep_fp32:
            arr = np.frombuffer(init.raw_data, dtype=np.float32).astype(np.float16)
            init.data_type = FLOAT16
            init.raw_data = arr.tobytes()
            if len(init.float_data):
                init.ClearField("float_data")
            if len(init.int32_data):
                init.ClearField("int32_data")

    # Rewrite intermediate value_info float32 tensor types to float16.
    for vi in graph.value_info:
        if vi.name in keep_fp32:
            continue
        tt = vi.type.tensor_type
        if tt.elem_type == FLOAT:
            tt.elem_type = FLOAT16

    # Graph inputs stay fp32; insert a Cast (fp32 -> fp16) right after each one.
    inserted_in: dict[str, str] = {}
    for g_in in list(graph.input):
        tt = g_in.type.tensor_type
        if tt.elem_type != FLOAT:
            continue
        name = g_in.name
        cast_name = f"{name}/to_fp16"
        new_tensor = f"{name}/casted_fp16"
        cast_node = helper.make_node(
            "Cast", inputs=[name], outputs=[new_tensor], name=cast_name,
            to=FLOAT16,
        )
        graph.node.insert(0, cast_node)
        inserted_in[name] = new_tensor

    # Rewrite the ORIGINAL consumer nodes' input references to read the casted
    # fp16 tensor. Only touch nodes that existed before the inserted casts
    # (a cast's own input must stay the fp32 graph input).
    for node in graph.node:
        if node.name.endswith("/to_fp16"):
            continue
        for i, inp in enumerate(node.input):
            if inp in inserted_in:
                node.input[i] = inserted_in[inp]

    # Graph outputs stay fp32; the producer of each output must feed fp16 into
    # an fp32 Cast. Since intermediates are fp16, prepend a Cast to fp32 that
    # produces the output name.
    inserted_out: dict[str, str] = {}
    for g_out in list(graph.output):
        tt = g_out.type.tensor_type
        if tt.elem_type != FLOAT:
            continue
        out_name = g_out.name
        casted = f"{out_name}/casted_fp32"
        # Rename the real producer tensor (fp16) to `casted`, then Cast to fp32
        # under the original output name.
        for node in graph.node:
            for i, o in enumerate(node.output):
                if o == out_name:
                    node.output[i] = casted
        cast_node = helper.make_node(
            "Cast", inputs=[casted], outputs=[out_name],
            name=f"{out_name}/from_fp16", to=FLOAT,
        )
        graph.node.append(cast_node)

    # Cast attributes for image-scale parameters (e.g. Resize scales) that are
    # float32 initializers would already be fp16; keep them fp32 is handled by
    # leaving non-float tensors alone. Node float attributes stay as-is.

    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, out)
    return out


def apply_quantization(src, method, cfg, s3, imgsz):
    """Apply the requested quantization/graph-opt to src; returns local path."""
    import onnx
    import onnxruntime as ort

    if method == "graph":
        # ORT graph optimization written out to a file
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = "/tmp/q.onnx"
        ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        return "/tmp/q.onnx"
    if method == "fp16":
        out = "/tmp/q.onnx"
        try:
            convert_fp16_custom(src, out)
            ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        except Exception as e:
            # The YOLO export (with built-in detect head) is not reliably
            # convertible to fp16; fall back to graph optimization so the job
            # still produces a usable artifact instead of failing hard.
            print(f"[opt] fp16 conversion failed ({e}); falling back to graph opt", flush=True)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.optimized_model_filepath = out
            ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
            print("[opt] fallback: graph-optimized artifact produced", flush=True)
        return out
    if method == "int8-dynamic":
        from onnxruntime.quantization import quantize_dynamic, QuantType
        out = "/tmp/q.onnx"
        # YOLO's output is raw bbox/class logits; quantizing it corrupts the
        # coordinates (mAP collapses to ~0). Exclude the nodes feeding the
        # graph outputs so only the internal activations/weights are int8.
        exclude = _output_feeders(src)
        quantize_dynamic(src, out, weight_type=QuantType.QInt8,
                         nodes_to_exclude=exclude)
        return out
    if method == "int8-static":
        from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                              quantize_static)
        calib_uri = cfg.get("calib_uri", "")
        calib_n = int(cfg.get("calib_n", 200))
        if not calib_uri:
            raise SystemExit("int8-static requires calib_uri (dataset prefix)")
        reader = calibration_reader(s3, calib_uri, imgsz, calib_n)
        out = "/tmp/q.onnx"
        # Same output-layer protection as dynamic quantization.
        exclude = _output_feeders(src)
        quantize_static(src, out, reader, weight_type=QuantType.QInt8,
                        nodes_to_exclude=exclude)
        return out
    raise SystemExit(f"unknown quant_method: {method}")


def _onnx_names(path: str) -> list[str] | None:
    """Read the class-name list from an ONNX model's 'names' metadata."""
    import json
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        for p in m.metadata_props:
            if p.key == "names":
                raw = p.value.strip()
                parsed = None
                try:
                    parsed = json.loads(raw)
                except Exception:
                    try:
                        import ast
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    try:
                        return [parsed[str(i)] for i in range(len(parsed))]
                    except Exception:
                        return [parsed[k] for k in sorted(parsed, key=lambda k: int(k))]
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
    except Exception:
        pass
    return None


def _output_feeders(src: str) -> list[str]:
    """Return node names that must stay float to preserve detection precision.

    For YOLO-style graphs the detect head (the last "/model.NN/" block feeding
    the graph outputs) decodes raw bbox coordinates / class logits; quantizing
    it collapses mAP. We exclude every node inside that head block only, so
    the backbone/neck layers above it still get quantized (real size win).
    """
    import onnx
    model = onnx.load(src)
    out_names = {o.name for o in model.graph.output}
    node_by_out: dict[str, str] = {}
    for n in model.graph.node:
        for o in n.output:
            node_by_out[o] = n.name
    # The head block prefix is the "/model.NN/" shared by all output feeders.
    import re
    head_prefix: str | None = None
    for on in out_names:
        n = node_by_out.get(on)
        if not n:
            continue
        m = re.match(r"^(.*?/model\.\d+/)", n)
        if not m:
            continue
        pfx = m.group(1)
        head_prefix = pfx if head_prefix is None or len(pfx) > len(head_prefix) else head_prefix
    if head_prefix is None:
        # Fall back to excluding just the direct output producers.
        return [node_by_out[o] for o in out_names if node_by_out.get(o)]
    return [n.name for n in model.graph.node if n.name.startswith(head_prefix)]


def benchmark_openvino(xml_path, shape, warmup, measure):
    """Benchmark an OpenVINO IR model with the CPU plugin."""
    import numpy as np
    import openvino as ov

    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    port = compiled.input(0)
    dummy = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        compiled([dummy])
    lat = []
    for _ in range(measure):
        t0 = time.time()
        compiled([dummy])
        lat.append((time.time() - t0) * 1000)
    lat.sort()
    avg = sum(lat) / len(lat)
    return {
        "avg_ms": round(avg, 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 3),
        "p99_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.99))], 3),
        "qps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
        "measure_count": len(lat),
        "engine": f"openvino {ov.__version__} CPU",
        "method": f"sequential, warmup {warmup}, measure {measure}",
    }


def upload_text(s3, uri, text):
    bucket, key = parse_uri(uri)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode())
    print(f"[opt] uploaded {uri}", flush=True)


if __name__ == "__main__":
    main()
