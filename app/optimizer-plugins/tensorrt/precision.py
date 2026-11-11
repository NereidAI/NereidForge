# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Precision Validator: Compares FP32 ONNX model outputs with TensorRT engine outputs

Metrics:
- Cosine similarity
- Maximum absolute error
- Mean relative error
- NaN/Inf count
"""

import json
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Optional

from gpu_runtime import initialize_cuda

try:
    import tensorrt as trt
    import pycuda.driver as cuda
except ModuleNotFoundError:
    # Metric-only unit tests do not need a GPU; validate() rejects hardware use.
    trt = None
    cuda = None

logger = logging.getLogger(__name__)


class PrecisionValidator:
    """Validates numerical precision of TensorRT engine against ONNX reference."""
    
    def __init__(
        self,
        num_samples: int = 100,
        rtol: float = 1e-3,
        atol: float = 1e-5
    ):
        self.num_samples = num_samples
        self.rtol = rtol
        self.atol = atol
        self.trt_logger = trt.Logger(trt.Logger.WARNING) if trt is not None else None
    
    def validate(
        self,
        onnx_path: Path,
        plan_path: Path,
        precision: str = "fp32",
        calibration_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Compare ONNX (FP32) and TensorRT engine outputs.
        
        Returns:
            Dictionary with precision metrics
        """
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT and PyCUDA are required for engine validation")
        initialize_cuda()

        # Load ONNX model
        onnx_session = ort.InferenceSession(str(onnx_path))
        onnx_inputs = onnx_session.get_inputs()
        
        # Load TensorRT engine
        with open(plan_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Get input/output info
        input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) 
                       if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) 
                        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        
        logger.info(f"Validating precision: {len(input_names)} inputs, {len(output_names)} outputs")
        
        all_metrics = {
            "cosine_similarity": [],
            "max_abs_error": [],
            "mean_rel_error": [],
            "nan_or_inf_count": 0,
            "num_samples_tested": 0,
            "per_output_metrics": {}
        }
        
        for sample_idx in range(self.num_samples):
            # Generate random input data
            feed_dict = {}
            trt_buffers = {}
            
            for inp in onnx_inputs:
                shape = list(inp.shape)
                # Replace dynamic dimensions with batch size
                shape = [1 if dim <= 0 else dim for dim in shape]
                data = np.random.randn(*shape).astype(np.float32)
                feed_dict[inp.name] = data
            
            # Run ONNX inference
            onnx_results = onnx_session.run(None, feed_dict)
            
            # Prepare TensorRT buffers
            for i, name in enumerate(input_names):
                data = feed_dict[name]
                d_input = cuda.mem_alloc(data.nbytes)
                cuda.memcpy_htod(d_input, data)
                trt_buffers[name] = d_input
                context.set_input_shape(name, data.shape)
            
            # Allocate output buffers
            output_buffers = {}
            for name in output_names:
                shape = context.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                h_output = np.empty(shape, dtype=dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)
                output_buffers[name] = (h_output, d_output)
                trt_buffers[name] = d_output
            
            # Set tensor addresses
            for name, buf in trt_buffers.items():
                context.set_tensor_address(name, int(buf))
            
            # Run TensorRT inference
            context.execute_async_v3(stream_handle=0)
            cuda.Context.synchronize()
            
            # Copy outputs back
            trt_results = {}
            for name, (h_output, d_output) in output_buffers.items():
                cuda.memcpy_dtoh(h_output, d_output)
                trt_results[name] = h_output
            
            # Compare outputs
            for i, out_name in enumerate(output_names):
                onnx_out = onnx_results[i]
                trt_out = trt_results[out_name]
                
                # Ensure same shape
                if onnx_out.shape != trt_out.shape:
                    logger.warning(f"Shape mismatch for {out_name}: ONNX {onnx_out.shape} vs TRT {trt_out.shape}")
                    continue
                
                metrics = self._compute_metrics(onnx_out, trt_out, out_name)
                
                all_metrics["cosine_similarity"].append(metrics["cosine_similarity"])
                all_metrics["max_abs_error"].append(metrics["max_abs_error"])
                all_metrics["mean_rel_error"].append(metrics["mean_rel_error"])
                all_metrics["nan_or_inf_count"] += metrics["nan_or_inf_count"]
                
                if out_name not in all_metrics["per_output_metrics"]:
                    all_metrics["per_output_metrics"][out_name] = {
                        "cosine_similarity": [],
                        "max_abs_error": [],
                        "mean_rel_error": [],
                        "nan_or_inf_count": 0
                    }
                
                all_metrics["per_output_metrics"][out_name]["cosine_similarity"].append(
                    metrics["cosine_similarity"]
                )
                all_metrics["per_output_metrics"][out_name]["max_abs_error"].append(
                    metrics["max_abs_error"]
                )
                all_metrics["per_output_metrics"][out_name]["mean_rel_error"].append(
                    metrics["mean_rel_error"]
                )
                all_metrics["per_output_metrics"][out_name]["nan_or_inf_count"] += (
                    metrics["nan_or_inf_count"]
                )
            
            all_metrics["num_samples_tested"] += 1
            
            # Cleanup
            for buf in trt_buffers.values():
                buf.free()
        
        # Aggregate results
        report = self._aggregate_metrics(all_metrics)
        report["precision_mode"] = precision
        report["num_samples"] = self.num_samples
        
        return report
    
    def _compute_metrics(
        self,
        ref: np.ndarray,
        test: np.ndarray,
        name: str
    ) -> Dict[str, float]:
        """Compute precision metrics between reference and test outputs."""
        # Flatten arrays
        ref_flat = ref.flatten()
        test_flat = test.flatten()
        
        # Check for NaN/Inf
        nan_inf_ref = np.sum(np.isnan(ref_flat) | np.isinf(ref_flat))
        nan_inf_test = np.sum(np.isnan(test_flat) | np.isinf(test_flat))
        nan_or_inf_count = int(nan_inf_ref + nan_inf_test)
        
        if nan_or_inf_count > 0:
            logger.warning(f"{name}: Found {nan_or_inf_count} NaN/Inf values")
        
        # Cosine similarity
        ref_norm = np.linalg.norm(ref_flat)
        test_norm = np.linalg.norm(test_flat)
        if ref_norm > 0 and test_norm > 0:
            cosine_sim = np.dot(ref_flat, test_flat) / (ref_norm * test_norm)
        else:
            cosine_sim = 1.0 if ref_norm == test_norm == 0 else 0.0
        
        # Maximum absolute error
        max_abs_error = np.max(np.abs(ref_flat - test_flat))
        
        # Mean relative error (avoid division by zero)
        abs_ref = np.abs(ref_flat)
        rel_error = np.abs(ref_flat - test_flat) / np.maximum(abs_ref, 1e-8)
        mean_rel_error = np.mean(rel_error)
        
        return {
            "cosine_similarity": float(cosine_sim),
            "max_abs_error": float(max_abs_error),
            "mean_rel_error": float(mean_rel_error),
            "nan_or_inf_count": nan_or_inf_count
        }
    
    def _aggregate_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics."""
        report = {
            "summary": {
                "cosine_similarity": {
                    "mean": float(np.mean(all_metrics["cosine_similarity"])),
                    "min": float(np.min(all_metrics["cosine_similarity"])),
                    "max": float(np.max(all_metrics["cosine_similarity"])),
                    "std": float(np.std(all_metrics["cosine_similarity"]))
                },
                "max_abs_error": {
                    "mean": float(np.mean(all_metrics["max_abs_error"])),
                    "min": float(np.min(all_metrics["max_abs_error"])),
                    "max": float(np.max(all_metrics["max_abs_error"])),
                    "std": float(np.std(all_metrics["max_abs_error"]))
                },
                "mean_rel_error": {
                    "mean": float(np.mean(all_metrics["mean_rel_error"])),
                    "min": float(np.min(all_metrics["mean_rel_error"])),
                    "max": float(np.max(all_metrics["mean_rel_error"])),
                    "std": float(np.std(all_metrics["mean_rel_error"]))
                },
                "nan_or_inf_count": all_metrics["nan_or_inf_count"],
                "num_samples_tested": all_metrics["num_samples_tested"]
            },
            "per_output": {}
        }
        
        for out_name, metrics in all_metrics["per_output_metrics"].items():
            report["per_output"][out_name] = {
                "cosine_similarity": {
                    "mean": float(np.mean(metrics["cosine_similarity"])),
                    "min": float(np.min(metrics["cosine_similarity"])),
                    "max": float(np.max(metrics["cosine_similarity"])),
                    "std": float(np.std(metrics["cosine_similarity"]))
                },
                "max_abs_error": {
                    "mean": float(np.mean(metrics["max_abs_error"])),
                    "min": float(np.min(metrics["max_abs_error"])),
                    "max": float(np.max(metrics["max_abs_error"])),
                    "std": float(np.std(metrics["max_abs_error"]))
                },
                "mean_rel_error": {
                    "mean": float(np.mean(metrics["mean_rel_error"])),
                    "min": float(np.min(metrics["mean_rel_error"])),
                    "max": float(np.max(metrics["mean_rel_error"])),
                    "std": float(np.std(metrics["mean_rel_error"]))
                },
                "nan_or_inf_count": metrics["nan_or_inf_count"]
            }
        
        # Overall pass/fail based on thresholds
        report["passed"] = (
            report["summary"]["cosine_similarity"]["mean"] > 0.99 and
            report["summary"]["max_abs_error"]["mean"] < 1e-2 and
            report["summary"]["mean_rel_error"]["mean"] < 1e-2 and
            report["summary"]["nan_or_inf_count"] == 0
        )
        
        return report


def validate_precision(
    onnx_path: str,
    plan_path: str,
    precision: str = "fp32",
    num_samples: int = 100
) -> Dict[str, Any]:
    """Convenience function for precision validation."""
    validator = PrecisionValidator(num_samples=num_samples)
    return validator.validate(
        onnx_path=Path(onnx_path),
        plan_path=Path(plan_path),
        precision=precision
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate TensorRT engine precision")
    parser.add_argument("--onnx", required=True, help="Reference ONNX model path")
    parser.add_argument("--plan", required=True, help="TensorRT engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--samples", type=int, default=100)
    
    args = parser.parse_args()
    
    report = validate_precision(
        onnx_path=args.onnx,
        plan_path=args.plan,
        precision=args.precision,
        num_samples=args.samples
    )
    
    print(json.dumps(report, indent=2))
    
    import sys
    sys.exit(0 if report["passed"] else 1)
