# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for end-to-end TensorRT conversion

Tests conversion of a small ONNX model (MNIST-like) and validates output artifacts.
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto

from gpu_runtime import initialize_cuda

pytest.importorskip("tensorrt", reason="TensorRT hardware integration requires the runtime")
pytest.importorskip("pycuda", reason="TensorRT hardware integration requires CUDA")
try:
    initialize_cuda()
except RuntimeError as exc:
    pytest.skip(str(exc), allow_module_level=True)

# Hardware-dependent imports must follow the collection skips.
from converter import TensorRTConverter  # noqa: E402
from precision import PrecisionValidator  # noqa: E402
from benchmark import BenchmarkRunner  # noqa: E402


class TestSmoke:
    """End-to-end smoke tests for TensorRT conversion."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        tmpdir = Path(tempfile.mkdtemp())
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def simple_onnx_model(self, temp_dir):
        """Create a simple MNIST-like ONNX model for testing."""
        # Input: [batch, 1, 28, 28]
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [1, 1, 28, 28]
        )
        
        # Output: [batch, 10]
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, 10]
        )
        
        # Simple conv -> relu -> fc
        conv_weight = helper.make_tensor(
            'conv_weight', TensorProto.FLOAT, [8, 1, 3, 3],
            np.random.randn(8, 1, 3, 3).astype(np.float32).flatten()
        )
        conv_bias = helper.make_tensor(
            'conv_bias', TensorProto.FLOAT, [8],
            np.zeros(8, dtype=np.float32)
        )
        
        fc_weight = helper.make_tensor(
            'fc_weight', TensorProto.FLOAT, [10, 8 * 26 * 26],
            np.random.randn(10, 8 * 26 * 26).astype(np.float32).flatten()
        )
        fc_bias = helper.make_tensor(
            'fc_bias', TensorProto.FLOAT, [10],
            np.zeros(10, dtype=np.float32)
        )
        
        nodes = [
            helper.make_node('Conv', ['input', 'conv_weight', 'conv_bias'], ['conv_out'],
                           kernel_shape=[3, 3], pads=[0, 0, 0, 0]),
            helper.make_node('Relu', ['conv_out'], ['relu_out']),
            helper.make_node('Flatten', ['relu_out'], ['flat_out']),
            helper.make_node('Gemm', ['flat_out', 'fc_weight', 'fc_bias'], ['output'])
        ]
        
        graph = helper.make_graph(
            nodes, 'mnist_model',
            [input_tensor], [output_tensor],
            [conv_weight, conv_bias, fc_weight, fc_bias]
        )
        
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        
        onnx_path = temp_dir / "mnist.onnx"
        onnx.save(model, str(onnx_path))
        
        return onnx_path
    
    def test_fp32_conversion(self, temp_dir, simple_onnx_model):
        """Test FP32 conversion produces valid artifacts."""
        plan_path = temp_dir / "model_fp32.plan"
        
        converter = TensorRTConverter(workspace_size=1073741824, batch_size=1)
        converter.convert(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp32"
        )
        
        # Verify engine file exists and is non-empty
        assert plan_path.exists()
        assert plan_path.stat().st_size > 0
        
        # Verify engine can be loaded
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(plan_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine is not None
    
    def test_fp16_conversion(self, temp_dir, simple_onnx_model):
        """Test FP16 conversion produces valid artifacts."""
        plan_path = temp_dir / "model_fp16.plan"
        
        converter = TensorRTConverter(workspace_size=1073741824, batch_size=1)
        converter.convert(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp16"
        )
        
        assert plan_path.exists()
        assert plan_path.stat().st_size > 0
        
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(plan_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine is not None
    
    def test_precision_validation(self, temp_dir, simple_onnx_model):
        """Test precision validation runs without errors."""
        plan_path = temp_dir / "model_fp32.plan"
        
        converter = TensorRTConverter(workspace_size=1073741824, batch_size=1)
        converter.convert(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp32"
        )
        
        validator = PrecisionValidator(num_samples=10)  # Few samples for speed
        report = validator.validate(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp32"
        )
        
        # Check report structure
        assert "summary" in report
        assert "per_output" in report
        assert "passed" in report
        assert "precision_mode" in report
        assert report["precision_mode"] == "fp32"
        assert report["num_samples"] == 10
        
        # Check metrics exist
        assert "cosine_similarity" in report["summary"]
        assert "max_abs_error" in report["summary"]
        assert "mean_rel_error" in report["summary"]
        assert "nan_or_inf_count" in report["summary"]
    
    def test_benchmark(self, temp_dir, simple_onnx_model):
        """Test benchmark runs and produces metrics."""
        plan_path = temp_dir / "model_fp32.plan"
        
        converter = TensorRTConverter(workspace_size=1073741824, batch_size=1)
        converter.convert(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp32"
        )
        
        runner = BenchmarkRunner(warmup_iterations=2, measure_iterations=5)
        report = runner.run(plan_path=plan_path, precision="fp32")
        
        # Check report structure
        assert "precision" in report
        assert "latency_ms" in report
        assert "throughput_qps" in report
        assert "gpu_memory_mb" in report
        
        # Check latency percentiles
        latency = report["latency_ms"]
        assert "p50" in latency
        assert "p95" in latency
        assert "p99" in latency
        assert "p999" in latency
        
        # Check values are reasonable
        assert latency["p50"] > 0
        assert latency["p99"] >= latency["p50"]
        assert report["throughput_qps"] > 0
        assert report["gpu_memory_mb"]["peak"] >= 0
    
    def test_artifacts_completeness(self, temp_dir, simple_onnx_model):
        """Test all required artifacts are produced."""
        plan_path = temp_dir / "model.plan"
        metadata_path = temp_dir / "metadata.json"
        precision_report_path = temp_dir / "precision_report.json"
        benchmark_report_path = temp_dir / "benchmark_report.json"
        
        # Convert
        converter = TensorRTConverter(workspace_size=1073741824, batch_size=1)
        converter.convert(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp32"
        )
        
        # Precision validation
        validator = PrecisionValidator(num_samples=5)
        precision_report = validator.validate(
            onnx_path=simple_onnx_model,
            plan_path=plan_path,
            precision="fp32"
        )
        with open(precision_report_path, 'w') as f:
            json.dump(precision_report, f)
        
        # Benchmark
        runner = BenchmarkRunner(warmup_iterations=2, measure_iterations=3)
        benchmark_report = runner.run(plan_path=plan_path, precision="fp32")
        with open(benchmark_report_path, 'w') as f:
            json.dump(benchmark_report, f)
        
        # Metadata
        metadata = {
            "artifact_id": "test-artifact",
            "plugin": "nereid-tensorrt",
            "plugin_version": "1.0.0",
            "input_format": "onnx",
            "output_format": "plan",
            "precision": "fp32",
            "hardware_target": "nvidia-tensorrt",
            "files": {
                "model": "model.plan",
                "metadata": "metadata.json",
                "precision_report": "precision_report.json",
                "benchmark_report": "benchmark_report.json"
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Verify all artifacts exist
        assert plan_path.exists()
        assert metadata_path.exists()
        assert precision_report_path.exists()
        assert benchmark_report_path.exists()
        
        # Verify metadata.json structure
        with open(metadata_path) as f:
            meta = json.load(f)
        assert meta["artifact_id"] == "test-artifact"
        assert meta["plugin"] == "nereid-tensorrt"
        assert meta["output_format"] == "plan"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
