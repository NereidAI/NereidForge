# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark Runner: Measures latency, throughput, and memory usage of TensorRT engines

Metrics:
- Latency: p50, p95, p99, p999
- Throughput: QPS (queries per second)
- Peak GPU memory usage
"""

import logging
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from pathlib import Path
from typing import Dict, Any, List, Optional

from gpu_runtime import initialize_cuda

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks on TensorRT engines."""
    
    def __init__(
        self,
        warmup_iterations: int = 10,
        measure_iterations: int = 100,
        batch_size: int = 1
    ):
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
        self.batch_size = batch_size
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
    
    def run(
        self,
        plan_path: Path,
        precision: str = "fp32",
        input_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark on TensorRT engine.
        
        Returns:
            Dictionary with benchmark metrics
        """
        initialize_cuda()

        # Load engine
        with open(plan_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Get I/O tensor info
        input_names = []
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        
        logger.info(f"Benchmarking engine: {len(input_names)} inputs, {len(output_names)} outputs")
        
        # Prepare input data if not provided
        if input_data is None:
            input_data = self._generate_random_inputs(context, engine, input_names)
        
        # Set input shapes
        for name in input_names:
            context.set_input_shape(name, input_data[name].shape)
        
        # Allocate buffers
        buffers = self._allocate_buffers(context, engine, input_names, output_names, input_data)
        
        # Warmup
        logger.info(f"Warming up ({self.warmup_iterations} iterations)...")
        for _ in range(self.warmup_iterations):
            self._run_inference(context, buffers, input_names, output_names, input_data)
        
        # Measure
        logger.info(f"Measuring ({self.measure_iterations} iterations)...")
        latencies = []
        gpu_memory_before = self._get_gpu_memory()
        peak_memory = 0
        
        for i in range(self.measure_iterations):
            # Measure GPU memory
            current_memory = self._get_gpu_memory()
            peak_memory = max(peak_memory, current_memory)
            
            # Run inference with timing
            start = time.perf_counter()
            self._run_inference(context, buffers, input_names, output_names, input_data)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        gpu_memory_after = self._get_gpu_memory()
        peak_memory = max(peak_memory, gpu_memory_after)
        
        # Compute statistics
        latencies = np.array(latencies)
        
        report = {
            "precision": precision,
            "batch_size": self.batch_size,
            "warmup_iterations": self.warmup_iterations,
            "measure_iterations": self.measure_iterations,
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "p999": float(np.percentile(latencies, 99.9))
            },
            "throughput_qps": float(1000.0 / np.mean(latencies) * self.batch_size),
            "gpu_memory_mb": {
                "initial": gpu_memory_before,
                "peak": peak_memory,
                "final": gpu_memory_after,
                "delta": peak_memory - gpu_memory_before
            }
        }
        
        # Cleanup
        for buf in buffers.values():
            buf.free()
        
        logger.info(f"Benchmark complete: {report['throughput_qps']:.2f} QPS, "
                    f"p99 latency: {report['latency_ms']['p99']:.2f} ms")
        
        return report
    
    def _generate_random_inputs(
        self,
        context: trt.IExecutionContext,
        engine: trt.ICudaEngine,
        input_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Generate random input data for benchmarking."""
        input_data = {}
        
        for name in input_names:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            
            # Replace dynamic dimensions
            shape = [self.batch_size if dim <= 0 else dim for dim in shape]
            
            # Generate random data
            if dtype == np.float32 or dtype == np.float16:
                data = np.random.randn(*shape).astype(dtype)
            elif dtype == np.int32 or dtype == np.int64:
                data = np.random.randint(0, 1000, size=shape).astype(dtype)
            else:
                data = np.ones(shape, dtype=dtype)
            
            input_data[name] = data
        
        return input_data
    
    def _allocate_buffers(
        self,
        context: trt.IExecutionContext,
        engine: trt.ICudaEngine,
        input_names: List[str],
        output_names: List[str],
        input_data: Dict[str, np.ndarray]
    ) -> Dict[str, cuda.DeviceAllocation]:
        """Allocate device buffers for inputs and outputs."""
        buffers = {}
        
        # Input buffers
        for name in input_names:
            data = input_data[name]
            d_buffer = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(d_buffer, data)
            buffers[name] = d_buffer
            context.set_tensor_address(name, int(d_buffer))
        
        # Output buffers
        for name in output_names:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            h_buffer = np.empty(shape, dtype=dtype)
            d_buffer = cuda.mem_alloc(h_buffer.nbytes)
            buffers[name] = d_buffer
            context.set_tensor_address(name, int(d_buffer))
        
        return buffers
    
    def _run_inference(
        self,
        context: trt.IExecutionContext,
        buffers: Dict[str, cuda.DeviceAllocation],
        input_names: List[str],
        output_names: List[str],
        input_data: Dict[str, np.ndarray]
    ) -> None:
        """Run single inference iteration."""
        # Copy input data to device
        for name in input_names:
            data = input_data[name]
            cuda.memcpy_htod(buffers[name], data)
        
        # Execute
        context.execute_async_v3(stream_handle=0)
        cuda.Context.synchronize()
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        free, total = cuda.mem_get_info()
        used_mb = (total - free) / (1024 * 1024)
        return used_mb


def run_benchmark(
    plan_path: str,
    precision: str = "fp32",
    warmup: int = 10,
    measure: int = 100,
    batch_size: int = 1
) -> Dict[str, Any]:
    """Convenience function for running benchmarks."""
    runner = BenchmarkRunner(
        warmup_iterations=warmup,
        measure_iterations=measure,
        batch_size=batch_size
    )
    return runner.run(Path(plan_path), precision=precision)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engine")
    parser.add_argument("--plan", required=True, help="TensorRT engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    
    args = parser.parse_args()
    
    report = run_benchmark(
        plan_path=args.plan,
        precision=args.precision,
        warmup=args.warmup,
        measure=args.measure,
        batch_size=args.batch_size
    )
    
    print(json.dumps(report, indent=2))
