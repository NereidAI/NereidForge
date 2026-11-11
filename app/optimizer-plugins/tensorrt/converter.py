# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT Converter: ONNX -> TensorRT .plan engine

Uses TensorRT Python API to build optimized engines with support for:
- Explicit batch dimensions
- Dynamic optimization profiles
- FP32, FP16, and INT8 precision modes
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

import tensorrt as trt
import numpy as np

logger = logging.getLogger(__name__)


class TensorRTConverter:
    """Converts ONNX models to TensorRT engines."""
    
    def __init__(
        self,
        workspace_size: int = 1073741824,  # 1GB default
        batch_size: int = 1,
        logger_level: int = trt.Logger.WARNING
    ):
        self.workspace_size = workspace_size
        self.batch_size = batch_size
        self.trt_logger = trt.Logger(logger_level)
    
    def convert(
        self,
        onnx_path: Path,
        plan_path: Path,
        precision: str = "fp32",
        calibration_path: Optional[Path] = None
    ) -> None:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to input ONNX model
            plan_path: Path to output TensorRT engine (.plan)
            precision: Target precision (fp32, fp16, int8)
            calibration_path: Path to calibration data (required for INT8)
        """
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
        
        # Create network with explicit batch
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        
        # Parse ONNX
        parser = trt.OnnxParser(network, self.trt_logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(parser.get_error(i))
                raise RuntimeError(f"ONNX parsing failed: {errors}")
        
        logger.info(f"Parsed ONNX model with {network.num_inputs} inputs, {network.num_outputs} outputs")
        
        # Configure optimization profiles for dynamic shapes
        self._setup_optimization_profiles(builder, config, network)
        
        # Configure precision
        self._configure_precision(config, precision, calibration_path)
        
        # Build engine
        logger.info(f"Building TensorRT engine with precision={precision}...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"Engine saved to {plan_path}")
    
    def _setup_optimization_profiles(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        network: trt.INetworkDefinition
    ) -> None:
        """Set up optimization profiles for dynamic input shapes."""
        profile = builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            shape = input_tensor.shape
            
            # Handle dynamic dimensions (-1)
            min_shape = []
            opt_shape = []
            max_shape = []
            
            for dim in shape:
                if dim == -1:
                    # Dynamic dimension: use batch_size as default
                    min_shape.append(1)
                    opt_shape.append(self.batch_size)
                    max_shape.append(self.batch_size * 4)
                else:
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        
        config.add_optimization_profile(profile)
        logger.info("Added optimization profile for dynamic shapes")
    
    def _configure_precision(
        self,
        config: trt.IBuilderConfig,
        precision: str,
        calibration_path: Optional[Path]
    ) -> None:
        """Configure precision settings (FP32, FP16, INT8)."""
        if precision == "fp16":
            if config.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 mode enabled")
            else:
                logger.warning("FP16 not supported on this platform, falling back to FP32")
        
        elif precision == "int8":
            if config.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 mode enabled")
                
                if calibration_path is None:
                    raise ValueError("INT8 precision requires calibration data")
                
                # Set up calibrator
                calibrator = self._create_calibrator(calibration_path, config)
                config.int8_calibrator = calibrator
            else:
                logger.warning("INT8 not supported on this platform, falling back to FP32")
        
        # FP32 is default, no flags needed
    
    def _create_calibrator(
        self,
        calibration_path: Path,
        config: trt.IBuilderConfig
    ) -> trt.IInt8Calibrator:
        """Create INT8 calibrator from calibration data."""
        # Load calibration data
        if calibration_path.suffix == '.npy':
            calibration_data = np.load(calibration_path)
        elif calibration_path.suffix == '.npz':
            calibration_data = np.load(calibration_path)
            # Assume first array
            calibration_data = list(calibration_data.values())[0]
        else:
            # Raw binary data
            calibration_data = np.fromfile(calibration_path, dtype=np.float32)
        
        # Reshape if needed
        if calibration_data.ndim == 1:
            # Assume batch of single samples
            calibration_data = calibration_data.reshape(-1, 1, 1, 1)
        
        class Int8Calibrator(trt.IInt8Calibrator):
            def __init__(self, data: np.ndarray, cache_file: str = "calibration.cache"):
                trt.IInt8Calibrator.__init__(self)
                self.data = data
                self.cache_file = cache_file
                self.batch_size = 1
                self.current_index = 0
                
                # Try to load existing cache
                self.cache = None
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        self.cache = f.read()
            
            def get_batch_size(self) -> int:
                return self.batch_size
            
            def get_batch(self, names: List[str], p_str: List[int]) -> List[int]:
                if self.current_index >= len(self.data):
                    return None
                
                batch = self.data[self.current_index:self.current_index + self.batch_size]
                self.current_index += self.batch_size
                
                # Allocate device memory
                import pycuda.driver as cuda
                import pycuda.autoinit  # noqa: F401  # Initialize CUDA before calibration buffer allocation.
                
                d_batch = cuda.mem_alloc(batch.nbytes)
                cuda.memcpy_htod(d_batch, batch)
                
                return [int(d_batch)]
            
            def read_calibration_cache(self) -> bytes:
                return self.cache
            
            def write_calibration_cache(self, cache: bytes) -> None:
                self.cache = cache
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)
        
        return Int8Calibrator(calibration_data)


def convert_onnx_to_trt(
    onnx_path: str,
    plan_path: str,
    precision: str = "fp32",
    workspace_size: int = 1073741824,
    batch_size: int = 1,
    calibration_path: Optional[str] = None
) -> None:
    """
    Convenience function for direct conversion.
    
    Args:
        onnx_path: Path to ONNX model
        plan_path: Path to output TensorRT engine
        precision: Target precision (fp32, fp16, int8)
        workspace_size: Max workspace size in bytes
        batch_size: Batch size for optimization
        calibration_path: Path to calibration data (for INT8)
    """
    converter = TensorRTConverter(workspace_size=workspace_size, batch_size=batch_size)
    converter.convert(
        onnx_path=Path(onnx_path),
        plan_path=Path(plan_path),
        precision=precision,
        calibration_path=Path(calibration_path) if calibration_path else None
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--plan", required=True, help="Output TensorRT engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--workspace", type=int, default=1073741824)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration", help="Calibration data path (for INT8)")
    
    args = parser.parse_args()
    
    convert_onnx_to_trt(
        onnx_path=args.onnx,
        plan_path=args.plan,
        precision=args.precision,
        workspace_size=args.workspace,
        batch_size=args.batch_size,
        calibration_path=args.calibration
    )
