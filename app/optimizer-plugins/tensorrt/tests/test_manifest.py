# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Tests for manifest.yaml validation
"""

import yaml
import pytest
from pathlib import Path


MANIFEST_PATH = Path(__file__).parent.parent / "manifest.yaml"


class TestManifest:
    """Test manifest.yaml required fields and artifact-spec compliance."""
    
    @pytest.fixture
    def manifest(self):
        with open(MANIFEST_PATH, 'r') as f:
            return yaml.safe_load(f)
    
    def test_required_fields(self, manifest):
        """Test all required fields from plugin-spec §2 are present."""
        required_fields = [
            "name", "kind", "version", "vendor",
            "supported_formats", "hardware_target",
            "supported_precision", "architectures", "resources"
        ]
        
        for field in required_fields:
            assert field in manifest, f"Missing required field: {field}"
    
    def test_name(self, manifest):
        assert manifest["name"] == "nereid-tensorrt"
    
    def test_kind(self, manifest):
        assert manifest["kind"] == "optimizer"
    
    def test_version(self, manifest):
        assert manifest["version"] == "1.0.0"
    
    def test_vendor(self, manifest):
        assert manifest["vendor"] == "nvidia"
    
    def test_supported_formats(self, manifest):
        formats = manifest["supported_formats"]
        assert "input" in formats
        assert "output" in formats
        assert "onnx" in formats["input"]
        assert "plan" in formats["output"]
    
    def test_hardware_target(self, manifest):
        assert manifest["hardware_target"] == "nvidia-tensorrt"
    
    def test_supported_precision(self, manifest):
        precision = manifest["supported_precision"]
        assert "fp32" in precision
        assert "fp16" in precision
        assert "int8" in precision
    
    def test_architectures(self, manifest):
        archs = manifest["architectures"]
        assert "x86_64" in archs
        assert "arm64" in archs
    
    def test_resources(self, manifest):
        resources = manifest["resources"]
        assert "requests" in resources
        assert "limits" in resources
        
        requests = resources["requests"]
        assert requests["cpu"] == "2"
        assert requests["memory"] == "8Gi"
        assert requests["gpu"] == "1"
        
        limits = resources["limits"]
        assert limits["cpu"] == "8"
        assert limits["memory"] == "32Gi"
        assert limits["gpu"] == "1"
    
    def test_config_schema(self, manifest):
        """Test config schema matches plugin.py expectations."""
        schema = manifest.get("config_schema", {})
        assert schema.get("type") == "object"
        
        props = schema.get("properties", {})
        assert "precision" in props
        assert "calibration_uri" in props
        assert "workspace_size" in props
        assert "batch_size" in props
        
        # Check precision enum
        precision_prop = props["precision"]
        assert precision_prop["type"] == "string"
        assert set(precision_prop["enum"]) == {"fp32", "fp16", "int8"}
        
        # Check required
        required = schema.get("required", [])
        assert "precision" in required


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
