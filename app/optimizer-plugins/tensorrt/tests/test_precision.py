# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for precision validation metrics

Tests numerical accuracy of cosine similarity, max absolute error,
mean relative error, and NaN/Inf detection.
"""

from pathlib import Path

import numpy as np
import pytest

import gpu_runtime
import precision as precision_module
from precision import PrecisionValidator


class TestPrecisionMetrics:
    """Test precision metric computations."""
    
    @pytest.fixture
    def validator(self):
        return PrecisionValidator(num_samples=1)
    
    def test_cosine_similarity_identical(self, validator):
        """Cosine similarity should be 1.0 for identical arrays."""
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert np.isclose(metrics["cosine_similarity"], 1.0)
    
    def test_cosine_similarity_opposite(self, validator):
        """Cosine similarity should be -1.0 for opposite arrays."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert np.isclose(metrics["cosine_similarity"], -1.0)
    
    def test_cosine_similarity_orthogonal(self, validator):
        """Cosine similarity should be 0.0 for orthogonal arrays."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert np.isclose(metrics["cosine_similarity"], 0.0, atol=1e-6)
    
    def test_cosine_similarity_zero_vectors(self, validator):
        """Cosine similarity should handle zero vectors."""
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert metrics["cosine_similarity"] == 1.0
        
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert metrics["cosine_similarity"] == 0.0
    
    def test_max_abs_error(self, validator):
        """Max absolute error should be correct."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.1, 2.0, 2.9], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert np.isclose(metrics["max_abs_error"], 0.1)
    
    def test_mean_rel_error(self, validator):
        """Mean relative error should be correct."""
        a = np.array([10.0, 100.0, 1000.0], dtype=np.float32)
        b = np.array([11.0, 105.0, 1010.0], dtype=np.float32)
        # rel errors: 0.1, 0.05, 0.01 -> mean = 0.0533...
        
        metrics = validator._compute_metrics(a, b, "test")
        expected = (0.1 + 0.05 + 0.01) / 3
        assert np.isclose(metrics["mean_rel_error"], expected, rtol=1e-3)
    
    def test_nan_detection(self, validator):
        """Should detect NaN values."""
        a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert metrics["nan_or_inf_count"] >= 1
    
    def test_inf_detection(self, validator):
        """Should detect Inf values."""
        a = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert metrics["nan_or_inf_count"] >= 1
    
    def test_neg_inf_detection(self, validator):
        """Should detect -Inf values."""
        a = np.array([1.0, -np.inf, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        metrics = validator._compute_metrics(a, b, "test")
        assert metrics["nan_or_inf_count"] >= 1


class TestPrecisionAggregation:
    """Test metric aggregation."""
    
    @pytest.fixture
    def validator(self):
        return PrecisionValidator(num_samples=10)
    
    def test_aggregate_metrics_structure(self, validator):
        """Aggregated metrics should have correct structure."""
        all_metrics = {
            "cosine_similarity": [0.99, 0.98, 0.99, 0.97, 0.99],
            "max_abs_error": [0.01, 0.02, 0.01, 0.03, 0.01],
            "mean_rel_error": [0.005, 0.01, 0.005, 0.015, 0.005],
            "nan_or_inf_count": 0,
            "num_samples_tested": 5,
            "per_output_metrics": {
                "output1": {
                    "cosine_similarity": [0.99, 0.98, 0.99, 0.97, 0.99],
                    "max_abs_error": [0.01, 0.02, 0.01, 0.03, 0.01],
                    "mean_rel_error": [0.005, 0.01, 0.005, 0.015, 0.005],
                    "nan_or_inf_count": 0
                }
            }
        }
        
        report = validator._aggregate_metrics(all_metrics)
        
        # Check summary structure
        assert "summary" in report
        summary = report["summary"]
        for metric in ["cosine_similarity", "max_abs_error", "mean_rel_error"]:
            assert metric in summary
            for stat in ["mean", "min", "max", "std"]:
                assert stat in summary[metric]
        
        # Check per_output structure
        assert "per_output" in report
        assert "output1" in report["per_output"]
        
        # Check passed field
        assert "passed" in report
    
    def test_passed_criteria(self, validator):
        """Test pass/fail criteria."""
        # Good metrics - should pass
        good_metrics = {
            "cosine_similarity": [0.999] * 10,
            "max_abs_error": [0.001] * 10,
            "mean_rel_error": [0.001] * 10,
            "nan_or_inf_count": 0,
            "num_samples_tested": 10,
            "per_output_metrics": {}
        }
        report = validator._aggregate_metrics(good_metrics)
        assert report["passed"] is True
        
        # Bad cosine - should fail
        bad_cosine = {
            "cosine_similarity": [0.9] * 10,
            "max_abs_error": [0.001] * 10,
            "mean_rel_error": [0.001] * 10,
            "nan_or_inf_count": 0,
            "num_samples_tested": 10,
            "per_output_metrics": {}
        }
        report = validator._aggregate_metrics(bad_cosine)
        assert report["passed"] is False
        
        # Bad max_abs_error - should fail
        bad_abs = {
            "cosine_similarity": [0.999] * 10,
            "max_abs_error": [0.1] * 10,
            "mean_rel_error": [0.001] * 10,
            "nan_or_inf_count": 0,
            "num_samples_tested": 10,
            "per_output_metrics": {}
        }
        report = validator._aggregate_metrics(bad_abs)
        assert report["passed"] is False
        
        # NaN present - should fail
        bad_nan = {
            "cosine_similarity": [0.999] * 10,
            "max_abs_error": [0.001] * 10,
            "mean_rel_error": [0.001] * 10,
            "nan_or_inf_count": 1,
            "num_samples_tested": 10,
            "per_output_metrics": {}
        }
        report = validator._aggregate_metrics(bad_nan)
        assert report["passed"] is False


class TestPrecisionValidatorIntegration:
    """Integration tests for PrecisionValidator (require ONNX/TensorRT)."""

    def test_installed_but_unavailable_cuda_driver_is_explicit(self, monkeypatch):
        """A broken driver must not break imports or cause an unrelated file error."""
        validator = PrecisionValidator(num_samples=1)
        monkeypatch.setattr(precision_module, "trt", object())
        monkeypatch.setattr(precision_module, "cuda", object())

        def fail_import(_name):
            raise RuntimeError("CUDA driver unavailable")

        monkeypatch.setattr(gpu_runtime.importlib, "import_module", fail_import)
        with pytest.raises(RuntimeError, match="working CUDA driver"):
            validator.validate(Path("unused.onnx"), Path("unused.plan"))

    @pytest.mark.skip(reason="Requires TensorRT runtime")
    def test_validate_identity_model(self):
        """Test precision validation on identity model."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
