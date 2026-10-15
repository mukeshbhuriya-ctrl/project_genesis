"""
Tests for Hardware-Targeted Micro-Quantization Engine (Phase 14).
"""

import math
import pytest
from pathlib import Path

from genesis.core.database import initialize_database
from genesis.models.quantization import (
    MicroQuantizer,
    QuantizationConfig,
    QuantizationResult,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "quant_test.db"
    initialize_database(db_file)
    return db_file


def test_symmetric_quantization():
    """Verify symmetric linear uniform quantization math and dequantization reconstruction."""
    weights = [-1.0, -0.5, 0.0, 0.25, 0.75, 1.0]
    q, scale, deq = MicroQuantizer.quantize_symmetric(weights, bits=4)

    assert len(q) == len(weights)
    assert len(deq) == len(weights)
    q_max = (1 << (4 - 1)) - 1  # 7
    assert all(-q_max <= val <= q_max for val in q)
    # Check max deviation bounded by half scale step
    max_err = max(abs(w - d) for w, d in zip(weights, deq))
    assert max_err <= scale * 0.6


def test_asymmetric_quantization():
    """Verify asymmetric affine quantization with zero-point offset."""
    weights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5]
    q, scale, zero_pt, deq = MicroQuantizer.quantize_asymmetric(weights, bits=4)

    assert len(q) == len(weights)
    q_max = (1 << 4) - 1  # 15
    assert all(0 <= val <= q_max for val in q)
    assert 0 <= zero_pt <= q_max
    max_err = max(abs(w - d) for w, d in zip(weights, deq))
    assert max_err <= scale * 0.6


def test_blocked_quantization():
    """Verify blocked quantization partitions and processes sub-chunks."""
    quantizer = MicroQuantizer(db_path=":memory:")
    weights = [float(i) * 0.01 for i in range(128)]

    q, scales, deq = quantizer.quantize_blocked(weights, bits=4, block_size=32)
    assert len(q) == 128
    assert len(scales) == 4  # 128 / 32 = 4 blocks
    assert len(deq) == 128


def test_awq_salient_outlier_protection():
    """Verify salient activation channels are preserved without quantization error."""
    quantizer = MicroQuantizer(db_path=":memory:")

    # Create synthetic matrix: 8 rows x 16 columns
    # Column 3 and Column 7 have large outlier activations
    matrix = [[0.05 * ((r + c) % 5) for c in range(16)] for r in range(8)]
    for r in range(8):
        matrix[r][3] = 12.5 + r * 0.1  # Outlier
        matrix[r][7] = -15.0 - r * 0.1  # Outlier

    res = quantizer.quantize_awq_salient(
        matrix, bits=4, block_size=32, outlier_ratio=0.15
    )

    assert "outlier_channels" in res
    assert 3 in res["outlier_channels"]
    assert 7 in res["outlier_channels"]
    assert res["cosine_similarity"] > 0.98

    # Verify outlier columns are preserved exactly in FP16
    rec = res["reconstructed"]
    for r in range(8):
        assert rec[r][3] == matrix[r][3]
        assert rec[r][7] == matrix[r][7]


def test_profile_quantization_and_persistence(test_db: Path):
    """Verify end-to-end profiling, compression metrics, and SQLite persistence."""
    quantizer = MicroQuantizer(db_path=test_db)

    cfg = QuantizationConfig(
        model_name="qwen2.5:7b",
        bits=4,
        strategy="awq_outlier_salient",
        block_size=64,
        outlier_channel_ratio=0.01,
    )

    result = quantizer.profile_quantization(cfg)
    assert isinstance(result, QuantizationResult)
    assert result.model_name == "qwen2.5:7b"
    assert result.bits == 4
    assert result.compression_ratio > 3.0
    assert result.quantized_size_mb < result.original_size_mb
    assert result.cosine_similarity > 0.95

    # List profiles from SQLite
    profiles = quantizer.list_profiles()
    assert len(profiles) == 1
    assert profiles[0].id == result.id
    assert profiles[0].bits == 4

    # Get single profile
    fetched = quantizer.get_profile(result.id)
    assert fetched is not None
    assert fetched.id == result.id
    assert fetched.strategy == "awq_outlier_salient"


def test_quantization_config_validation():
    """Verify validation of bit-widths and block sizes."""
    with pytest.raises(ValueError, match="Unsupported bits"):
        QuantizationConfig(bits=5)

    with pytest.raises(ValueError, match="Unsupported block_size"):
        QuantizationConfig(block_size=50)
