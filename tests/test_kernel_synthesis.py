"""
Unit tests for Autonomous LLM-Driven Kernel Synthesis & Verification (Phase 12).
"""

import pytest
from pathlib import Path

from genesis.core.database import initialize_database
from genesis.research.kernel_synthesis import (
    KernelSynthesisEngine,
    KernelSynthesisExperiment,
    ref_dot_product,
    ref_gelu,
    ref_quantize_int8,
    ref_rmsnorm,
    ref_swiglu,
)


def test_reference_math_accuracy():
    """Verify ground-truth reference tensor math implementations."""
    # GELU
    assert pytest.approx(ref_gelu([0.0])[0], 0.001) == 0.0
    assert ref_gelu([1.0])[0] > 0.8
    assert ref_gelu([-3.0])[0] < 0.0

    # RMSNorm
    x = [2.0, 2.0, 2.0, 2.0]
    w = [1.0, 1.0, 1.0, 1.0]
    res_rms = ref_rmsnorm(x, w, eps=0.0)
    for v in res_rms:
        assert pytest.approx(v, 0.001) == 1.0

    # SwiGLU
    res_swi = ref_swiglu([0.0], [2.0])
    # 0 * sigmoid(0) * 2 = 0
    assert pytest.approx(res_swi[0], 0.001) == 0.0

    # Quantize int8
    res_q = ref_quantize_int8([-1.0, 0.0, 1.0], 50.0)
    assert res_q == [-50, 0, 50]

    # Dot product
    assert pytest.approx(ref_dot_product([1.0, 2.0], [3.0, 4.0]), 0.001) == 11.0


def test_kernel_synthesis_loop_execution(tmp_path):
    """Verify autonomous kernel synthesis loop compiles, benchmarks, and records experiment."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    engine = KernelSynthesisEngine(db_path=test_db)

    # Run loop for dot_product
    exp_dot = engine.run_synthesis_loop(
        kernel_name="dot_product",
        operation_type="reduction",
        max_iterations=2,
        tolerance=1e-3,
    )

    assert isinstance(exp_dot, KernelSynthesisExperiment)
    assert exp_dot.kernel_name == "dot_product"
    assert exp_dot.status == "success"
    assert exp_dot.passed_numerical_check is True
    assert exp_dot.speedup_factor > 0.0
    assert exp_dot.max_abs_error <= 1e-3

    # Run loop for gelu
    exp_gelu = engine.run_synthesis_loop(
        kernel_name="gelu",
        operation_type="activation",
        max_iterations=2,
        tolerance=1e-3,
    )
    assert exp_gelu.status == "success"
    assert exp_gelu.passed_numerical_check is True

    # Check persistence in SQLite
    experiments = engine.list_experiments()
    assert len(experiments) >= 2
    names = [e.kernel_name for e in experiments]
    assert "dot_product" in names
    assert "gelu" in names


def test_kernel_synthesis_reflective_repair(tmp_path):
    """Verify engine invokes reflective repair when C code has missing includes or syntax issues."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    engine = KernelSynthesisEngine(db_path=test_db)

    # Provide initial code without standard headers
    broken_source = """
EXPORT float dot_product(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
"""
    repaired = engine._repair_source(broken_source, "implicit declaration of function")
    assert "#include <stddef.h>" in repaired
    assert "#include <math.h>" in repaired

    # Run synthesis with repaired code
    exp = engine.run_synthesis_loop(
        kernel_name="dot_product",
        operation_type="reduction",
        initial_c_source=repaired,
        max_iterations=2,
    )
    assert exp.status == "success"
    assert exp.passed_numerical_check is True
