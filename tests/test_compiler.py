"""
Unit tests for Real-Time Self-Compilation & Native Binary Extensions (Phase 11).
"""

import sqlite3
import pytest
from pathlib import Path

from genesis.core.compiler import (
    CompilationResult,
    NativeExtensionManager,
    C_DOT_PRODUCT_SOURCE,
    C_QUANTIZE_INT8_SOURCE,
    _py_dot_product,
    _py_quantize_int8,
    _py_matrix_vector_mul,
)


def test_pure_python_kernel_math():
    """Verify mathematical correctness of the built-in pure Python kernel fallbacks."""
    # Dot product
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    res_dot = _py_dot_product(a, b)
    assert pytest.approx(res_dot, 0.001) == 32.0  # 4 + 10 + 18 = 32

    # Quantize int8
    vals = [-1.0, 0.0, 1.0, 2.0]
    scale = 10.0
    res_q = _py_quantize_int8(vals, scale)
    assert res_q == [-10, 0, 10, 20]

    # Matrix-vector multiply
    mat = [[1.0, 2.0], [3.0, 4.0]]
    vec = [2.0, 3.0]
    res_mv = _py_matrix_vector_mul(mat, vec)
    # [1*2 + 2*3, 3*2 + 4*3] = [8, 18]
    assert pytest.approx(res_mv[0], 0.001) == 8.0
    assert pytest.approx(res_mv[1], 0.001) == 18.0


def test_compiler_detection(tmp_path):
    """Test host compiler detection routine."""
    test_db = tmp_path / "test_genesis.db"
    mgr = NativeExtensionManager(db_path=test_db)
    compilers = mgr.detect_compilers()
    assert isinstance(compilers, dict)
    assert isinstance(mgr.has_native_compiler(), bool)


def test_compile_fallback_and_persistence(tmp_path):
    """Test compiling C extension gracefully registers Python fallback when compiler is absent."""
    test_db = tmp_path / "test_genesis.db"
    from genesis.core.database import initialize_database
    initialize_database(test_db)

    mgr = NativeExtensionManager(db_path=test_db)

    result = mgr.compile_c_extension(
        name="vector_dot_product",
        source_code=C_DOT_PRODUCT_SOURCE,
        fallback_func=_py_dot_product,
    )

    assert isinstance(result, CompilationResult)
    assert result.name == "vector_dot_product"
    assert result.status in ("compiled", "fallback_python")
    assert result.id.startswith("ext-")

    # Verify invocation through manager
    val = mgr.invoke("vector_dot_product", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert pytest.approx(val, 0.001) == 32.0

    # Verify persistence in database
    extensions = mgr.list_extensions()
    assert len(extensions) >= 1
    found = [e for e in extensions if e.name == "vector_dot_product"]
    assert len(found) == 1
    assert found[0].name == "vector_dot_product"


def test_benchmark_kernel(tmp_path):
    """Test benchmarking a registered kernel."""
    test_db = tmp_path / "test_genesis.db"
    from genesis.core.database import initialize_database
    initialize_database(test_db)

    mgr = NativeExtensionManager(db_path=test_db)
    mgr.compile_c_extension("vector_dot_product", C_DOT_PRODUCT_SOURCE, fallback_func=_py_dot_product)

    metrics = mgr.benchmark_kernel("vector_dot_product", iterations=100)
    assert metrics["kernel"] == "vector_dot_product"
    assert metrics["iterations"] == 100
    assert metrics["baseline_ms"] >= 0.0
    assert metrics["active_ms"] >= 0.0
    assert metrics["speedup_factor"] > 0.0

    # Non-existent kernel raises KeyError
    with pytest.raises(KeyError):
        mgr.benchmark_kernel("non_existent_kernel", iterations=50)


def test_invoke_unknown_kernel_raises_keyerror(tmp_path):
    """Test invoking an unregistered kernel raises KeyError."""
    test_db = tmp_path / "test_genesis.db"
    from genesis.core.database import initialize_database
    initialize_database(test_db)

    mgr = NativeExtensionManager(db_path=test_db)
    with pytest.raises(KeyError):
        mgr.invoke("unknown_magic_kernel", 1, 2, 3)
