"""
Unit tests for Neuro-Symbolic Constraint Verification Engine.
"""

import tempfile
from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.research.symbolic import (
    NeuroSymbolicVerifier,
    SymbolicConstraint,
)


@pytest.fixture
def test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


def test_code_invariant_verification(test_db):
    verifier = NeuroSymbolicVerifier(test_db)

    # 1. Valid code
    valid_code = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    res_valid = verifier.verify_code_invariants(valid_code)
    assert res_valid.status == "verified"
    assert len(res_valid.violations) == 0

    # 2. Infinite loop detection
    infinite_loop_code = """
def spin():
    while True:
        pass
"""
    res_inf = verifier.verify_code_invariants(infinite_loop_code)
    assert res_inf.status == "violated"
    assert any("Termination violation" in v for v in res_inf.violations)

    # 3. Purity check (forbidden eval)
    impure_code = """
def run_dynamic(cmd):
    return eval(cmd)
"""
    res_impure = verifier.verify_code_invariants(impure_code)
    assert res_impure.status == "violated"
    assert any("Purity violation" in v for v in res_impure.violations)

    # 4. Division by literal zero
    div_zero_code = """
def calc(x):
    return x / 0
"""
    res_div = verifier.verify_code_invariants(div_zero_code)
    assert res_div.status == "violated"
    assert any("Division-by-zero" in v for v in res_div.violations)

    # 5. Recursion without base condition
    unbounded_recursion = """
def loop(x):
    return loop(x - 1)
"""
    res_rec = verifier.verify_code_invariants(unbounded_recursion)
    assert res_rec.status == "violated"
    assert any("Recursive function 'loop' has no base condition" in v for v in res_rec.violations)


def test_mathematical_bounds_verification(test_db):
    verifier = NeuroSymbolicVerifier(test_db)

    # 1. Non-negativity satisfied: x**2 over [-5, 5]
    constraints_nonneg = [SymbolicConstraint(constraint_type="non_negativity")]
    res_pos = verifier.verify_mathematical_bounds(
        expression_str="x ** 2",
        variable_ranges={"x": (-5.0, 5.0)},
        constraints=constraints_nonneg,
    )
    assert res_pos.status == "verified"

    # 2. Non-negativity violated: x over [-2, 2]
    res_neg = verifier.verify_mathematical_bounds(
        expression_str="x",
        variable_ranges={"x": (-2.0, 2.0)},
        constraints=constraints_nonneg,
    )
    assert res_neg.status == "violated"
    assert any("Non-negativity violated" in v for v in res_neg.violations)

    # 3. Range bound satisfied: tanh(x) in [-1.0, 1.0]
    constraints_range = [SymbolicConstraint(constraint_type="range_bound", params={"min_val": -1.0, "max_val": 1.0})]
    res_tanh = verifier.verify_mathematical_bounds(
        expression_str="tanh(x)",
        variable_ranges={"x": (-10.0, 10.0)},
        constraints=constraints_range,
    )
    assert res_tanh.status == "verified"

    # 4. Monotonic increasing check: 2 * x + 3
    constraints_mono = [SymbolicConstraint(constraint_type="monotonic_increasing")]
    res_mono = verifier.verify_mathematical_bounds(
        expression_str="2 * x + 3",
        variable_ranges={"x": (-5.0, 5.0)},
        constraints=constraints_mono,
    )
    assert res_mono.status == "verified"


def test_propositional_logic_resolution(test_db):
    verifier = NeuroSymbolicVerifier(test_db)

    # Modus Ponens: P, P -> Q |= Q
    res_mp = verifier.verify_propositional_logic(premises=["P", "P -> Q"], hypothesis="Q")
    assert res_mp.status == "verified"

    # Modus Tollens: ~Q, P -> Q |= ~P
    res_mt = verifier.verify_propositional_logic(premises=["~Q", "P -> Q"], hypothesis="~P")
    assert res_mt.status == "verified"

    # Affirming the Consequent fallacy: Q, P -> Q does NOT entail P
    res_fallacy = verifier.verify_propositional_logic(premises=["Q", "P -> Q"], hypothesis="P")
    assert res_fallacy.status == "inconclusive"


def test_verifications_persistence_and_listing(test_db):
    verifier = NeuroSymbolicVerifier(test_db)

    # Run verification
    res = verifier.verify_code_invariants("x = 10 + 20")
    assert res.id.startswith("sym-code-")

    # Fetch by ID
    fetched = verifier.get_verification(res.id)
    assert fetched is not None
    assert fetched.id == res.id
    assert fetched.status == "verified"

    # List
    history = verifier.list_verifications()
    assert len(history) >= 1
    assert history[0].id == res.id
