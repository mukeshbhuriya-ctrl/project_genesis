"""
Tests for Environment Transition Modeling Subsystem (Phase 15).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.world_model.transition import (
    Action,
    EnvironmentState,
    StateTransitionModel,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "trans_test.db"
    initialize_database(db_file)
    return db_file


def test_extract_ast_symbols():
    """Verify safe AST extraction identifies functions, classes, and imports."""
    code = (
        "import os\n"
        "from typing import Any\n\n"
        "class Optimizer:\n"
        "    def step(self, lr: float):\n"
        "        pass\n"
    )
    symbols, err = StateTransitionModel.extract_ast_symbols(code)
    assert err is None
    assert "import os" in symbols
    assert "from typing import Any" in symbols
    assert "class Optimizer" in symbols
    assert "def step(self, lr)" in symbols


def test_extract_ast_symbols_syntax_error():
    """Verify syntax error is flagged gracefully without crashing."""
    code = "def broken(:"
    symbols, err = StateTransitionModel.extract_ast_symbols(code)
    assert err is not None
    assert "SyntaxError" in err
    assert len(symbols) == 0


def test_check_safety_invariants(test_db: Path):
    """Verify dangerous commands and path escapes trigger safety score degradation."""
    model = StateTransitionModel(db_path=test_db)

    # 1. Path escape
    act1 = Action(action_type="file_write", target="../secrets.txt", payload="data")
    score1, viols1 = model.check_safety(act1)
    assert score1 < 1.0
    assert any("escape" in v.lower() for v in viols1)

    # 2. Dangerous command
    act2 = Action(action_type="command_exec", target="shell", payload="rm -rf /")
    score2, viols2 = model.check_safety(act2)
    assert score2 <= 0.2
    assert any("dangerous" in v.lower() for v in viols2)

    # 3. Safe action
    act3 = Action(action_type="file_write", target="models/layer.py", payload="x = 10\n")
    score3, viols3 = model.check_safety(act3)
    assert score3 == 1.0
    assert len(viols3) == 0


def test_predict_transition_file_write(test_db: Path):
    """Verify file_write state prediction correctly simulates files and AST deltas."""
    model = StateTransitionModel(db_path=test_db)
    curr = EnvironmentState()

    code = "def forward(x):\n    return x * 2\n"
    act = Action(action_type="file_write", target="model.py", payload=code)

    pred = model.predict_transition(curr, act)

    assert "model.py" in pred.predicted_state.files
    assert pred.predicted_state.files["model.py"] == code
    assert "model.py" in pred.predicted_state.ast_symbols
    assert "def forward(x)" in pred.predicted_state.ast_symbols["model.py"]
    assert "model.py" in pred.delta_files
    assert pred.safety_score == 1.0
    assert pred.predicted_exit_code == 0


def test_predict_transition_file_delete(test_db: Path):
    """Verify file_delete removes file from simulated state."""
    model = StateTransitionModel(db_path=test_db)
    curr = EnvironmentState(
        files={"temp.txt": "scaffolding"},
        ast_symbols={"temp.txt": []},
    )

    act = Action(action_type="file_delete", target="temp.txt")
    pred = model.predict_transition(curr, act)

    assert "temp.txt" not in pred.predicted_state.files
    assert "-temp.txt" in pred.delta_files


def test_transition_error_and_persistence(test_db: Path):
    """Verify calculating transition error and persisting records in SQLite."""
    model = StateTransitionModel(db_path=test_db)
    curr = EnvironmentState()

    act = Action(action_type="file_write", target="math_utils.py", payload="def add(a, b):\n    return a + b\n")
    pred = model.predict_transition(curr, act)

    # Actual state matches perfectly
    actual_state = EnvironmentState(
        files={"math_utils.py": "def add(a, b):\n    return a + b\n"},
        ast_symbols={"math_utils.py": ["def add(a, b)"]},
        stdout="Wrote 35 bytes to math_utils.py",
        exit_code=0,
        invariants={"safety_pass": True, "syntax_pass": True},
    )

    rec = model.record_transition(
        session_id="sess-001",
        action=act,
        prediction=pred,
        actual_state=actual_state,
        record_id="trans-rec-001",
    )

    assert rec.id == "trans-rec-001"
    assert rec.transition_error_score == 0.0
    assert rec.safety_predicted is True

    # Retrieve from DB
    fetched = model.get_transition("trans-rec-001")
    assert fetched is not None
    assert fetched.session_id == "sess-001"

    listed = model.list_transitions(session_id="sess-001")
    assert len(listed) == 1
