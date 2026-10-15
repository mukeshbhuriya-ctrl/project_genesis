"""
Multimodal Environment Transition Modeling Subsystem (Phase 15).

Predicts future environment execution states (filesystem deltas, AST mutations,
stdout/stderr streams, exit codes, and safety invariants) from candidate actions
prior to physical execution.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection

log = logging.getLogger(__name__)

DANGEROUS_PATTERNS = [
    r"rm\s+-rf",
    r"del\s+/[sfq]",
    r"format\s+[a-z]:",
    r":\(\)\s*\{\s*:\|:&\s*\};:",  # fork bomb
    r"dd\s+if=",
    r"mkfs",
    r">\s*/dev/sd",
    r">\s*\\\\\\.\\PhysicalDrive",
    r"shutdown",
    r"reboot",
]


@dataclass
class EnvironmentState:
    """Represents a snapshot of the workspace and execution state."""

    files: dict[str, str] = field(default_factory=dict)  # rel_path -> content
    ast_symbols: dict[str, list[str]] = field(default_factory=dict)  # rel_path -> list of symbols
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    invariants: dict[str, bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentState:
        return cls(
            files=data.get("files", {}),
            ast_symbols=data.get("ast_symbols", {}),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", 0),
            invariants=data.get("invariants", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class Action:
    """A candidate action to be simulated or executed."""

    action_type: str  # 'file_write', 'file_delete', 'ast_mutation', 'command_exec', 'tool_call'
    target: str
    payload: str | dict[str, Any] = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            action_type=data.get("action_type", "tool_call"),
            target=data.get("target", ""),
            payload=data.get("payload", ""),
            params=data.get("params", {}),
        )


@dataclass
class TransitionPrediction:
    """Predicted future environment state and invariant safety analysis."""

    predicted_state: EnvironmentState
    delta_files: list[str] = field(default_factory=list)
    delta_ast: list[str] = field(default_factory=list)
    confidence: float = 1.0
    safety_score: float = 1.0  # 1.0 = completely safe, 0.0 = dangerous
    safety_violations: list[str] = field(default_factory=list)
    predicted_exit_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_state": self.predicted_state.to_dict(),
            "delta_files": self.delta_files,
            "delta_ast": self.delta_ast,
            "confidence": self.confidence,
            "safety_score": self.safety_score,
            "safety_violations": self.safety_violations,
            "predicted_exit_code": self.predicted_exit_code,
        }


@dataclass
class TransitionRecord:
    """Historical audit record comparing transition prediction against ground-truth execution."""

    id: str
    session_id: str
    action_type: str
    predicted_state: dict[str, Any]
    actual_state: dict[str, Any]
    transition_error_score: float
    safety_predicted: bool
    safety_actual: bool
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateTransitionModel:
    """
    World Model Environment Transition Predictor.
    
    Predicts workspace mutations, code AST changes, and execution safety scores
    before physical execution, and measures transition error against real execution.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.db_path = get_db_path_from_url(get_settings().database_url)
        else:
            self.db_path = Path(db_path)

    @staticmethod
    def extract_ast_symbols(code_str: str) -> tuple[list[str], str | None]:
        """Safely parse Python code and extract defined symbols (functions, classes, imports)."""
        symbols: list[str] = []
        syntax_err: str | None = None
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    args = [a.arg for a in node.args.args]
                    symbols.append(f"def {node.name}({', '.join(args)})")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"class {node.name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbols.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    names = [a.name for a in node.names]
                    symbols.append(f"from {mod} import {', '.join(names)}")
        except SyntaxError as e:
            syntax_err = f"SyntaxError at line {e.lineno}: {e.msg}"
        return sorted(list(set(symbols))), syntax_err

    def check_safety(self, action: Action) -> tuple[float, list[str]]:
        """Evaluate action against deterministic security and invariant rules."""
        violations: list[str] = []
        score = 1.0

        # Check path traversal
        if ".." in action.target or action.target.startswith("/") or re.match(r"^[A-Za-z]:", action.target):
            violations.append(f"Potential directory escape detected in target: {action.target}")
            score -= 0.5

        # Check dangerous command patterns
        payload_str = str(action.payload)
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, payload_str, re.IGNORECASE):
                violations.append(f"Dangerous system command pattern detected: {pat}")
                score -= 0.8

        # Check Python syntax if applicable
        if action.target.endswith(".py") and isinstance(action.payload, str):
            _, syntax_err = self.extract_ast_symbols(action.payload)
            if syntax_err:
                violations.append(f"Code syntax violation: {syntax_err}")
                score -= 0.3

        score = max(0.0, min(1.0, score))
        return score, violations

    def predict_transition(
        self, current_state: EnvironmentState, action: Action
    ) -> TransitionPrediction:
        """Simulate and predict future state before running the action."""
        pred_files = dict(current_state.files)
        pred_ast = {k: list(v) for k, v in current_state.ast_symbols.items()}
        delta_files: list[str] = []
        delta_ast: list[str] = []
        pred_stdout = ""
        pred_stderr = ""
        pred_exit_code = 0

        safety_score, violations = self.check_safety(action)

        if action.action_type == "file_write":
            content = str(action.payload)
            pred_files[action.target] = content
            delta_files.append(action.target)

            if action.target.endswith(".py"):
                symbols, syntax_err = self.extract_ast_symbols(content)
                pred_ast[action.target] = symbols
                delta_ast.extend(symbols)
                if syntax_err:
                    pred_stderr = syntax_err
                    pred_exit_code = 1
            pred_stdout = f"Wrote {len(content)} bytes to {action.target}"

        elif action.action_type == "file_delete":
            if action.target in pred_files:
                del pred_files[action.target]
                delta_files.append(f"-{action.target}")
            if action.target in pred_ast:
                del pred_ast[action.target]
                delta_ast.append(f"-ast:{action.target}")
            pred_stdout = f"Deleted {action.target}"

        elif action.action_type == "ast_mutation":
            # Targeted symbol mutation
            existing_code = pred_files.get(action.target, "")
            mutated_code = existing_code + "\n" + str(action.payload)
            pred_files[action.target] = mutated_code
            symbols, syntax_err = self.extract_ast_symbols(mutated_code)
            pred_ast[action.target] = symbols
            delta_ast.extend(symbols)
            if syntax_err:
                pred_stderr = syntax_err
                pred_exit_code = 1
            pred_stdout = f"Mutated AST in {action.target}"

        elif action.action_type == "command_exec":
            cmd = str(action.payload)
            if any(v in cmd for v in ["--help", "-h"]):
                pred_stdout = f"Usage and help options for command: {cmd}"
            elif "pytest" in cmd:
                pred_stdout = "Running pytest suite in simulated world sandbox"
            elif violations:
                pred_stderr = f"Command execution rejected or failed invariants: {violations}"
                pred_exit_code = 126
            else:
                pred_stdout = f"Executed {cmd} successfully"

        elif action.action_type == "tool_call":
            pred_stdout = f"Invoked tool {action.target} with payload"

        confidence = 0.95 if not violations else 0.60

        pred_invariants = dict(current_state.invariants)
        pred_invariants["safety_pass"] = safety_score >= 0.8
        pred_invariants["syntax_pass"] = pred_exit_code == 0

        next_state = EnvironmentState(
            files=pred_files,
            ast_symbols=pred_ast,
            stdout=pred_stdout,
            stderr=pred_stderr,
            exit_code=pred_exit_code,
            invariants=pred_invariants,
            timestamp=time.time(),
        )

        return TransitionPrediction(
            predicted_state=next_state,
            delta_files=delta_files,
            delta_ast=delta_ast,
            confidence=confidence,
            safety_score=safety_score,
            safety_violations=violations,
            predicted_exit_code=pred_exit_code,
        )

    def calculate_transition_error(
        self, predicted: EnvironmentState, actual: EnvironmentState
    ) -> float:
        """
        Compute normalized transition error score [0.0, 1.0].
        
        0.0 indicates perfect agreement between world prediction and actual outcome.
        """
        penalties = 0.0
        total_checks = 4.0

        # Check 1: Exit code match
        if predicted.exit_code != actual.exit_code:
            penalties += 1.0

        # Check 2: Files match
        pred_keys = set(predicted.files.keys())
        actual_keys = set(actual.files.keys())
        if pred_keys != actual_keys:
            penalties += 1.0
        else:
            diff_contents = sum(1 for k in pred_keys if predicted.files[k] != actual.files[k])
            if diff_contents > 0:
                penalties += min(1.0, diff_contents / max(1, len(pred_keys)))

        # Check 3: AST symbols match
        pred_ast_keys = set(predicted.ast_symbols.keys())
        actual_ast_keys = set(actual.ast_symbols.keys())
        if pred_ast_keys != actual_ast_keys:
            penalties += 0.5
        else:
            ast_diff = 0
            for k in pred_ast_keys:
                if set(predicted.ast_symbols[k]) != set(actual.ast_symbols.get(k, [])):
                    ast_diff += 1
            if ast_diff > 0:
                penalties += min(1.0, ast_diff / max(1, len(pred_ast_keys)))

        # Check 4: Invariants
        for inv_key, inv_val in predicted.invariants.items():
            if inv_key in actual.invariants and actual.invariants[inv_key] != inv_val:
                penalties += 0.5

        return min(1.0, round(penalties / total_checks, 4))

    def record_transition(
        self,
        session_id: str,
        action: Action,
        prediction: TransitionPrediction,
        actual_state: EnvironmentState,
        record_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TransitionRecord:
        """Persist transition prediction and actual outcome comparison in SQLite."""
        rid = record_id or f"trans-{uuid.uuid4().hex[:10]}"
        error_score = self.calculate_transition_error(prediction.predicted_state, actual_state)
        safety_pred = prediction.safety_score >= 0.8
        safety_act = actual_state.invariants.get("safety_pass", True)

        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO world_transitions (
                    id, session_id, action_type, predicted_state_json, actual_state_json,
                    transition_error_score, safety_predicted, safety_actual, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    session_id,
                    action.action_type,
                    json.dumps(prediction.predicted_state.to_dict()),
                    json.dumps(actual_state.to_dict()),
                    error_score,
                    1 if safety_pred else 0,
                    1 if safety_act else 0,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

            cursor = conn.execute("SELECT created_at FROM world_transitions WHERE id = ?", (rid,))
            row = cursor.fetchone()
            created_at = row["created_at"] if row else ""

            return TransitionRecord(
                id=rid,
                session_id=session_id,
                action_type=action.action_type,
                predicted_state=prediction.predicted_state.to_dict(),
                actual_state=actual_state.to_dict(),
                transition_error_score=error_score,
                safety_predicted=safety_pred,
                safety_actual=safety_act,
                created_at=created_at,
                metadata=metadata or {},
            )
        finally:
            conn.close()

    def get_transition(self, record_id: str) -> TransitionRecord | None:
        """Fetch a specific transition record by ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, session_id, action_type, predicted_state_json, actual_state_json,
                       transition_error_score, safety_predicted, safety_actual, created_at, metadata
                FROM world_transitions WHERE id = ?
                """,
                (record_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return TransitionRecord(
                id=row["id"],
                session_id=row["session_id"],
                action_type=row["action_type"],
                predicted_state=json.loads(row["predicted_state_json"]),
                actual_state=json.loads(row["actual_state_json"]),
                transition_error_score=float(row["transition_error_score"]),
                safety_predicted=bool(row["safety_predicted"]),
                safety_actual=bool(row["safety_actual"]),
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_transitions(
        self, session_id: str | None = None, limit: int = 50
    ) -> list[TransitionRecord]:
        """List transition records ordered by creation date."""
        conn = get_sync_connection(self.db_path)
        try:
            if session_id:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, action_type, predicted_state_json, actual_state_json,
                           transition_error_score, safety_predicted, safety_actual, created_at, metadata
                    FROM world_transitions WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, action_type, predicted_state_json, actual_state_json,
                           transition_error_score, safety_predicted, safety_actual, created_at, metadata
                    FROM world_transitions ORDER BY created_at DESC LIMIT ?
                    """,
                    (limit,),
                )
            records = []
            for row in cursor.fetchall():
                records.append(
                    TransitionRecord(
                        id=row["id"],
                        session_id=row["session_id"],
                        action_type=row["action_type"],
                        predicted_state=json.loads(row["predicted_state_json"]),
                        actual_state=json.loads(row["actual_state_json"]),
                        transition_error_score=float(row["transition_error_score"]),
                        safety_predicted=bool(row["safety_predicted"]),
                        safety_actual=bool(row["safety_actual"]),
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return records
        finally:
            conn.close()
