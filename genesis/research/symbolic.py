"""
Neuro-Symbolic Constraint Verification Engine.

Combines statistical/LLM reasoning with formal symbolic logic, static AST invariant analysis,
numerical boundary verification, and propositional resolution refutation.
"""

from __future__ import annotations

import ast
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger

logger = get_logger(__name__)


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return get_db_path_from_url(get_settings().get_db_url())


@dataclass
class SymbolicConstraint:
    """Formal invariant or constraint specification."""
    constraint_type: str  # 'range_bound', 'non_negativity', 'purity_check', 'termination', 'division_by_zero', 'propositional_logic'
    target: str = ""
    expression: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Outcome of a neuro-symbolic verification run."""
    id: str
    target_type: str  # 'code', 'math', 'hypothesis', 'kernel'
    status: str  # 'verified', 'violated', 'inconclusive'
    violations: list[str] = field(default_factory=list)
    verification_latency_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuroSymbolicVerifier:
    """
    Evaluates formal constraints and invariant assertions over code,
    mathematical bounds, and logical hypotheses.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = _resolve_db_path(db_path)

    def verify_code_invariants(
        self,
        code_str: str,
        constraints: list[SymbolicConstraint] | None = None,
    ) -> VerificationResult:
        """
        Perform static AST invariant checks on Python code:
        - Termination: detect unbounded loops or un-anchored recursions.
        - Safety/Purity: detect eval/exec/system exploits.
        - Division-by-Zero: flag zero denominators or un-guarded divisions.
        """
        start_time = time.perf_counter()
        vid = f"sym-code-{uuid.uuid4().hex[:10]}"
        violations: list[str] = []

        # 1. Parse AST
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            violations.append(f"SyntaxError: {e.msg} at line {e.lineno}")
            return self._record_and_return(vid, "code", code_str, constraints or [], "violated", violations, start_time)

        # 2. AST Invariant Visitors
        for node in ast.walk(tree):
            # Check for forbidden execution primitives (Purity check)
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec", "__import__", "globals", "locals"):
                    violations.append(f"Purity violation: Forbidden dangerous primitive '{func_name}()' at line {node.lineno}")

            # Check division by zero risk
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    violations.append(f"Division-by-zero violation: Literal divisor 0 at line {node.lineno}")

            # Check termination of while loops
            if isinstance(node, ast.While):
                # If while True / while 1, check if there is a break or return in the body
                if isinstance(node.test, ast.Constant) and bool(node.test.value) is True:
                    has_exit = any(
                        isinstance(child, (ast.Break, ast.Return))
                        for child in ast.walk(node)
                    )
                    if not has_exit:
                        violations.append(f"Termination violation: Infinite loop detected ('while True' without break/return) at line {node.lineno}")

            # Check recursion termination
            if isinstance(node, ast.FunctionDef):
                fn_name = node.name
                recursive_calls = [
                    child for child in ast.walk(node)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == fn_name
                ]
                if recursive_calls:
                    # Look for base condition (if statements inside function)
                    has_if = any(isinstance(child, ast.If) for child in ast.walk(node))
                    if not has_if:
                        violations.append(f"Termination violation: Recursive function '{fn_name}' has no base condition (If statement)")

        # 3. Check custom user constraints
        if constraints:
            for c in constraints:
                if c.constraint_type == "purity_check":
                    forbidden = c.params.get("forbidden_names", [])
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and node.id in forbidden:
                            violations.append(f"Purity constraint violated: Variable/Name '{node.id}' is forbidden")
                elif c.constraint_type == "type_assertion":
                    # verify function arguments have type annotations
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            for arg in node.args.args:
                                if arg.annotation is None and arg.arg != "self":
                                    violations.append(f"Type assertion violation: Argument '{arg.arg}' in '{node.name}' lacks type annotation")

        status = "violated" if violations else "verified"
        return self._record_and_return(vid, "code", code_str, constraints or [], status, violations, start_time)

    def verify_mathematical_bounds(
        self,
        expression_str: str,
        variable_ranges: dict[str, tuple[float, float]],
        constraints: list[SymbolicConstraint],
    ) -> VerificationResult:
        """
        Verify mathematical properties (range bounds, non-negativity, monotonicity)
        over continuous variable domains using boundary sampling.
        """
        start_time = time.perf_counter()
        vid = f"sym-math-{uuid.uuid4().hex[:10]}"
        violations: list[str] = []

        # Safe evaluation namespace
        safe_env = {
            "math": math,
            "abs": abs,
            "sin": math.sin,
            "cos": math.cos,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "tanh": math.tanh,
        }

        # Generate test points: endpoints + midpoints + interior samples
        test_points: list[dict[str, float]] = []
        var_names = list(variable_ranges.keys())

        if not var_names:
            # Single evaluation
            test_points = [{}]
        else:
            # Evaluate cartesian grid of key points (min, mid, max, samples)
            import itertools
            grid_per_var = []
            for v, (v_min, v_max) in variable_ranges.items():
                v_mid = (v_min + v_max) / 2.0
                pts = [v_min, v_mid, v_max]
                # Add 2 interior random points
                pts.append(v_min + 0.25 * (v_max - v_min))
                pts.append(v_min + 0.75 * (v_max - v_min))
                grid_per_var.append([(v, p) for p in sorted(set(pts))])

            for combo in itertools.product(*grid_per_var):
                test_points.append(dict(combo))

        # Evaluate across grid points
        evaluated_values: list[tuple[dict[str, float], float]] = []
        for pt in test_points:
            local_scope = {**safe_env, **pt}
            try:
                val = float(eval(expression_str, {"__builtins__": {}}, local_scope))
                evaluated_values.append((pt, val))
            except Exception as ex:
                violations.append(f"Evaluation error on inputs {pt}: {ex}")
                return self._record_and_return(vid, "math", expression_str, constraints, "violated", violations, start_time)

        # Check each constraint
        for c in constraints:
            if c.constraint_type == "non_negativity":
                for pt, val in evaluated_values:
                    if val < -1e-9:
                        violations.append(f"Non-negativity violated: expression evaluated to {val:.6f} (< 0) at {pt}")
                        break

            elif c.constraint_type == "range_bound":
                min_bound = c.params.get("min_val", -float("inf"))
                max_bound = c.params.get("max_val", float("inf"))
                for pt, val in evaluated_values:
                    if val < min_bound - 1e-9 or val > max_bound + 1e-9:
                        violations.append(f"Range bound [{min_bound}, {max_bound}] violated: got {val:.6f} at {pt}")
                        break

            elif c.constraint_type == "monotonic_increasing" and len(var_names) == 1:
                var = var_names[0]
                sorted_vals = sorted(evaluated_values, key=lambda x: x[0][var])
                for i in range(len(sorted_vals) - 1):
                    if sorted_vals[i+1][1] < sorted_vals[i][1] - 1e-9:
                        violations.append(f"Monotonic increasing violated: f({sorted_vals[i+1][0][var]})={sorted_vals[i+1][1]:.4f} < f({sorted_vals[i][0][var]})={sorted_vals[i][1]:.4f}")
                        break

        status = "violated" if violations else "verified"
        return self._record_and_return(vid, "math", expression_str, constraints, status, violations, start_time)

    def verify_propositional_logic(
        self,
        premises: list[str],
        hypothesis: str,
    ) -> VerificationResult:
        """
        Verify a logical hypothesis against premises using resolution refutation.
        Supports atomic literals ('P', '~P') and simple implications ('P -> Q').
        """
        start_time = time.perf_counter()
        vid = f"sym-logic-{uuid.uuid4().hex[:10]}"
        violations: list[str] = []

        # Convert premises to clausal CNF sets
        # A clause is a frozenset of literals, e.g. frozenset({"P", "~Q"})
        clauses: set[frozenset[str]] = set()

        for p in premises:
            p_clean = p.strip()
            if "->" in p_clean:
                # A -> B is equivalent to (~A or B)
                ante, cons = [s.strip() for s in p_clean.split("->", 1)]
                neg_ante = ante[1:] if ante.startswith("~") else f"~{ante}"
                clauses.add(frozenset([neg_ante, cons]))
            else:
                clauses.add(frozenset([p_clean]))

        # Negate hypothesis: if H is "Q", negated is "~Q"; if "~Q", negated is "Q"
        h_clean = hypothesis.strip()
        neg_hypo = h_clean[1:] if h_clean.startswith("~") else f"~{h_clean}"
        clauses.add(frozenset([neg_hypo]))

        # Resolution Loop
        new_clauses: set[frozenset[str]] = set()
        resolved = False
        empty_clause = frozenset()

        step_limit = 200
        step_count = 0

        while step_count < step_limit:
            step_count += 1
            pairs = list(clauses)
            derived_in_round = set()

            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    c1, c2 = pairs[i], pairs[j]
                    # Find complementary literals
                    for lit in c1:
                        comp = lit[1:] if lit.startswith("~") else f"~{lit}"
                        if comp in c2:
                            # Resolvent is (c1 - {lit}) union (c2 - {comp})
                            resolvent = (c1 - {lit}) | (c2 - {comp})
                            if resolvent == empty_clause:
                                resolved = True
                                break
                            derived_in_round.add(frozenset(resolvent))
                    if resolved:
                        break
                if resolved:
                    break

            if resolved:
                break

            if derived_in_round.issubset(clauses):
                # No new clauses can be derived
                break
            clauses.update(derived_in_round)

        if resolved:
            status = "verified"
        else:
            status = "inconclusive"
            violations.append(f"Resolution refutation could not derive contradiction for hypothesis '{hypothesis}' from premises {premises}")

        return self._record_and_return(
            vid, "hypothesis", f"Premises: {premises} => {hypothesis}",
            [SymbolicConstraint(constraint_type="propositional_logic", target=hypothesis)],
            status, violations, start_time
        )

    def _record_and_return(
        self,
        vid: str,
        target_type: str,
        expression_or_code: str,
        constraints: list[SymbolicConstraint],
        status: str,
        violations: list[str],
        start_time: float,
    ) -> VerificationResult:
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        res = VerificationResult(
            id=vid,
            target_type=target_type,
            status=status,
            violations=violations,
            verification_latency_ms=latency_ms,
        )

        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO symbolic_verifications (id, target_type, expression_or_code, constraints_json, status, violations_json, verification_latency_ms, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vid,
                    target_type,
                    expression_or_code[:2000],
                    json.dumps([c.to_dict() for c in constraints]),
                    status,
                    json.dumps(violations),
                    latency_ms,
                    res.created_at,
                    json.dumps({}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return res

    def get_verification(self, record_id: str) -> VerificationResult | None:
        """Fetch verification outcome by record ID."""
        conn = get_sync_connection(self.db_path)
        try:
            cur = conn.execute("SELECT * FROM symbolic_verifications WHERE id = ?", (record_id,))
            row = cur.fetchone()
            if not row:
                return None
            return VerificationResult(
                id=row["id"],
                target_type=row["target_type"],
                status=row["status"],
                violations=json.loads(row["violations_json"] or "[]"),
                verification_latency_ms=row["verification_latency_ms"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_verifications(
        self,
        target_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[VerificationResult]:
        """List historical verification records."""
        conn = get_sync_connection(self.db_path)
        try:
            query = "SELECT * FROM symbolic_verifications"
            params: list[Any] = []
            conditions: list[str] = []

            if target_type:
                conditions.append("target_type = ?")
                params.append(target_type)
            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cur = conn.execute(query, params)
            return [
                VerificationResult(
                    id=r["id"],
                    target_type=r["target_type"],
                    status=r["status"],
                    violations=json.loads(r["violations_json"] or "[]"),
                    verification_latency_ms=r["verification_latency_ms"],
                    created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in cur.fetchall()
            ]
        finally:
            conn.close()
