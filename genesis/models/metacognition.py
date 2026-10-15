"""
Metacognitive Introspection & Epistemic Confidence Calibration Engine (Phase 16).

Provides autonomous self-monitoring of cognitive load, token perplexity surrogate metrics,
expected calibration error (ECE), and dynamic reasoning budget allocation (fast-path,
reflective self-repair, or multi-agent swarm escalation).
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger

logger = get_logger(__name__)


class MetacognitiveDecision(str, Enum):
    """Reasoning budget allocation based on introspective uncertainty."""
    FAST_PATH = "fast_path"
    REFLECTIVE_REPAIR = "reflective_repair"
    SWARM_ESCALATION = "swarm_escalation"


class MetacognitiveEvaluation(BaseModel):
    """Structured evaluation record of model metacognitive state."""
    id: str = Field(default_factory=lambda: f"metacog-{uuid.uuid4().hex[:10]}")
    prompt: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    epistemic_uncertainty: float = Field(ge=0.0, le=1.0)
    perplexity: float = Field(ge=1.0)
    lexical_complexity: float = Field(ge=0.0, le=1.0)
    decision: MetacognitiveDecision
    calibration_error: float = Field(ge=0.0, le=1.0, default=0.0)
    latency_ms: float = 0.0
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetacognitiveIntrospector:
    """
    Introspective self-monitoring engine calculating calibration metrics,
    cognitive load, and deciding whether to fast-path or escalate reasoning depth.
    """

    HEDGING_KEYWORDS = {
        "maybe", "perhaps", "possibly", "uncertain", "not sure", "unclear",
        "cannot determine", "might be", "doubtful", "tentative", "speculative"
    }

    CONFIDENCE_KEYWORDS = {
        "certainly", "definitely", "verified", "proven", "guaranteed", "confirmed",
        "exact", "invariant held", "mathematically bound", "demonstrated"
    }

    COMPLEXITY_KEYWORDS = {
        "distributed", "consensus", "byzantine", "concurrency", "asynchronous",
        "invariant", "deadlock", "quantum", "differential", "eigenvalue",
        "ast", "recursion", "cryptographic", "merkle", "zero-knowledge", "pareto"
    }

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or get_db_path_from_url(get_settings().database_url))

    @classmethod
    def compute_lexical_complexity(cls, text: str) -> float:
        """
        Compute normalized lexical and structural complexity [0.0, 1.0].
        Considers word length, technical domain keywords, and constraint density.
        """
        if not text.strip():
            return 0.0

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        # Unique token ratio
        unique_ratio = len(set(words)) / len(words)

        # Average token length (normalized around ~8 chars)
        avg_len = sum(len(w) for w in words) / len(words)
        len_score = min(1.0, avg_len / 9.0)

        # Keyword density
        complexity_hits = sum(1 for w in words if w in cls.COMPLEXITY_KEYWORDS)
        kw_density = min(1.0, (complexity_hits * 3.0) / max(len(words), 1))

        # Punctuation/formula density
        symbols = len(re.findall(r"[\{\}\[\]\(\)<>=+\-*/\\|&^%#@!]", text))
        sym_score = min(1.0, (symbols * 2.0) / max(len(words), 1))

        complexity = (0.25 * unique_ratio) + (0.25 * len_score) + (0.35 * kw_density) + (0.15 * sym_score)
        return round(min(1.0, max(0.0, complexity)), 4)

    @classmethod
    def estimate_perplexity(
        cls, text: str, token_log_probs: list[float] | None = None
    ) -> float:
        """
        Estimate perplexity: PPL = exp(-1/N * sum(log p_i)).
        Falls back to lexical entropy heuristic if raw log probs are not provided.
        """
        if token_log_probs and len(token_log_probs) > 0:
            avg_neg_log = -sum(token_log_probs) / len(token_log_probs)
            # Bound exponent to prevent overflow
            avg_neg_log = min(10.0, max(0.0, avg_neg_log))
            return round(math.exp(avg_neg_log), 3)

        # Heuristic estimation from token frequency entropy
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 1.0

        freqs: dict[str, int] = {}
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1

        n = len(words)
        entropy = -sum((c / n) * math.log2(c / n) for c in freqs.values())
        # Map entropy (typically 2 to 6) to a realistic perplexity baseline
        est_ppl = math.pow(2.0, min(6.0, max(1.0, entropy * 0.75)))
        return round(est_ppl, 3)

    @classmethod
    def evaluate_confidence(
        cls,
        prompt: str,
        candidate_output: str | None = None,
        token_log_probs: list[float] | None = None,
    ) -> tuple[float, float]:
        """
        Evaluate epistemic confidence C in [0.0, 1.0] and uncertainty U in [0.0, 1.0].
        """
        combined = f"{prompt} {candidate_output or ''}".lower()
        complexity = cls.compute_lexical_complexity(prompt)

        # Keyword counters
        hedging_count = sum(1 for kw in cls.HEDGING_KEYWORDS if kw in combined)
        confidence_count = sum(1 for kw in cls.CONFIDENCE_KEYWORDS if kw in combined)

        base_confidence = 0.75
        base_confidence -= min(0.40, hedging_count * 0.12)
        base_confidence += min(0.20, confidence_count * 0.08)
        base_confidence -= min(0.25, complexity * 0.25)

        if token_log_probs and len(token_log_probs) > 0:
            ppl = cls.estimate_perplexity("", token_log_probs)
            # Higher perplexity suppresses confidence
            if ppl > 4.0:
                base_confidence -= min(0.30, (ppl - 4.0) * 0.05)
            elif ppl < 1.5:
                base_confidence += 0.15
            elif ppl < 2.5:
                base_confidence += 0.08

        confidence = round(min(0.99, max(0.05, base_confidence)), 4)
        uncertainty = round(min(0.95, max(0.01, 1.0 - confidence)), 4)
        return confidence, uncertainty

    @staticmethod
    def calculate_expected_calibration_error(
        confidences: list[float], accuracies: list[float], n_bins: int = 5
    ) -> float:
        """
        Compute Expected Calibration Error (ECE):
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        if not confidences or len(confidences) != len(accuracies):
            return 0.0

        n = len(confidences)
        bin_size = 1.0 / n_bins
        ece = 0.0

        for b in range(n_bins):
            bin_lower = b * bin_size
            bin_upper = (b + 1) * bin_size

            # Gather samples in bin
            bin_indices = [
                i for i, c in enumerate(confidences)
                if (bin_lower <= c < bin_upper) or (b == n_bins - 1 and bin_lower <= c <= bin_upper)
            ]

            if not bin_indices:
                continue

            bin_conf = sum(confidences[i] for i in bin_indices) / len(bin_indices)
            bin_acc = sum(accuracies[i] for i in bin_indices) / len(bin_indices)
            bin_weight = len(bin_indices) / n

            ece += bin_weight * abs(bin_acc - bin_conf)

        return round(min(1.0, max(0.0, ece)), 4)

    def introspect(
        self,
        prompt: str,
        candidate_output: str | None = None,
        token_log_probs: list[float] | None = None,
        historical_accuracy: float | None = None,
    ) -> MetacognitiveEvaluation:
        """
        Perform complete introspective self-evaluation on input/output pair.
        Persists outcome in metacognitive_evaluations table.
        """
        start_t = time.perf_counter()

        complexity = self.compute_lexical_complexity(prompt)
        ppl = self.estimate_perplexity(candidate_output or prompt, token_log_probs)
        confidence, uncertainty = self.evaluate_confidence(prompt, candidate_output, token_log_probs)

        # Decide reasoning budget
        if confidence >= 0.82 and ppl <= 3.5 and complexity < 0.65:
            decision = MetacognitiveDecision.FAST_PATH
        elif confidence >= 0.55 or (complexity < 0.80 and ppl <= 6.0):
            decision = MetacognitiveDecision.REFLECTIVE_REPAIR
        else:
            decision = MetacognitiveDecision.SWARM_ESCALATION

        # Calibration error estimate against reference accuracy if provided
        acc = historical_accuracy if historical_accuracy is not None else (1.0 if confidence >= 0.60 else 0.0)
        cal_error = round(abs(confidence - acc), 4)
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        record_id = f"metacog-{uuid.uuid4().hex[:10]}"
        evaluation = MetacognitiveEvaluation(
            id=record_id,
            prompt=prompt,
            confidence_score=confidence,
            epistemic_uncertainty=uncertainty,
            perplexity=ppl,
            lexical_complexity=complexity,
            decision=decision,
            calibration_error=cal_error,
            latency_ms=latency_ms,
            metadata={"candidate_output_length": len(candidate_output) if candidate_output else 0},
        )

        # Persist to DB
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO metacognitive_evaluations (
                    id, prompt, confidence_score, epistemic_uncertainty, perplexity,
                    lexical_complexity, decision, calibration_error, latency_ms, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation.id,
                    evaluation.prompt,
                    evaluation.confidence_score,
                    evaluation.epistemic_uncertainty,
                    evaluation.perplexity,
                    evaluation.lexical_complexity,
                    evaluation.decision.value,
                    evaluation.calibration_error,
                    evaluation.latency_ms,
                    json.dumps(evaluation.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return evaluation

    def list_evaluations(self, limit: int = 50) -> list[MetacognitiveEvaluation]:
        """List historical introspective evaluations from database."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, prompt, confidence_score, epistemic_uncertainty, perplexity,
                       lexical_complexity, decision, calibration_error, latency_ms,
                       created_at, metadata
                FROM metacognitive_evaluations
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            )
            evals = []
            for row in cursor.fetchall():
                evals.append(
                    MetacognitiveEvaluation(
                        id=row["id"],
                        prompt=row["prompt"],
                        confidence_score=row["confidence_score"],
                        epistemic_uncertainty=row["epistemic_uncertainty"],
                        perplexity=row["perplexity"],
                        lexical_complexity=row["lexical_complexity"],
                        decision=MetacognitiveDecision(row["decision"]),
                        calibration_error=row["calibration_error"],
                        latency_ms=row["latency_ms"],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return evals
        finally:
            conn.close()

    def get_calibration_summary(self) -> dict[str, Any]:
        """Aggregate summary statistics over historical evaluations."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as total_evals,
                       AVG(confidence_score) as avg_confidence,
                       AVG(epistemic_uncertainty) as avg_uncertainty,
                       AVG(calibration_error) as avg_calibration_error,
                       AVG(perplexity) as avg_ppl,
                       AVG(latency_ms) as avg_latency
                FROM metacognitive_evaluations
                """
            )
            row = cursor.fetchone()
            total = row["total_evals"] if row else 0

            # Count decisions
            dec_cursor = conn.execute(
                """
                SELECT decision, COUNT(*) as count
                FROM metacognitive_evaluations
                GROUP BY decision
                """
            )
            dec_counts = {r["decision"]: r["count"] for r in dec_cursor.fetchall()}

            return {
                "total_evaluations": total,
                "avg_confidence": round(row["avg_confidence"] or 0.0, 4) if row else 0.0,
                "avg_uncertainty": round(row["avg_uncertainty"] or 0.0, 4) if row else 0.0,
                "avg_calibration_error": round(row["avg_calibration_error"] or 0.0, 4) if row else 0.0,
                "avg_perplexity": round(row["avg_ppl"] or 1.0, 2) if row else 1.0,
                "avg_latency_ms": round(row["avg_latency"] or 0.0, 2) if row else 0.0,
                "decisions": dec_counts,
            }
        finally:
            conn.close()
