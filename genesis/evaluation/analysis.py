"""
Automated failure analysis and error categorization for evaluation runs.
"""

from __future__ import annotations

import re
from typing import Any

from genesis.evaluation.models import EvalCase, EvalResult, EvalType, FailureCategory


class FailureAnalyzer:
    """Classifies model evaluation failures into an actionable error taxonomy."""

    @staticmethod
    def classify_failure(
        case: EvalCase,
        output: str,
        error: str | None = None,
    ) -> FailureCategory:
        """Categorize why a model response failed an evaluation test case."""
        if error:
            err_lower = error.lower()
            if "timeout" in err_lower or "timed out" in err_lower:
                return FailureCategory.TIMEOUT
            if "syntaxerror" in err_lower or "indentationerror" in err_lower:
                return FailureCategory.SYNTAX_ERROR
            if "permission" in err_lower or "securityerror" in err_lower or "not found in registry" in err_lower:
                return FailureCategory.TOOL_MISUSE

        clean_output = output.strip().lower() if output else ""

        if not clean_output:
            return FailureCategory.FORMAT_ERROR

        if case.eval_type == EvalType.TOOL_CALL:
            # Check if model attempted to call a tool or produced raw text instead
            if not any(token in clean_output for token in ["tool_call", "function", "{", "call"]):
                return FailureCategory.TOOL_MISUSE
            return FailureCategory.TOOL_MISUSE

        if case.eval_type in (EvalType.EXACT_MATCH, EvalType.CONTAINS):
            # Check if output is formatted poorly (e.g. refused to answer or verbose boilerplate)
            if any(phrase in clean_output for phrase in ["i cannot", "as an ai", "apologize", "unable to"]):
                return FailureCategory.HALLUCINATION

            # Check if response contains syntax errors in generated code
            if "syntaxerror" in clean_output:
                return FailureCategory.SYNTAX_ERROR

            return FailureCategory.INCORRECT_ANSWER

        if case.eval_type == EvalType.NUMERIC_TOLERANCE:
            # Check if numeric characters exist
            if not re.search(r"[-+]?\d*\.?\d+", clean_output):
                return FailureCategory.FORMAT_ERROR
            return FailureCategory.INCORRECT_ANSWER

        return FailureCategory.INCORRECT_ANSWER

    @staticmethod
    def summarize_failures(results: list[EvalResult]) -> dict[str, Any]:
        """Compute aggregate statistics and distribution of failure categories."""
        failed_results = [r for r in results if not r.passed]
        total_failures = len(failed_results)

        if total_failures == 0:
            return {
                "total_failures": 0,
                "distribution": {},
                "dominant_failure": "none",
                "recommendation": "All test cases passed. Model meets performance criteria.",
            }

        distribution: dict[str, int] = {}
        for r in failed_results:
            cat_name = r.failure_category.value
            distribution[cat_name] = distribution.get(cat_name, 0) + 1

        dominant_cat = max(distribution.items(), key=lambda item: item[1])[0]

        # Actionable recommendations based on failure modes
        recommendations = {
            FailureCategory.TOOL_MISUSE.value: "Improve tool-calling system prompts or fine-tune tool parameter formatting.",
            FailureCategory.TIMEOUT.value: "Reduce output token limits, check GPU offloading layers, or optimize model prompt length.",
            FailureCategory.SYNTAX_ERROR.value: "Strengthen coding prompt constraints or add syntax verification linter steps.",
            FailureCategory.FORMAT_ERROR.value: "Enforce JSON mode / structured schema outputs in generation requests.",
            FailureCategory.INCORRECT_ANSWER.value: "Augment memory context or benchmark against fine-tuned model checkpoint.",
            FailureCategory.HALLUCINATION.value: "Increase hybrid retrieval weighting or ground prompt with relevant knowledge chunks.",
        }

        return {
            "total_failures": total_failures,
            "distribution": distribution,
            "distribution_percentages": {
                k: round((v / total_failures) * 100, 1) for k, v in distribution.items()
            },
            "dominant_failure": dominant_cat,
            "recommendation": recommendations.get(dominant_cat, "Investigate failure cases individually."),
        }
