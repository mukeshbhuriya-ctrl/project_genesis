"""
Evaluation reporting engine producing Markdown, JSON, and summary tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.evaluation.models import EvalRun
from genesis.evaluation.regression import RegressionComparison


class EvaluationReporter:
    """
    Renders structured evaluation reports from benchmark runs and regression comparisons.
    """

    @staticmethod
    def generate_run_markdown(run: EvalRun, benchmark_name: str | None = None) -> str:
        """
        Generate a detailed Markdown report for a single evaluation run.
        """
        bm_title = benchmark_name or run.benchmark_id
        fail_summary = run.metadata.get("failure_summary", {}) if run.metadata else {}
        dominant_fail = fail_summary.get("dominant_failure", "none")
        recommendation = fail_summary.get("recommendation", "N/A")

        status_badge = "✅ PASSED" if run.failed_cases == 0 else "⚠️ DEFECTS DETECTED"

        md = [
            f"# Evaluation Report: {bm_title}",
            "",
            f"**Run ID:** `{run.id}`  ",
            f"**Target Model:** `{run.model_id}`  ",
            f"**Status:** {run.status.value.upper()} ({status_badge})  ",
            f"**Executed At:** {run.started_at}  ",
            f"**Total Duration:** {run.duration_ms / 1000:.2f}s  ",
            "",
            "## Executive Summary",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Overall Score** | **{run.score:.2f}%** |",
            f"| **Total Cases** | {run.total_cases} |",
            f"| **Passed Cases** | {run.passed_cases} |",
            f"| **Failed Cases** | {run.failed_cases} |",
            f"| **Avg Case Latency** | {(run.duration_ms / max(1, run.total_cases)):.1f} ms |",
            f"| **Primary Failure Mode** | `{dominant_fail}` |",
            "",
            f"> **Engineering Recommendation:** {recommendation}",
            "",
            "## Case-by-Case Breakdown",
            "",
            "| Case ID | Status | Score | Latency (ms) | Failure Mode | Error / Notes |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for res in run.results:
            icon = "✅ Pass" if res.passed else "❌ Fail"
            err = (res.error or "").replace("\n", " ")[:60] if res.error else "-"
            md.append(
                f"| `{res.case_id}` | {icon} | {res.score:.2f} | {res.latency_ms:.1f} | "
                f"`{res.failure_category.value}` | {err} |"
            )

        if fail_summary and fail_summary.get("distribution"):
            md.extend([
                "",
                "## Failure Mode Distribution",
                "",
                "| Failure Category | Occurrences | Share |",
                "| :--- | :---: | :---: |",
            ])
            dist = fail_summary.get("distribution", {})
            pcts = fail_summary.get("distribution_percentages", {})
            for cat, count in dist.items():
                pct = pcts.get(cat, 0.0)
                md.append(f"| `{cat}` | {count} | {pct:.1f}% |")

        md.append("")
        return "\n".join(md)

    @staticmethod
    def generate_comparison_markdown(comp: RegressionComparison) -> str:
        """
        Generate a comprehensive Markdown report comparing candidate run vs baseline.
        """
        verdict_badge = {
            "PASSED": "✅ NO REGRESSIONS",
            "IMPROVED": "🚀 CAPABILITY IMPROVED",
            "REGRESSED": "🚨 REGRESSION DETECTED",
            "BASELINE_ESTABLISHED": "📌 BASELINE ESTABLISHED",
        }.get(comp.verdict, comp.verdict)

        delta_sign = "+" if comp.score_delta > 0 else ""
        lat_sign = "+" if comp.latency_change_percent > 0 else ""

        md = [
            f"# Benchmark Regression Comparison: `{comp.benchmark_id}`",
            "",
            f"**Verdict:** {verdict_badge}  ",
            f"**Baseline Run:** `{comp.baseline_run_id}` ({comp.baseline_model})  ",
            f"**Candidate Run:** `{comp.candidate_run_id}` ({comp.candidate_model})  ",
            f"**Generated:** {comp.created_at}  ",
            "",
            f"> {comp.summary}",
            "",
            "## Metrics Comparison",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Score** | {comp.baseline_score:.2f}% | {comp.candidate_score:.2f}% | **{delta_sign}{comp.score_delta:.2f}%** |",
            f"| **Total Latency** | {comp.baseline_duration_ms / 1000:.2f}s | {comp.candidate_duration_ms / 1000:.2f}s | {lat_sign}{comp.latency_change_percent:.1f}% |",
            f"| **Common Cases** | {comp.common_cases_count} | {comp.common_cases_count} | - |",
            f"| **Regressed Cases** | - | {len(comp.regressed_cases)} | {len(comp.regressed_cases)} |",
            f"| **Fixed / New Passing** | - | {len(comp.fixed_cases)} | {len(comp.fixed_cases)} |",
            "",
        ]

        if comp.regressed_cases:
            md.extend([
                "## 🚨 Regressed Cases",
                "",
                "The following test cases passed in baseline but failed in candidate:",
                "",
                "| Case ID | Baseline Score | Candidate Score | Failure Category | Actual Output Snippet |",
                "| :--- | :---: | :---: | :---: | :--- |",
            ])
            for rc in comp.regressed_cases:
                snippet = (rc.get("actual_output") or "").replace("\n", " ")[:60]
                md.append(
                    f"| `{rc['case_id']}` | {rc['baseline_score']:.2f} | {rc['candidate_score']:.2f} | "
                    f"`{rc['failure_category']}` | {snippet} |"
                )
            md.append("")

        if comp.fixed_cases:
            md.extend([
                "## 🚀 Resolved / Improved Cases",
                "",
                "The following test cases failed in baseline but passed in candidate:",
                "",
                "| Case ID | Baseline Score | Candidate Score | Actual Output Snippet |",
                "| :--- | :---: | :---: | :--- |",
            ])
            for fc in comp.fixed_cases:
                snippet = (fc.get("actual_output") or "").replace("\n", " ")[:60]
                md.append(
                    f"| `{fc['case_id']}` | {fc['baseline_score']:.2f} | {fc['candidate_score']:.2f} | {snippet} |"
                )
            md.append("")

        return "\n".join(md)

    @staticmethod
    def export_run_json(run: EvalRun) -> str:
        """Serialize an evaluation run to formatted JSON string."""
        return json.dumps(run.to_dict(), indent=2)

    @staticmethod
    def export_comparison_json(comp: RegressionComparison) -> str:
        """Serialize a regression comparison to formatted JSON string."""
        return json.dumps(comp.to_dict(), indent=2)

    @staticmethod
    def save_report_to_disk(content: str, filename: str) -> Path:
        """Save report content to the project's data/reports directory."""
        reports_dir = get_settings().project_root / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / filename
        report_path.write_text(content, encoding="utf-8")
        return report_path
