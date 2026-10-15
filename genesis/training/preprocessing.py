"""
Data quality, deduplication, syntax validation, and benchmark contamination prevention.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.training.datasets import DatasetSample

log = get_logger("training.preprocessing")


@dataclass
class PreprocessingReport:
    """Summary metrics of the preprocessing and filtering pipeline."""
    initial_count: int = 0
    final_count: int = 0
    duplicates_removed: int = 0
    length_filtered: int = 0
    syntax_errors_filtered: int = 0
    contaminated_removed: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_count": self.initial_count,
            "final_count": self.final_count,
            "retained_percentage": round((self.final_count / max(1, self.initial_count)) * 100.0, 1),
            "duplicates_removed": self.duplicates_removed,
            "length_filtered": self.length_filtered,
            "syntax_errors_filtered": self.syntax_errors_filtered,
            "contaminated_removed": self.contaminated_removed,
            "details": self.details,
        }


class DataPreprocessor:
    """
    Cleans training samples, suppresses duplicates, verifies code syntax,
    and defends against benchmark contamination.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            settings = get_settings()
            self.db_path = get_db_path_from_url(settings.get_db_url())

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    # ── Contamination Guard ──

    def _get_benchmark_signatures(self) -> set[tuple[str, ...]]:
        """Fetch prompt n-grams from eval_cases to detect data contamination."""
        signatures: set[tuple[str, ...]] = set()
        try:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT prompt FROM eval_cases")
                for row in cursor.fetchall():
                    prompt = row["prompt"].strip().lower()
                    words = re.findall(r"\w+", prompt)
                    if len(words) >= 8:
                        for i in range(len(words) - 8 + 1):
                            signatures.add(tuple(words[i : i + 8]))
                    elif len(words) >= 4:
                        signatures.add(tuple(words))
            finally:
                conn.close()
        except sqlite3.OperationalError:
            log.warning("eval_cases table not found for contamination check; skipping.")
        return signatures

    # ── Syntax Validator ──

    @staticmethod
    def validate_code_syntax(sample: DatasetSample) -> bool:
        """
        Extract Python code snippets and verify they parse cleanly with AST.
        Returns True if code is valid or no code present; False if code has SyntaxError.
        """
        combined = f"{sample.instruction}\n{sample.output}"
        # Extract markdown code blocks ```python ... ``` or ``` ... ```
        blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", combined, re.DOTALL)
        if not blocks:
            # Check if output looks like raw python code
            if sample.output.strip().startswith(("def ", "class ", "import ", "from ")):
                blocks = [sample.output]

        for block in blocks:
            try:
                ast.parse(block)
            except SyntaxError as exc:
                log.debug(f"Syntax validation failed on sample {sample.id}: {exc}")
                return False

        return True

    # ── Shingling / Jaccard Similarity ──

    @staticmethod
    def _compute_shingles(text: str, k: int = 3) -> set[str]:
        """Compute k-shingles (word n-grams) for similarity matching."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < k:
            return set(words)
        return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}

    @staticmethod
    def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    # ── Pipeline Execution ──

    def clean_and_filter(
        self,
        samples: list[DatasetSample],
        min_tokens: int = 3,
        max_tokens: int = 4096,
        deduplicate: bool = True,
        validate_syntax: bool = True,
        check_contamination: bool = True,
        near_duplicate_threshold: float = 0.90,
    ) -> tuple[list[DatasetSample], PreprocessingReport]:
        """
        Run the complete data cleaning and filtering pipeline.
        """
        report = PreprocessingReport(initial_count=len(samples))
        benchmark_signatures = self._get_benchmark_signatures() if check_contamination else set()

        seen_hashes: set[str] = set()
        seen_shingles: list[set[str]] = []
        cleaned_samples: list[DatasetSample] = []

        for s in samples:
            # 1. Content completeness check
            has_instruction = bool(s.instruction.strip())
            has_messages = bool(s.messages)
            if not has_instruction and not has_messages:
                report.length_filtered += 1
                continue

            # 2. Token length bounds
            tok_count = s.estimated_tokens()
            if tok_count < min_tokens or tok_count > max_tokens:
                report.length_filtered += 1
                continue

            # 3. Exact Hash Deduplication
            content_repr = f"{s.instruction.strip()}###{s.input.strip()}###{s.output.strip()}###{str(s.messages)}"
            content_hash = hashlib.sha256(content_repr.encode("utf-8")).hexdigest()
            if deduplicate and content_hash in seen_hashes:
                report.duplicates_removed += 1
                continue

            # 4. Near-Duplicate Suppression (Jaccard similarity on shingles)
            if deduplicate and near_duplicate_threshold < 1.0:
                current_shingles = self._compute_shingles(content_repr, k=3)
                is_near_dup = False
                for existing_shingles in seen_shingles:
                    if self.jaccard_similarity(current_shingles, existing_shingles) >= near_duplicate_threshold:
                        is_near_dup = True
                        break
                if is_near_dup:
                    report.duplicates_removed += 1
                    continue
                seen_shingles.append(current_shingles)

            # 5. Benchmark Contamination Check
            if check_contamination and benchmark_signatures:
                sample_text = f"{s.instruction} {s.input} {s.output}".lower()
                sample_words = re.findall(r"\w+", sample_text)
                is_contaminated = False
                if len(sample_words) >= 8:
                    for i in range(len(sample_words) - 8 + 1):
                        if tuple(sample_words[i : i + 8]) in benchmark_signatures:
                            is_contaminated = True
                            break
                elif len(sample_words) >= 4:
                    if tuple(sample_words) in benchmark_signatures:
                        is_contaminated = True

                if is_contaminated:
                    report.contaminated_removed += 1
                    log.warning(f"Sample {s.id} rejected: contains evaluation benchmark signature.")
                    continue

            # 6. AST Code Syntax Validation
            if validate_syntax and not self.validate_code_syntax(s):
                report.syntax_errors_filtered += 1
                continue

            # Passed all filters
            seen_hashes.add(content_hash)
            cleaned_samples.append(s)

        report.final_count = len(cleaned_samples)
        log.info(
            f"Preprocessing complete: {report.initial_count} -> {report.final_count} samples "
            f"({report.duplicates_removed} duplicates, {report.length_filtered} length, "
            f"{report.syntax_errors_filtered} syntax, {report.contaminated_removed} contaminated)"
        )
        return cleaned_samples, report
