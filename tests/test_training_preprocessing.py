"""
Tests for DataPreprocessor: deduplication, length filtering, syntax linting, and contamination checks.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.evaluation.benchmarks import BenchmarkRunner
from genesis.training.datasets import DatasetSample
from genesis.training.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Tests for quality filtering, deduplication, syntax checking, and contamination guard."""

    def test_deduplicate_exact_duplicates(self):
        preprocessor = DataPreprocessor()
        samples = [
            DatasetSample(instruction="What is gradient descent?", output="An optimization algorithm."),
            DatasetSample(instruction="What is gradient descent?", output="An optimization algorithm."),  # duplicate
            DatasetSample(instruction="What is backpropagation?", output="Gradient computation method."),
        ]

        cleaned, report = preprocessor.clean_and_filter(samples, deduplicate=True)
        assert len(cleaned) == 2
        assert report.duplicates_removed == 1
        assert report.final_count == 2

    def test_length_filtering(self):
        preprocessor = DataPreprocessor()
        samples = [
            DatasetSample(instruction="", output=""),                       # empty -> filtered
            DatasetSample(instruction="hi", output="ok"),                   # < 3 tokens -> filtered
            DatasetSample(instruction="This is a valid instruction of sufficient length.", output="This is a valid detailed output."),
        ]

        cleaned, report = preprocessor.clean_and_filter(samples, min_tokens=5)
        assert len(cleaned) == 1
        assert report.length_filtered == 2

    def test_ast_syntax_validation(self):
        preprocessor = DataPreprocessor()
        valid_sample = DatasetSample(
            instruction="Write a hello function",
            output="```python\ndef hello():\n    return 'world'\n```",
        )
        invalid_sample = DatasetSample(
            instruction="Write a broken function",
            output="```python\ndef broken(:\n    return\n```",
        )

        assert preprocessor.validate_code_syntax(valid_sample) is True
        assert preprocessor.validate_code_syntax(invalid_sample) is False

        cleaned, report = preprocessor.clean_and_filter([valid_sample, invalid_sample], validate_syntax=True)
        assert len(cleaned) == 1
        assert report.syntax_errors_filtered == 1
        assert cleaned[0].instruction == "Write a hello function"

    def test_benchmark_contamination_prevention(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        runner = BenchmarkRunner(db_path=tmp_db_path)
        preprocessor = DataPreprocessor(db_path=tmp_db_path)

        # Active evaluation benchmark case from Phase 4:
        # "If all transformers use self-attention, and Model Alpha is a transformer..."
        contaminated_sample = DatasetSample(
            instruction="If all transformers use self-attention, and Model Alpha is a transformer, does Model Alpha use self-attention?",
            output="Yes, it does.",
        )
        clean_sample = DatasetSample(
            instruction="Explain how attention weights are scaled by sqrt(d_k).",
            output="Scaling prevents dot products from growing excessively large in high dimensions.",
        )

        cleaned, report = preprocessor.clean_and_filter(
            [contaminated_sample, clean_sample],
            check_contamination=True,
        )

        assert len(cleaned) == 1
        assert report.contaminated_removed == 1
        assert cleaned[0].instruction == clean_sample.instruction
