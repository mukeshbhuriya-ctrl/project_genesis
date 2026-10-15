"""
Tests for DatasetManager, normalization, statistics, and deterministic splitting.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.training.datasets import (
    DatasetFormat,
    DatasetManager,
    DatasetSample,
    DatasetSplit,
    DatasetStats,
)


class TestDatasetNormalization:
    """Tests for normalizing diverse dataset structures into DatasetSample."""

    def test_normalize_alpaca_instruction_format(self):
        sample_dict = {
            "instruction": "Explain LoRA in machine learning.",
            "input": "Keep it under 50 words.",
            "output": "Low-Rank Adaptation freezes base weights and trains rank decomposition matrices.",
        }
        sample, fmt = DatasetManager.normalize_sample(sample_dict)
        assert fmt == DatasetFormat.INSTRUCTION
        assert sample.instruction == "Explain LoRA in machine learning."
        assert sample.input == "Keep it under 50 words."
        assert "Low-Rank Adaptation" in sample.output
        assert sample.estimated_tokens() > 10

    def test_normalize_chat_messages_format(self):
        sample_dict = {
            "system": "You are a coding assistant.",
            "messages": [
                {"role": "user", "content": "How to sort a list?"},
                {"role": "assistant", "content": "Use sorted(lst) or lst.sort()."},
            ],
        }
        sample, fmt = DatasetManager.normalize_sample(sample_dict)
        assert fmt == DatasetFormat.CHAT
        assert len(sample.messages) == 2
        assert sample.messages[0]["role"] == "user"
        assert sample.system == "You are a coding assistant."

    def test_normalize_sharegpt_conversations_format(self):
        sample_dict = {
            "conversations": [
                {"from": "human", "value": "Write a python function"},
                {"from": "gpt", "value": "def foo(): pass"},
            ]
        }
        sample, fmt = DatasetManager.normalize_sample(sample_dict)
        assert fmt == DatasetFormat.CHAT
        assert len(sample.messages) == 2
        assert sample.messages[0]["role"] == "user"
        assert sample.messages[1]["role"] == "assistant"

    def test_normalize_prompt_completion_format(self):
        sample_dict = {"prompt": "What is 2+2?", "completion": "4"}
        sample, fmt = DatasetManager.normalize_sample(sample_dict)
        assert fmt == DatasetFormat.COMPLETION
        assert sample.instruction == "What is 2+2?"
        assert sample.output == "4"


class TestDatasetManager:
    """Tests for DatasetManager persistence, statistics, and splitting."""

    def test_compute_statistics(self):
        samples = [
            DatasetSample(instruction="Short text", output="Answer"),
            DatasetSample(instruction="A much longer instruction with multiple sentences and details.", output="Detailed long answer"),
        ]
        stats = DatasetManager.compute_statistics(samples, "instruction")
        assert stats.total_samples == 2
        assert stats.total_tokens > 10
        assert stats.min_tokens < stats.max_tokens
        assert stats.avg_tokens_per_sample > 0

    def test_split_dataset_deterministic(self):
        samples = [DatasetSample(instruction=f"Prompt {i}", output=f"Answer {i}") for i in range(100)]
        split1 = DatasetManager.split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        split2 = DatasetManager.split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

        assert len(split1.train) == 80
        assert len(split1.val) == 10
        assert len(split1.test) == 10

        # Determinism check
        assert [s.instruction for s in split1.train] == [s.instruction for s in split2.train]

    def test_register_and_load_dataset(self, tmp_db_path: Path):
        initialize_database(tmp_db_path)
        mgr = DatasetManager(db_path=tmp_db_path)

        samples = [
            DatasetSample(instruction="Define overfitting", output="High variance on test data"),
            DatasetSample(instruction="Define underfitting", output="High bias on training data"),
        ]

        ds = mgr.register_dataset(
            name="ml-fundamentals",
            samples=samples,
            description="Core machine learning concepts",
            format_type=DatasetFormat.INSTRUCTION,
            version="1.0.0",
        )

        assert ds.name == "ml-fundamentals"
        assert ds.sample_count == 2
        assert len(ds.checksum) == 64  # SHA256 length

        # Retrieve from database
        loaded_ds = mgr.get_dataset(ds.id)
        assert loaded_ds is not None
        assert loaded_ds.id == ds.id
        assert loaded_ds.sample_count == 2

        # Load samples from disk
        loaded_samples = mgr.load_samples(ds.id)
        assert len(loaded_samples) == 2
        assert loaded_samples[0].instruction == "Define overfitting"

        # List datasets
        datasets = mgr.list_datasets()
        assert len(datasets) >= 1
        assert any(d.id == ds.id for d in datasets)
