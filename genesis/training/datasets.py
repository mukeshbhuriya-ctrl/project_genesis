"""
Dataset management, multi-format ingestion, normalization, and versioning for Project Genesis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger

log = get_logger("training.datasets")


class DatasetFormat(str, Enum):
    """Format schema of the training dataset."""
    INSTRUCTION = "instruction"   # instruction, input, output
    CHAT = "chat"                 # list of role/content messages
    COMPLETION = "completion"     # prompt, completion
    RAW = "raw"                   # plain text lines


@dataclass
class DatasetSample:
    """A single normalized data point within a dataset."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str = ""
    input: str = ""
    output: str = ""
    system: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id}
        if self.instruction:
            d["instruction"] = self.instruction
        if self.input:
            d["input"] = self.input
        if self.output:
            d["output"] = self.output
        if self.system:
            d["system"] = self.system
        if self.messages:
            d["messages"] = self.messages
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def estimated_tokens(self) -> int:
        """Estimate token count across all fields (~4 chars per token average)."""
        text_len = (
            len(self.instruction) +
            len(self.input) +
            len(self.output) +
            len(self.system) +
            sum(len(m.get("content", "")) for m in self.messages)
        )
        return max(1, math.ceil(text_len / 4.0))


@dataclass
class DatasetStats:
    """Statistical summary of sample lengths and token distributions."""
    total_samples: int = 0
    total_tokens: int = 0
    avg_tokens_per_sample: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    format: str = "instruction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "sample_count": self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_sample": round(self.avg_tokens_per_sample, 1),
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "format": self.format,
        }


@dataclass
class DatasetSplit:
    """Partitioned dataset samples for train/validation/test workflows."""
    train: list[DatasetSample] = field(default_factory=list)
    val: list[DatasetSample] = field(default_factory=list)
    test: list[DatasetSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        return {
            "train_count": len(self.train),
            "val_count": len(self.val),
            "test_count": len(self.test),
            "total": len(self.train) + len(self.val) + len(self.test),
        }


@dataclass
class Dataset:
    """Metadata record for a versioned dataset registered in SQLite."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    format: DatasetFormat = DatasetFormat.INSTRUCTION
    file_path: str = ""
    sample_count: int = 0
    total_tokens: int = 0
    checksum: str = ""
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
            "file_path": self.file_path,
            "sample_count": self.sample_count,
            "total_tokens": self.total_tokens,
            "checksum": self.checksum,
            "version": self.version,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DatasetManager:
    """
    Manages dataset ingestion, parsing, normalization, SQLite tracking, and splitting.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            settings = get_settings()
            self.db_path = get_db_path_from_url(settings.get_db_url())

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    # ── Parsing and Normalization ──

    @staticmethod
    def parse_samples_from_file(file_path: Path | str) -> tuple[list[DatasetSample], DatasetFormat]:
        """
        Parse raw file (JSONL, CSV, JSON, TXT) and normalize samples into DatasetSample.
        Auto-detects format if not explicitly known.
        """
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        suffix = p.suffix.lower()
        samples: list[DatasetSample] = []
        detected_format = DatasetFormat.INSTRUCTION

        if suffix in (".jsonl", ".ndjson"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        sample, fmt = DatasetManager.normalize_sample(obj)
                        samples.append(sample)
                        detected_format = fmt
                    except json.JSONDecodeError as exc:
                        log.warning(f"Skipping malformed JSON line in {p.name}: {exc}")

        elif suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        sample, fmt = DatasetManager.normalize_sample(item)
                        samples.append(sample)
                        detected_format = fmt
                elif isinstance(data, dict):
                    sample, fmt = DatasetManager.normalize_sample(data)
                    samples.append(sample)
                    detected_format = fmt

        elif suffix == ".csv":
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample, fmt = DatasetManager.normalize_sample(dict(row))
                    samples.append(sample)
                    detected_format = fmt

        else:
            # Plain text: line-by-line completion/raw format
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(DatasetSample(instruction=line, output=""))
            detected_format = DatasetFormat.RAW

        return samples, detected_format

    @staticmethod
    def normalize_sample(data: Any) -> tuple[DatasetSample, DatasetFormat]:
        """
        Normalize arbitrary dict schemas into standardized DatasetSample.
        Supports Alpaca, ShareGPT, OpenAI Chat, and Prompt-Completion formats.
        """
        if not isinstance(data, dict):
            return DatasetSample(instruction=str(data)), DatasetFormat.RAW

        # 1. Chat format: list of messages
        if "messages" in data and isinstance(data["messages"], list):
            msgs = []
            for m in data["messages"]:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    msgs.append({"role": str(m["role"]), "content": str(m["content"])})
            return (
                DatasetSample(
                    messages=msgs,
                    system=data.get("system", ""),
                    metadata=data.get("metadata", {}),
                ),
                DatasetFormat.CHAT,
            )

        # 2. ShareGPT format: conversations with 'from'/'value'
        if "conversations" in data and isinstance(data["conversations"], list):
            msgs = []
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            for turn in data["conversations"]:
                r = role_map.get(turn.get("from", ""), turn.get("from", "user"))
                msgs.append({"role": r, "content": str(turn.get("value", ""))})
            return (
                DatasetSample(messages=msgs, metadata=data.get("metadata", {})),
                DatasetFormat.CHAT,
            )

        # 3. Instruction / Alpaca format
        if "instruction" in data or "output" in data or "response" in data:
            instruction = data.get("instruction") or data.get("prompt") or ""
            input_val = data.get("input") or data.get("context") or ""
            output_val = data.get("output") or data.get("response") or data.get("completion") or ""
            system = data.get("system") or ""
            return (
                DatasetSample(
                    instruction=str(instruction),
                    input=str(input_val),
                    output=str(output_val),
                    system=str(system),
                    metadata=data.get("metadata", {}),
                ),
                DatasetFormat.INSTRUCTION,
            )

        # 4. Prompt - Completion
        if "prompt" in data:
            prompt = str(data.get("prompt", ""))
            completion = str(data.get("completion", ""))
            return (
                DatasetSample(
                    instruction=prompt,
                    output=completion,
                    metadata=data.get("metadata", {}),
                ),
                DatasetFormat.COMPLETION,
            )

        # Fallback to key-value string dump
        first_val = next(iter(data.values())) if data else ""
        return DatasetSample(instruction=str(first_val)), DatasetFormat.RAW

    # ── Statistics & Analysis ──

    @staticmethod
    def compute_statistics(samples: list[DatasetSample], format_type: str = "instruction") -> DatasetStats:
        """Compute sample counts, token estimations, and length distributions."""
        if not samples:
            return DatasetStats(format=format_type)

        tokens = [s.estimated_tokens() for s in samples]
        total_tok = sum(tokens)
        return DatasetStats(
            total_samples=len(samples),
            total_tokens=total_tok,
            avg_tokens_per_sample=total_tok / len(samples) if samples else 0.0,
            min_tokens=min(tokens) if tokens else 0,
            max_tokens=max(tokens) if tokens else 0,
            format=format_type,
        )

    # ── Splitting ──

    @staticmethod
    def split_dataset(
        samples: list[DatasetSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplit:
        """
        Deterministically split dataset samples into train, validation, and test subsets.
        """
        if not samples:
            return DatasetSplit()

        shuffled = list(samples)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = shuffled[:n_train]
        val_set = shuffled[n_train : n_train + n_val]
        test_set = shuffled[n_train + n_val :]

        # Guarantee at least 1 sample in train if samples exist
        if not train_set and samples:
            train_set = [samples[0]]
            val_set = []
            test_set = []

        return DatasetSplit(train=train_set, val=val_set, test=test_set)

    # ── Persistence & Database CRUD ──

    def register_dataset(
        self,
        name: str,
        samples: list[DatasetSample],
        description: str = "",
        format_type: DatasetFormat = DatasetFormat.INSTRUCTION,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        """
        Save samples to data/datasets/<id>.jsonl, compute SHA256 checksum, and save metadata to SQLite.
        """
        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        settings = get_settings()
        dest_dir = settings.project_root / "data" / "datasets"
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_path = dest_dir / f"{dataset_id}.jsonl"

        hasher = hashlib.sha256()
        with open(file_path, "w", encoding="utf-8") as f:
            for s in samples:
                line = json.dumps(s.to_dict()) + "\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))

        checksum = hasher.hexdigest()
        stats = self.compute_statistics(samples, format_type.value)

        ds = Dataset(
            id=dataset_id,
            name=name,
            description=description,
            format=format_type,
            file_path=str(file_path),
            sample_count=len(samples),
            total_tokens=stats.total_tokens,
            checksum=checksum,
            version=version,
            metadata=metadata or {},
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO training_datasets (
                    id, name, description, format, file_path, sample_count, total_tokens,
                    checksum, version, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    format = excluded.format,
                    file_path = excluded.file_path,
                    sample_count = excluded.sample_count,
                    total_tokens = excluded.total_tokens,
                    checksum = excluded.checksum,
                    version = excluded.version,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    ds.id,
                    ds.name,
                    ds.description,
                    ds.format.value,
                    ds.file_path,
                    ds.sample_count,
                    ds.total_tokens,
                    ds.checksum,
                    ds.version,
                    json.dumps(ds.metadata),
                    ds.created_at,
                    ds.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        log.info(f"Registered dataset '{name}' ({ds.id}) with {ds.sample_count} samples ({ds.total_tokens} tokens)")
        return ds

    def get_dataset(self, dataset_id_or_name: str) -> Dataset | None:
        """Fetch dataset metadata by ID or unique name from SQLite."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM training_datasets WHERE id = ? OR name = ?",
                (dataset_id_or_name, dataset_id_or_name),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Dataset(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                format=DatasetFormat(row["format"]),
                file_path=row["file_path"],
                sample_count=row["sample_count"],
                total_tokens=row["total_tokens"],
                checksum=row["checksum"],
                version=row["version"],
                metadata=json.loads(row["metadata"] or "{}"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        finally:
            conn.close()

    def list_datasets(self) -> list[Dataset]:
        """List all datasets ordered by creation time descending."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT id FROM training_datasets ORDER BY created_at DESC")
            ids = [row["id"] for row in cursor.fetchall()]
        finally:
            conn.close()

        res = []
        for d_id in ids:
            ds = self.get_dataset(d_id)
            if ds:
                res.append(ds)
        return res

    def load_samples(self, dataset_id_or_name: str) -> list[DatasetSample]:
        """Read and parse samples from the dataset's backing storage file."""
        ds = self.get_dataset(dataset_id_or_name)
        if not ds:
            raise ValueError(f"Dataset '{dataset_id_or_name}' not found.")

        samples, _ = self.parse_samples_from_file(ds.file_path)
        return samples
