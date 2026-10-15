"""
Experiment tracking, hyperparameter configuration, and loss curve monitoring for Project Genesis.
"""

from __future__ import annotations

import json
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

log = get_logger("training.experiments")


class ExperimentStatus(str, Enum):
    """Lifecycle states of a training experiment."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Hyperparameters:
    """LoRA / QLoRA and training hyperparameters tuned for 8 GB VRAM workstation."""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_seq_length: int = 512
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs": self.epochs,
            "max_seq_length": self.max_seq_length,
            "quantization": self.quantization,
            "target_modules": self.target_modules,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hyperparameters:
        return cls(
            lora_r=int(d.get("lora_r", 16)),
            lora_alpha=int(d.get("lora_alpha", 32)),
            lora_dropout=float(d.get("lora_dropout", 0.05)),
            learning_rate=float(d.get("learning_rate", 2e-4)),
            batch_size=int(d.get("batch_size", 4)),
            gradient_accumulation_steps=int(d.get("gradient_accumulation_steps", 4)),
            epochs=int(d.get("epochs", 3)),
            max_seq_length=int(d.get("max_seq_length", 512)),
            quantization=str(d.get("quantization", "4bit")),
            target_modules=d.get("target_modules", ["q_proj", "v_proj"]),
            warmup_ratio=float(d.get("warmup_ratio", 0.03)),
            weight_decay=float(d.get("weight_decay", 0.01)),
        )


@dataclass
class TrainingMetricPoint:
    """Recorded metric telemetry at an individual training step."""
    step: int
    epoch: float
    loss: float
    eval_loss: float | None = None
    learning_rate: float = 0.0
    tokens_per_sec: float = 0.0
    vram_mb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": round(self.epoch, 2),
            "loss": round(self.loss, 4),
            "eval_loss": round(self.eval_loss, 4) if self.eval_loss is not None else None,
            "learning_rate": self.learning_rate,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "vram_mb": round(self.vram_mb, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class TrainingExperiment:
    """Representation of an experiment run tracked in SQLite."""
    id: str = field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:12]}")
    name: str = ""
    base_model: str = "qwen2.5:7b"
    dataset_id: str = ""
    status: ExperimentStatus = ExperimentStatus.PENDING
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    current_epoch: int = 0
    total_epochs: int = 1
    current_step: int = 0
    total_steps: int = 0
    training_loss: float | None = None
    eval_loss: float | None = None
    duration_ms: float = 0.0
    error: str | None = None
    metrics: list[TrainingMetricPoint] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_model": self.base_model,
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "hyperparameters": self.hyperparameters.to_dict(),
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_pct": round((self.current_step / max(1, self.total_steps)) * 100.0, 1) if self.total_steps > 0 else 0.0,
            "training_loss": round(self.training_loss, 4) if self.training_loss is not None else None,
            "eval_loss": round(self.eval_loss, 4) if self.eval_loss is not None else None,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "metric_points_count": len(self.metrics),
            "metrics": [m.to_dict() for m in self.metrics],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


class ExperimentTracker:
    """
    CRUD management and metrics telemetry persistence for training experiments.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            settings = get_settings()
            self.db_path = get_db_path_from_url(settings.get_db_url())

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    def create_experiment(
        self,
        name: str,
        base_model: str,
        dataset_id: str,
        hyperparameters: Hyperparameters | None = None,
        total_steps: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingExperiment:
        """Create and persist a new training experiment."""
        hp = hyperparameters or Hyperparameters()
        exp = TrainingExperiment(
            name=name,
            base_model=base_model,
            dataset_id=dataset_id,
            hyperparameters=hp,
            total_epochs=hp.epochs,
            total_steps=total_steps,
            metadata=metadata or {},
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO training_experiments (
                    id, name, base_model, dataset_id, status, hyperparameters,
                    current_epoch, total_epochs, current_step, total_steps,
                    training_loss, eval_loss, duration_ms, error, metrics,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.id,
                    exp.name,
                    exp.base_model,
                    exp.dataset_id,
                    exp.status.value,
                    json.dumps(exp.hyperparameters.to_dict()),
                    exp.current_epoch,
                    exp.total_epochs,
                    exp.current_step,
                    exp.total_steps,
                    exp.training_loss,
                    exp.eval_loss,
                    exp.duration_ms,
                    exp.error,
                    json.dumps([]),
                    exp.created_at,
                    json.dumps(exp.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        log.info(f"Created training experiment '{name}' ({exp.id}) targeting '{base_model}'")
        return exp

    def get_experiment(self, experiment_id: str) -> TrainingExperiment | None:
        """Retrieve full experiment details and metrics log by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM training_experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            if not row:
                return None

            hp_dict = json.loads(row["hyperparameters"] or "{}")
            metrics_raw = json.loads(row["metrics"] or "[]")
            metric_points = [
                TrainingMetricPoint(
                    step=m["step"],
                    epoch=m["epoch"],
                    loss=m["loss"],
                    eval_loss=m.get("eval_loss"),
                    learning_rate=m.get("learning_rate", 0.0),
                    tokens_per_sec=m.get("tokens_per_sec", 0.0),
                    vram_mb=m.get("vram_mb", 0.0),
                    timestamp=m.get("timestamp", ""),
                )
                for m in metrics_raw
            ]

            return TrainingExperiment(
                id=row["id"],
                name=row["name"],
                base_model=row["base_model"],
                dataset_id=row["dataset_id"],
                status=ExperimentStatus(row["status"]),
                hyperparameters=Hyperparameters.from_dict(hp_dict),
                current_epoch=row["current_epoch"],
                total_epochs=row["total_epochs"],
                current_step=row["current_step"],
                total_steps=row["total_steps"],
                training_loss=row["training_loss"],
                eval_loss=row["eval_loss"],
                duration_ms=row["duration_ms"],
                error=row["error"],
                metrics=metric_points,
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_experiments(self, limit: int = 50) -> list[TrainingExperiment]:
        """List historical experiments ordered by recency."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT id FROM training_experiments ORDER BY created_at DESC LIMIT ?", (limit,))
            ids = [row["id"] for row in cursor.fetchall()]
        finally:
            conn.close()

        res = []
        for e_id in ids:
            exp = self.get_experiment(e_id)
            if exp:
                res.append(exp)
        return res

    def record_metric(self, experiment_id: str, metric: TrainingMetricPoint) -> None:
        """Append metric telemetry to the experiment's log and update current status in SQLite."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT metrics FROM training_experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment '{experiment_id}' not found.")

            metrics_list = json.loads(row["metrics"] or "[]")
            metrics_list.append(metric.to_dict())

            conn.execute(
                """
                UPDATE training_experiments
                SET current_step = ?,
                    current_epoch = ?,
                    training_loss = ?,
                    eval_loss = COALESCE(?, eval_loss),
                    metrics = ?
                WHERE id = ?
                """,
                (
                    metric.step,
                    int(metric.epoch),
                    metric.loss,
                    metric.eval_loss,
                    json.dumps(metrics_list),
                    experiment_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def set_total_steps(self, experiment_id: str, total_steps: int) -> None:
        """Update planned total steps for an experiment."""
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE training_experiments SET total_steps = ? WHERE id = ?",
                (total_steps, experiment_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Update experiment execution lifecycle status."""
        conn = self._get_connection()
        try:
            completed_at = datetime.now(timezone.utc).isoformat() if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED) else None
            conn.execute(
                """
                UPDATE training_experiments
                SET status = ?,
                    error = COALESCE(?, error),
                    duration_ms = COALESCE(?, duration_ms),
                    completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (status.value, error, duration_ms, completed_at, experiment_id),
            )
            conn.commit()
        finally:
            conn.close()
