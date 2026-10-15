"""
Checkpoint management, Modelfile creation, Ollama model registration, and automated rollback.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger

log = get_logger("training.export")


@dataclass
class TrainingCheckpoint:
    """A saved model checkpoint or adapter weights directory."""
    id: str = field(default_factory=lambda: f"chk-{uuid.uuid4().hex[:12]}")
    experiment_id: str = ""
    step: int = 0
    epoch: float = 0.0
    training_loss: float = 0.0
    eval_loss: float | None = None
    file_path: str = ""
    format: str = "adapter"  # "adapter", "gguf", "modelfile"
    is_best: bool = False
    deployed_name: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "step": self.step,
            "epoch": round(self.epoch, 2),
            "training_loss": round(self.training_loss, 4),
            "eval_loss": round(self.eval_loss, 4) if self.eval_loss is not None else None,
            "file_path": self.file_path,
            "format": self.format,
            "is_best": self.is_best,
            "deployed_name": self.deployed_name,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class CheckpointManager:
    """
    Manages checkpoint storage, GGUF/Modelfile synthesis, Ollama service registration,
    and rollback to verified baselines.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            settings = get_settings()
            self.db_path = get_db_path_from_url(settings.get_db_url())

    def _get_connection(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    # ── Checkpoint Persistence ──

    def save_checkpoint(self, checkpoint: TrainingCheckpoint) -> None:
        """Insert or update checkpoint record in SQLite."""
        conn = self._get_connection()
        try:
            # If this is marked as best, unmark prior bests for this experiment
            if checkpoint.is_best:
                conn.execute(
                    "UPDATE training_checkpoints SET is_best = 0 WHERE experiment_id = ?",
                    (checkpoint.experiment_id,),
                )

            conn.execute(
                """
                INSERT INTO training_checkpoints (
                    id, experiment_id, step, epoch, training_loss, eval_loss,
                    file_path, format, is_best, deployed_name, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    training_loss = excluded.training_loss,
                    eval_loss = excluded.eval_loss,
                    is_best = excluded.is_best,
                    deployed_name = excluded.deployed_name,
                    metadata = excluded.metadata
                """,
                (
                    checkpoint.id,
                    checkpoint.experiment_id,
                    checkpoint.step,
                    checkpoint.epoch,
                    checkpoint.training_loss,
                    checkpoint.eval_loss,
                    checkpoint.file_path,
                    checkpoint.format,
                    1 if checkpoint.is_best else 0,
                    checkpoint.deployed_name,
                    checkpoint.created_at,
                    json.dumps(checkpoint.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_checkpoint(self, checkpoint_id: str) -> TrainingCheckpoint | None:
        """Retrieve checkpoint metadata by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM training_checkpoints WHERE id = ?", (checkpoint_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return TrainingCheckpoint(
                id=row["id"],
                experiment_id=row["experiment_id"],
                step=row["step"],
                epoch=row["epoch"],
                training_loss=row["training_loss"],
                eval_loss=row["eval_loss"],
                file_path=row["file_path"],
                format=row["format"],
                is_best=bool(row["is_best"]),
                deployed_name=row["deployed_name"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def list_checkpoints(self, experiment_id: str | None = None) -> list[TrainingCheckpoint]:
        """List checkpoints, optionally filtered by experiment ID."""
        conn = self._get_connection()
        try:
            if experiment_id:
                cursor = conn.execute(
                    "SELECT id FROM training_checkpoints WHERE experiment_id = ? ORDER BY step DESC",
                    (experiment_id,),
                )
            else:
                cursor = conn.execute("SELECT id FROM training_checkpoints ORDER BY created_at DESC")
            ids = [row["id"] for row in cursor.fetchall()]
        finally:
            conn.close()

        res = []
        for c_id in ids:
            chk = self.get_checkpoint(c_id)
            if chk:
                res.append(chk)
        return res

    def get_best_checkpoint(self, experiment_id: str) -> TrainingCheckpoint | None:
        """Find checkpoint with lowest eval loss or lowest training loss for an experiment."""
        checkpoints = self.list_checkpoints(experiment_id)
        if not checkpoints:
            return None

        # Prefer explicitly marked is_best
        for c in checkpoints:
            if c.is_best:
                return c

        # Otherwise sort by eval_loss if available, else training_loss
        with_eval = [c for c in checkpoints if c.eval_loss is not None]
        if with_eval:
            return min(with_eval, key=lambda c: c.eval_loss)  # type: ignore

        return min(checkpoints, key=lambda c: c.training_loss)

    # ── Modelfile Generation & Ollama Deployment ──

    @staticmethod
    def build_modelfile(
        base_model: str,
        system_prompt: str = "",
        parameters: dict[str, Any] | None = None,
        adapter_path: Path | str | None = None,
    ) -> str:
        """
        Synthesize an Ollama Modelfile specification for tuned configurations or adapters.
        """
        lines = [f"FROM {base_model}"]

        params = parameters or {
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }

        for param_name, param_val in params.items():
            if isinstance(param_val, list):
                for item in param_val:
                    lines.append(f'PARAMETER {param_name} "{item}"')
            else:
                lines.append(f"PARAMETER {param_name} {param_val}")

        if system_prompt.strip():
            # Multiline system prompt
            lines.append('SYSTEM """')
            lines.append(system_prompt.strip())
            lines.append('"""')

        if adapter_path:
            norm_path = Path(adapter_path).resolve().as_posix()
            lines.append(f"ADAPTER {norm_path}")

        return "\n".join(lines) + "\n"

    def deploy_to_ollama(
        self,
        checkpoint_id: str,
        deployed_name: str,
        base_model: str = "qwen2.5:7b",
        system_prompt: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """
        Write Modelfile, execute `ollama create <deployed_name> -f <modelfile>`,
        and register deployment in SQLite.
        """
        chk = self.get_checkpoint(checkpoint_id)
        if not chk:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

        safe_dir_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", deployed_name)
        deploy_dir = self.db_path.parent / "models" / "tuned" / safe_dir_name
        deploy_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = deploy_dir / "Modelfile"

        modelfile_content = self.build_modelfile(
            base_model=base_model,
            system_prompt=system_prompt,
            parameters=parameters,
            adapter_path=chk.file_path if chk.format == "adapter" and Path(chk.file_path).is_file() else None,
        )
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        # Execute Ollama CLI command if ollama executable is on system
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            local_bin = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
            if local_bin.is_file():
                ollama_bin = str(local_bin)

        if ollama_bin:
            try:
                cmd = [ollama_bin, "create", deployed_name, "-f", str(modelfile_path)]
                log.info(f"Invoking Ollama CLI: {' '.join(cmd)}")
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    log.warning(f"Ollama create output: {proc.stderr}")
            except Exception as exc:
                log.warning(f"Could not automatically execute ollama create: {exc}")

        # Update checkpoint deployed name in SQLite
        chk.deployed_name = deployed_name
        self.save_checkpoint(chk)
        log.info(f"Deployed checkpoint '{checkpoint_id}' as Ollama model '{deployed_name}'")
        return str(modelfile_path)

    # ── Rollback ──

    def rollback_model(self, experiment_id: str, baseline_model: str) -> dict[str, Any]:
        """
        Roll back active model configuration to a baseline model if a candidate regresses.
        """
        checkpoints = self.list_checkpoints(experiment_id)
        for c in checkpoints:
            if c.deployed_name:
                log.info(f"Deactivating candidate deployment '{c.deployed_name}'")
                c.metadata["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
                c.metadata["reverted_to"] = baseline_model
                self.save_checkpoint(c)

        deactivated = [c.deployed_name for c in checkpoints if c.deployed_name]
        return {
            "status": "rolled_back",
            "experiment_id": experiment_id,
            "active_baseline": baseline_model,
            "deactivated_models": deactivated,
            "deployed_name": deactivated[0] if deactivated else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
