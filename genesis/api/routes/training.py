"""
API routes for Dataset Management, Fine-Tuning Orchestration, and Checkpoint Deployment.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.training.datasets import DatasetFormat, DatasetManager, DatasetSample
from genesis.training.experiments import ExperimentTracker, Hyperparameters
from genesis.training.export import CheckpointManager
from genesis.training.preprocessing import DataPreprocessor
from genesis.training.trainer import FineTuningTrainer

router = APIRouter(prefix="/training", tags=["Training & Datasets"])


def _get_dataset_mgr() -> DatasetManager:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return DatasetManager(db_path=db_path)


def _get_preprocessor() -> DataPreprocessor:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return DataPreprocessor(db_path=db_path)


def _get_tracker() -> ExperimentTracker:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return ExperimentTracker(db_path=db_path)


def _get_trainer() -> FineTuningTrainer:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return FineTuningTrainer(db_path=db_path)


def _get_checkpoint_mgr() -> CheckpointManager:
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    return CheckpointManager(db_path=db_path)


# ── Request / Response Models ──

class CreateDatasetRequest(BaseModel):
    name: str = Field(..., description="Unique dataset name")
    description: str = Field(default="", description="Dataset summary")
    samples: list[dict[str, Any]] = Field(..., description="Array of sample dictionaries")
    format: str = Field(default="instruction", description="Dataset format: instruction, chat, completion, raw")
    version: str = Field(default="1.0.0", description="Semantic dataset version")


class PreprocessDatasetRequest(BaseModel):
    min_tokens: int = Field(default=3, ge=1)
    max_tokens: int = Field(default=4096, ge=10)
    deduplicate: bool = True
    validate_syntax: bool = True
    check_contamination: bool = True


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., description="Descriptive name of the experiment")
    base_model: str = Field(default="qwen2.5:7b", description="Base model identifier")
    dataset_id: str = Field(..., description="Dataset ID to train on")
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="LoRA and training hyperparameters")
    total_steps: int = Field(default=20, ge=1)


class DeployCheckpointRequest(BaseModel):
    deployed_name: str = Field(..., description="Model name to register in Ollama (e.g. qwen2.5-tuned:v1)")
    system_prompt: str = Field(default="", description="Optional customized system prompt")
    parameters: dict[str, Any] | None = Field(default=None, description="Optional inference parameters")


class RollbackRequest(BaseModel):
    baseline_model: str = Field(default="qwen2.5:7b", description="Baseline model name to restore")


# ── Dataset Endpoints ──

@router.get("/datasets")
async def list_datasets():
    """List all registered training datasets."""
    mgr = _get_dataset_mgr()
    datasets = mgr.list_datasets()
    return {
        "count": len(datasets),
        "datasets": [d.to_dict() for d in datasets],
    }


@router.post("/datasets/create")
async def create_dataset(req: CreateDatasetRequest):
    """Register and persist a new dataset from provided samples."""
    mgr = _get_dataset_mgr()
    fmt = DatasetFormat(req.format) if req.format in DatasetFormat._value2member_map_ else DatasetFormat.INSTRUCTION
    samples = [mgr.normalize_sample(s)[0] for s in req.samples]

    ds = mgr.register_dataset(
        name=req.name,
        samples=samples,
        description=req.description,
        format_type=fmt,
        version=req.version,
    )
    return ds.to_dict()


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, preview_limit: int = Query(10, ge=1, le=100)):
    """Retrieve dataset metadata, token statistics, and sample preview."""
    mgr = _get_dataset_mgr()
    ds = mgr.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    samples = mgr.load_samples(dataset_id)
    stats = mgr.compute_statistics(samples, ds.format.value)

    return {
        "dataset": ds.to_dict(),
        "statistics": stats.to_dict(),
        "preview": [s.to_dict() for s in samples[:preview_limit]],
    }


@router.post("/datasets/{dataset_id}/preprocess")
async def preprocess_dataset(dataset_id: str, req: PreprocessDatasetRequest):
    """Run data quality filtering, deduplication, syntax checking, and contamination guard."""
    mgr = _get_dataset_mgr()
    ds = mgr.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    samples = mgr.load_samples(dataset_id)
    preprocessor = _get_preprocessor()

    cleaned, report = preprocessor.clean_and_filter(
        samples=samples,
        min_tokens=req.min_tokens,
        max_tokens=req.max_tokens,
        deduplicate=req.deduplicate,
        validate_syntax=req.validate_syntax,
        check_contamination=req.check_contamination,
    )

    # Register as cleaned dataset variant
    cleaned_ds = mgr.register_dataset(
        name=f"{ds.name}-cleaned",
        samples=cleaned,
        description=f"Preprocessed variant of {ds.name}",
        format_type=ds.format,
        version=f"{ds.version}.clean",
    )

    return {
        "report": report.to_dict(),
        "cleaned_dataset": cleaned_ds.to_dict(),
    }


# ── Experiment Endpoints ──

@router.get("/experiments")
async def list_experiments(limit: int = Query(50, ge=1, le=200)):
    """List fine-tuning experiments ordered by recency."""
    tracker = _get_tracker()
    experiments = tracker.list_experiments(limit=limit)
    return {
        "count": len(experiments),
        "experiments": [e.to_dict() for e in experiments],
    }


@router.post("/experiments/create")
async def create_experiment(req: CreateExperimentRequest):
    """Configure a new fine-tuning experiment."""
    tracker = _get_tracker()
    mgr = _get_dataset_mgr()

    ds = mgr.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    hp = Hyperparameters.from_dict(req.hyperparameters)
    exp = tracker.create_experiment(
        name=req.name,
        base_model=req.base_model,
        dataset_id=req.dataset_id,
        hyperparameters=hp,
        total_steps=req.total_steps,
    )
    return exp.to_dict()


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Retrieve full experiment status, hyperparameters, and loss metrics trajectory."""
    tracker = _get_tracker()
    exp = tracker.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return exp.to_dict()


@router.post("/experiments/{experiment_id}/start")
async def start_training(experiment_id: str, step_delay_s: float = Query(0.02, ge=0.0, le=1.0)):
    """Trigger training loop execution for an experiment."""
    trainer = _get_trainer()
    try:
        completed_exp = await trainer.run_training_async(
            experiment_id=experiment_id,
            step_delay_s=step_delay_s,
        )
        return completed_exp.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Checkpoint & Deployment Endpoints ──

@router.get("/experiments/{experiment_id}/checkpoints")
async def list_checkpoints(experiment_id: str):
    """List all checkpoints generated for an experiment."""
    chk_mgr = _get_checkpoint_mgr()
    checkpoints = chk_mgr.list_checkpoints(experiment_id=experiment_id)
    best = chk_mgr.get_best_checkpoint(experiment_id)
    return {
        "count": len(checkpoints),
        "best_checkpoint_id": best.id if best else None,
        "checkpoints": [c.to_dict() for c in checkpoints],
    }


@router.post("/checkpoints/{checkpoint_id}/deploy")
async def deploy_checkpoint(checkpoint_id: str, req: DeployCheckpointRequest):
    """Deploy a checkpoint as a registered model in the local Ollama runtime."""
    chk_mgr = _get_checkpoint_mgr()
    chk = chk_mgr.get_checkpoint(checkpoint_id)
    if not chk:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found.")

    tracker = _get_tracker()
    exp = tracker.get_experiment(chk.experiment_id)
    base_model = exp.base_model if exp else "qwen2.5:7b"

    modelfile_path = chk_mgr.deploy_to_ollama(
        checkpoint_id=checkpoint_id,
        deployed_name=req.deployed_name,
        base_model=base_model,
        system_prompt=req.system_prompt,
        parameters=req.parameters,
    )

    return {
        "status": "deployed",
        "checkpoint_id": checkpoint_id,
        "deployed_name": req.deployed_name,
        "modelfile_path": modelfile_path,
    }


@router.post("/experiments/{experiment_id}/rollback")
async def rollback_model(experiment_id: str, req: RollbackRequest):
    """Roll back deployed model to baseline if regression is detected."""
    chk_mgr = _get_checkpoint_mgr()
    res = chk_mgr.rollback_model(experiment_id, baseline_model=req.baseline_model)
    return res
