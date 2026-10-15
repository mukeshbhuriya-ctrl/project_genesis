"""
Continuous Multi-Objective Pareto Optimization Engine (Phase 16).

Formulates evolutionary search procedures based on NSGA-II non-dominated sorting
and crowding distance estimation to discover the optimal Pareto trade-off frontier
between latency (TTFT), KV-cache VRAM footprint, throughput (tps), and accuracy.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.core.self_compiler import SelfHostingCompilerEngine

logger = get_logger(__name__)


class RuntimeConfig(BaseModel):
    """Candidate runtime inference hyperparameters for local workstation."""
    quant_bits: int = Field(default=4, ge=2, le=16)  # 2, 3, 4, 8, 16
    context_window: int = Field(default=2048, ge=512, le=16384)
    lookahead_k: int = Field(default=4, ge=1, le=8)
    batch_size: int = Field(default=1, ge=1, le=8)
    thread_count: int = Field(default=8, ge=1, le=20)


class ParetoPoint(BaseModel):
    """Evaluated runtime configuration on multi-objective metric space."""
    id: str = Field(default_factory=lambda: f"point-{uuid.uuid4().hex[:8]}")
    config: RuntimeConfig
    latency_ms: float = Field(ge=0.0)      # Minimize
    vram_mb: float = Field(ge=0.0)         # Minimize
    throughput_tps: float = Field(ge=0.0)  # Maximize
    accuracy_score: float = Field(ge=0.0, le=1.0)  # Maximize
    rank: int = 0
    crowding_distance: float = 0.0


class ParetoExperiment(BaseModel):
    """Complete record of an evolutionary Pareto frontier optimization run."""
    id: str = Field(default_factory=lambda: f"pareto-exp-{uuid.uuid4().hex[:8]}")
    name: str
    population_size: int = 16
    generations: int = 5
    pareto_front_size: int = 0
    status: str = "completed"
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParetoOptimizer:
    """
    Continuous Multi-Objective Evolutionary Optimizer implementing NSGA-II
    non-dominated sorting and crowding distance niching.
    """

    # Workstation Hardware Constraints (NVIDIA RTX 2000 Ada Generation)
    MAX_VRAM_MB = 8188.0  # 8 GB VRAM ceiling

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or get_db_path_from_url(get_settings().database_url))

    @classmethod
    def evaluate_config(cls, cfg: RuntimeConfig) -> tuple[float, float, float, float]:
        """
        Evaluate objective functions (latency_ms, vram_mb, throughput_tps, accuracy_score)
        based on exact transformer memory and compute formulas.
        """
        # 1. Weights Memory (7B model parameter baseline)
        weight_bytes = (7.0e9 * cfg.quant_bits) / 8.0
        weight_mb = weight_bytes / (1024.0 * 1024.0)

        # 2. KV-Cache Memory (28 layers, 8 kv heads, 128 head dim, FP16 = 2 bytes)
        kv_bytes = (2 * 28 * 8 * 128 * 2) * cfg.context_window * cfg.batch_size
        kv_mb = kv_bytes / (1024.0 * 1024.0)

        total_vram_mb = round(weight_mb + kv_mb + 450.0, 1)  # 450 MB CUDA context overhead

        # VRAM Thrashing Penalty (Spilling into system RAM)
        oom_penalty = 1.0
        if total_vram_mb > cls.MAX_VRAM_MB:
            oom_penalty = 6.0  # PCIe bus bandwidth bottleneck

        # 3. Latency (TTFT + inter-token ms)
        # Base latency proportional to model weight size and context length
        base_ttft = (cfg.context_window / 2048.0) * (cfg.quant_bits / 4.0) * 45.0
        spec_factor = 1.0 / (1.0 + (cfg.lookahead_k - 1) * 0.18)  # Speculative speedup
        thread_factor = 1.0 / (1.0 + math.log(cfg.thread_count) * 0.4)

        latency_ms = round((base_ttft + (spec_factor * 120.0)) * thread_factor * oom_penalty, 2)

        # 4. Throughput (Tokens per second)
        base_tps = (1000.0 / (18.0 * (cfg.quant_bits / 4.0))) * (1.0 + (cfg.lookahead_k - 1) * 0.15)
        batch_multiplier = math.pow(cfg.batch_size, 0.7)
        throughput_tps = round((base_tps * batch_multiplier) / (thread_factor * oom_penalty * 0.8), 2)

        # 5. Accuracy Score (Quantization fidelity degradation)
        acc_table = {16: 0.999, 8: 0.995, 4: 0.985, 3: 0.935, 2: 0.840}
        base_acc = acc_table.get(cfg.quant_bits, 0.950)
        # Context degradation for ultra-long context without fine-tuning
        if cfg.context_window > 8192:
            base_acc -= 0.04
        accuracy_score = round(min(1.0, max(0.1, base_acc)), 4)

        return latency_ms, total_vram_mb, throughput_tps, accuracy_score

    @staticmethod
    def dominates(p1: ParetoPoint, p2: ParetoPoint) -> bool:
        """
        Check if point p1 dominates point p2 (p1 > p2):
        p1 must be no worse than p2 on all 4 objectives, and strictly better on >= 1.
        Objectives: minimize latency, minimize vram, maximize throughput, maximize accuracy.
        """
        no_worse = (
            p1.latency_ms <= p2.latency_ms
            and p1.vram_mb <= p2.vram_mb
            and p1.throughput_tps >= p2.throughput_tps
            and p1.accuracy_score >= p2.accuracy_score
        )
        strictly_better = (
            p1.latency_ms < p2.latency_ms
            or p1.vram_mb < p2.vram_mb
            or p1.throughput_tps > p2.throughput_tps
            or p1.accuracy_score > p2.accuracy_score
        )
        return no_worse and strictly_better

    @classmethod
    def fast_non_dominated_sort(cls, population: list[ParetoPoint]) -> list[list[ParetoPoint]]:
        """
        NSGA-II Fast Non-Dominated Sorting algorithm.
        Partitions population into ranked Pareto fronts: [F_0, F_1, F_2, ...].
        """
        domination_counts: dict[str, int] = {}
        dominated_points: dict[str, list[ParetoPoint]] = {}
        fronts: list[list[ParetoPoint]] = [[]]

        for p in population:
            domination_counts[p.id] = 0
            dominated_points[p.id] = []

        for p in population:
            for q in population:
                if p.id == q.id:
                    continue
                if cls.dominates(p, q):
                    dominated_points[p.id].append(q)
                elif cls.dominates(q, p):
                    domination_counts[p.id] += 1

            if domination_counts[p.id] == 0:
                p.rank = 0
                fronts[0].append(p)

        current_rank = 0
        while current_rank < len(fronts) and len(fronts[current_rank]) > 0:
            next_front: list[ParetoPoint] = []
            for p in fronts[current_rank]:
                for q in dominated_points[p.id]:
                    domination_counts[q.id] -= 1
                    if domination_counts[q.id] == 0:
                        q.rank = current_rank + 1
                        next_front.append(q)
            current_rank += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    @staticmethod
    def calculate_crowding_distance(front: list[ParetoPoint]) -> None:
        """
        Calculate NSGA-II crowding distance to preserve diversity along the Pareto frontier.
        """
        if not front:
            return

        l = len(front)
        for p in front:
            p.crowding_distance = 0.0

        INF_DIST = 1_000_000.0
        if l <= 2:
            for p in front:
                p.crowding_distance = INF_DIST
            return

        objectives = ["latency_ms", "vram_mb", "throughput_tps", "accuracy_score"]

        for obj in objectives:
            front.sort(key=lambda p: getattr(p, obj))
            front[0].crowding_distance = INF_DIST
            front[-1].crowding_distance = INF_DIST

            obj_min = getattr(front[0], obj)
            obj_max = getattr(front[-1], obj)
            norm = obj_max - obj_min
            if norm == 0:
                continue

            for i in range(1, l - 1):
                if front[i].crowding_distance < INF_DIST:
                    dist = (getattr(front[i + 1], obj) - getattr(front[i - 1], obj)) / norm
                    front[i].crowding_distance += dist

    @classmethod
    def mutate_config(cls, cfg: RuntimeConfig) -> RuntimeConfig:
        """Perturb a runtime configuration with random genetic mutation."""
        c = copy.deepcopy(cfg)
        choice = random.choice(["quant", "context", "lookahead", "batch", "threads"])

        if choice == "quant":
            c.quant_bits = random.choice([2, 3, 4, 8, 16])
        elif choice == "context":
            c.context_window = random.choice([1024, 2048, 4096, 8192])
        elif choice == "lookahead":
            c.lookahead_k = random.randint(1, 8)
        elif choice == "batch":
            c.batch_size = random.choice([1, 2, 4])
        else:
            c.thread_count = random.choice([4, 8, 12, 16])

        return c

    def run_optimization(
        self,
        name: str = "Workstation-Pareto-Run",
        population_size: int = 16,
        generations: int = 4,
    ) -> tuple[ParetoExperiment, list[ParetoPoint]]:
        """
        Execute full continuous multi-objective Pareto evolutionary search.
        Persists experiment and frontier solutions in SQLite.
        """
        # Step 1: Initialize Random Population
        pop: list[ParetoPoint] = []
        seed_bits = [2, 3, 4, 8, 16]
        seed_contexts = [1024, 2048, 4096]

        for i in range(population_size):
            cfg = RuntimeConfig(
                quant_bits=seed_bits[i % len(seed_bits)],
                context_window=seed_contexts[(i // len(seed_bits)) % len(seed_contexts)],
                lookahead_k=random.randint(1, 6),
                batch_size=random.choice([1, 2, 4]),
                thread_count=random.choice([4, 8, 12]),
            )
            lat, vram, tps, acc = self.evaluate_config(cfg)
            pop.append(
                ParetoPoint(
                    config=cfg,
                    latency_ms=lat,
                    vram_mb=vram,
                    throughput_tps=tps,
                    accuracy_score=acc,
                )
            )

        # Step 2: Evolutionary Generations
        for gen in range(generations):
            fronts = self.fast_non_dominated_sort(pop)
            for f in fronts:
                self.calculate_crowding_distance(f)

            # Offspring generation via tournament selection & mutation
            offspring: list[ParetoPoint] = []
            while len(offspring) < population_size:
                p1, p2 = random.sample(pop, 2)
                parent = p1 if (p1.rank < p2.rank or (p1.rank == p2.rank and p1.crowding_distance > p2.crowding_distance)) else p2
                mut_cfg = self.mutate_config(parent.config)
                lat, vram, tps, acc = self.evaluate_config(mut_cfg)
                offspring.append(
                    ParetoPoint(
                        config=mut_cfg,
                        latency_ms=lat,
                        vram_mb=vram,
                        throughput_tps=tps,
                        accuracy_score=acc,
                    )
                )

            # Combined Population & Environmental Selection
            combined = pop + offspring
            combined_fronts = self.fast_non_dominated_sort(combined)
            for f in combined_fronts:
                self.calculate_crowding_distance(f)

            new_pop: list[ParetoPoint] = []
            for f in combined_fronts:
                if len(new_pop) + len(f) <= population_size:
                    new_pop.extend(f)
                else:
                    # Sort remaining by crowding distance descending
                    f.sort(key=lambda p: -p.crowding_distance)
                    new_pop.extend(f[: population_size - len(new_pop)])
                    break
            pop = new_pop

        # Final non-dominated sort
        final_fronts = self.fast_non_dominated_sort(pop)
        pareto_front = final_fronts[0]
        self.calculate_crowding_distance(pareto_front)

        exp_id = f"pareto-exp-{uuid.uuid4().hex[:8]}"
        experiment = ParetoExperiment(
            id=exp_id,
            name=name,
            population_size=population_size,
            generations=generations,
            pareto_front_size=len(pareto_front),
            metadata={
                "hardware_target": "NVIDIA RTX 2000 Ada",
                "max_vram_mb": self.MAX_VRAM_MB,
            },
        )

        # Step 3: Persist into SQLite
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO pareto_experiments (
                    id, name, population_size, generations, pareto_front_size, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.population_size,
                    experiment.generations,
                    experiment.pareto_front_size,
                    experiment.status,
                    json.dumps(experiment.metadata),
                ),
            )

            for p in pareto_front:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pareto_frontiers (
                        id, experiment_id, rank, config_json, latency_ms, vram_mb,
                        throughput_tps, accuracy_score, crowding_distance, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p.id,
                        experiment.id,
                        p.rank,
                        p.config.model_dump_json(),
                        p.latency_ms,
                        p.vram_mb,
                        p.throughput_tps,
                        p.accuracy_score,
                        0.0 if math.isinf(p.crowding_distance) else round(p.crowding_distance, 4),
                        json.dumps({"is_infinite_crowding": math.isinf(p.crowding_distance)}),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        return experiment, pareto_front

    def list_experiments(self, limit: int = 20) -> list[ParetoExperiment]:
        """List historical Pareto optimization runs."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, name, population_size, generations, pareto_front_size, status,
                       created_at, metadata
                FROM pareto_experiments
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            )
            experiments = []
            for row in cursor.fetchall():
                experiments.append(
                    ParetoExperiment(
                        id=row["id"],
                        name=row["name"],
                        population_size=row["population_size"],
                        generations=row["generations"],
                        pareto_front_size=row["pareto_front_size"],
                        status=row["status"],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return experiments
        finally:
            conn.close()

    def get_frontiers(self, experiment_id: str) -> list[ParetoPoint]:
        """Fetch discovered Pareto frontier points for an experiment."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, rank, config_json, latency_ms, vram_mb, throughput_tps,
                       accuracy_score, crowding_distance
                FROM pareto_frontiers
                WHERE experiment_id = ?
                ORDER BY rank ASC, accuracy_score DESC
                """,
                (experiment_id,),
            )
            points = []
            for row in cursor.fetchall():
                cfg_data = json.loads(row["config_json"])
                points.append(
                    ParetoPoint(
                        id=row["id"],
                        config=RuntimeConfig(**cfg_data),
                        latency_ms=row["latency_ms"],
                        vram_mb=row["vram_mb"],
                        throughput_tps=row["throughput_tps"],
                        accuracy_score=row["accuracy_score"],
                        rank=row["rank"],
                        crowding_distance=row["crowding_distance"],
                    )
                )
            return points
        finally:
            conn.close()
