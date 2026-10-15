"""
Cross-Modal Representation Alignment Subsystem (Phase 15).

Projects visual UI layouts, symbolic code AST structures, and natural language
reasoning traces into a unified normalized latent vector space for multi-modal
grounding and cross-modal retrieval.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection

log = logging.getLogger(__name__)


@dataclass
class ModalityEmbedding:
    """Normalized latent vector representation for a specific modality."""

    modality: str  # 'text', 'ast', 'visual'
    raw_repr: str
    tokens: list[str]
    vector: list[float]  # L2 normalized projection in R^d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentResult:
    """Multi-modal alignment assessment bridging text, AST, and visual representations."""

    id: str
    text_prompt: str
    ast_symbol: str
    visual_token: str
    text_ast_similarity: float
    ast_visual_similarity: float
    text_visual_similarity: float
    overall_alignment_score: float
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossModalAligner:
    """
    Unified projection engine that aligns symbolic code ASTs, textual reasoning,
    and visual inspection tokens into a joint latent space.
    """

    def __init__(self, db_path: str | Path | None = None, dim: int = 64) -> None:
        if db_path is None:
            self.db_path = get_db_path_from_url(get_settings().database_url)
        else:
            self.db_path = Path(db_path)
        self.dim = dim

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            return 0.0
        sim = dot / (norm1 * norm2)
        return max(-1.0, min(1.0, round(sim, 4)))

    @staticmethod
    def extract_text_tokens(text: str) -> list[str]:
        """Extract clean alphanumeric keywords from textual reasoning or requirements."""
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
        stopwords = {
            "the", "a", "an", "is", "in", "it", "to", "for", "with", "on", "at",
            "by", "and", "or", "of", "this", "that", "from", "as", "be", "are",
        }
        return [w for w in words if len(w) > 1 and w not in stopwords]

    @staticmethod
    def extract_ast_tokens(code_str: str) -> list[str]:
        """Extract structural AST node types and declared identifiers from code."""
        tokens: list[str] = []
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                node_type = type(node).__name__.lower()
                tokens.append(node_type)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    tokens.append(f"fn_{node.name.lower()}")
                    for part in node.name.lower().split("_"):
                        if len(part) > 1:
                            tokens.append(f"id_{part}")
                    for arg in node.args.args:
                        tokens.append(f"arg_{arg.arg.lower()}")
                        for part in arg.arg.lower().split("_"):
                            if len(part) > 1:
                                tokens.append(f"id_{part}")
                elif isinstance(node, ast.ClassDef):
                    tokens.append(f"class_{node.name.lower()}")
                    for part in node.name.lower().split("_"):
                        if len(part) > 1:
                            tokens.append(f"id_{part}")
                elif isinstance(node, ast.Name):
                    tokens.append(f"id_{node.id.lower()}")
                    for part in node.id.lower().split("_"):
                        if len(part) > 1:
                            tokens.append(f"id_{part}")
        except SyntaxError:
            # Fallback regex tokenization for incomplete or pseudo-code snippets
            raw = re.findall(r"[A-Za-z0-9]+", code_str.lower())
            tokens.extend([f"raw_{w}" for w in raw])
        return tokens

    @staticmethod
    def extract_visual_tokens(visual_repr: str | dict[str, Any]) -> list[str]:
        """Extract layout primitives, element tags, and visual styling cues."""
        tokens: list[str] = []
        if isinstance(visual_repr, dict):
            for k, v in visual_repr.items():
                tokens.append(f"ui_{k.lower()}")
                for part in str(k).lower().split("_"):
                    if len(part) > 1:
                        tokens.append(f"id_{part}")
                if isinstance(v, list):
                    for item in v:
                        tokens.append(f"ui_elem_{str(item).lower()}")
                        for part in str(item).lower().split("_"):
                            if len(part) > 1:
                                tokens.append(f"id_{part}")
                else:
                    tokens.append(f"ui_val_{str(v).lower()}")
                    for part in str(v).lower().split("_"):
                        if len(part) > 1:
                            tokens.append(f"id_{part}")
        else:
            raw = re.findall(r"[A-Za-z0-9]+", visual_repr.lower())
            for w in raw:
                tokens.append(f"vis_{w}")
                tokens.append(f"id_{w}")
        return tokens

    def project_tokens_to_latent(
        self, tokens: list[str], modality: str
    ) -> ModalityEmbedding:
        """
        Deterministically project token set into normalized R^d vector space
        using multi-hash basis embedding.
        """
        vec = [0.0] * self.dim

        if not tokens:
            vec[0] = 1.0
            return ModalityEmbedding(
                modality=modality,
                raw_repr="",
                tokens=[],
                vector=vec,
            )

        AST_BOILERPLATE = {
            "module", "load", "store", "del", "arguments", "arg", "call",
            "subscript", "attribute", "name", "expr", "pass", "constant",
        }

        # Multi-hash projection with semantic bridging
        for token in tokens:
            clean_token = re.sub(r"^(fn_|class_|arg_|id_|raw_|ui_|ui_elem_|ui_val_|vis_)", "", token)
            if clean_token in AST_BOILERPLATE:
                continue

            h_shared = int(hashlib.sha256(clean_token.encode()).hexdigest(), 16)
            idx_shared = h_shared % self.dim
            sign_shared = 1.0 if (h_shared & 1) else -1.0
            weight = 3.0 if token.startswith(("id_", "fn_", "class_", "arg_", "ui_elem_", "ui_val_")) else 1.0

            vec[idx_shared] += weight * sign_shared

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [round(x / norm, 6) for x in vec]
        else:
            vec[0] = 1.0

        return ModalityEmbedding(
            modality=modality,
            raw_repr=" ".join(tokens[:10]),
            tokens=tokens,
            vector=vec,
        )

    def align_modalities(
        self,
        text_prompt: str,
        ast_code: str,
        visual_spec: str | dict[str, Any],
        record_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AlignmentResult:
        """
        Embed all three modalities and compute cross-modal alignment scores.
        """
        aid = record_id or f"align-{uuid.uuid4().hex[:10]}"

        text_tokens = self.extract_text_tokens(text_prompt)
        ast_tokens = self.extract_ast_tokens(ast_code)
        vis_tokens = self.extract_visual_tokens(visual_spec)

        emb_text = self.project_tokens_to_latent(text_tokens, "text")
        emb_ast = self.project_tokens_to_latent(ast_tokens, "ast")
        emb_vis = self.project_tokens_to_latent(vis_tokens, "visual")

        sim_text_ast = self.cosine_similarity(emb_text.vector, emb_ast.vector)
        sim_ast_vis = self.cosine_similarity(emb_ast.vector, emb_vis.vector)
        sim_text_vis = self.cosine_similarity(emb_text.vector, emb_vis.vector)

        overall = round((sim_text_ast + sim_ast_vis + sim_text_vis) / 3.0, 4)

        vis_str = json.dumps(visual_spec) if isinstance(visual_spec, dict) else str(visual_spec)

        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO modal_alignments (
                    id, text_prompt, ast_symbol, visual_token, similarity_score, projection_dim, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    aid,
                    text_prompt,
                    ast_code[:200],
                    vis_str[:200],
                    overall,
                    self.dim,
                    json.dumps(
                        {
                            "text_ast": sim_text_ast,
                            "ast_vis": sim_ast_vis,
                            "text_vis": sim_text_vis,
                            **(metadata or {}),
                        }
                    ),
                ),
            )
            conn.commit()

            cursor = conn.execute("SELECT created_at FROM modal_alignments WHERE id = ?", (aid,))
            row = cursor.fetchone()
            created_at = row["created_at"] if row else ""

            return AlignmentResult(
                id=aid,
                text_prompt=text_prompt,
                ast_symbol=ast_code[:200],
                visual_token=vis_str[:200],
                text_ast_similarity=sim_text_ast,
                ast_visual_similarity=sim_ast_vis,
                text_visual_similarity=sim_text_vis,
                overall_alignment_score=overall,
                created_at=created_at,
                metadata=metadata or {},
            )
        finally:
            conn.close()

    def cross_modal_retrieve(
        self,
        query_modality: str,
        query_repr: str,
        candidates: list[tuple[str, str]],  # list of (modality, content)
    ) -> list[tuple[str, str, float]]:
        """
        Rank candidate artifacts in other modalities against the query.
        Returns list of (modality, content, cosine_sim) sorted descending by similarity.
        """
        if query_modality == "text":
            q_tokens = self.extract_text_tokens(query_repr)
        elif query_modality == "ast":
            q_tokens = self.extract_ast_tokens(query_repr)
        else:
            q_tokens = self.extract_visual_tokens(query_repr)

        q_emb = self.project_tokens_to_latent(q_tokens, query_modality)

        ranked: list[tuple[str, str, float]] = []
        for cand_mod, cand_content in candidates:
            if cand_mod == "text":
                c_tokens = self.extract_text_tokens(cand_content)
            elif cand_mod == "ast":
                c_tokens = self.extract_ast_tokens(cand_content)
            else:
                c_tokens = self.extract_visual_tokens(cand_content)

            c_emb = self.project_tokens_to_latent(c_tokens, cand_mod)
            sim = self.cosine_similarity(q_emb.vector, c_emb.vector)
            ranked.append((cand_mod, cand_content, sim))

        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    def list_alignments(self, limit: int = 50) -> list[AlignmentResult]:
        """List alignment records ordered by alignment score."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, text_prompt, ast_symbol, visual_token, similarity_score, created_at, metadata
                FROM modal_alignments ORDER BY similarity_score DESC LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                meta = json.loads(row["metadata"] or "{}")
                results.append(
                    AlignmentResult(
                        id=row["id"],
                        text_prompt=row["text_prompt"],
                        ast_symbol=row["ast_symbol"],
                        visual_token=row["visual_token"],
                        text_ast_similarity=float(meta.get("text_ast", row["similarity_score"])),
                        ast_visual_similarity=float(meta.get("ast_vis", row["similarity_score"])),
                        text_visual_similarity=float(meta.get("text_vis", row["similarity_score"])),
                        overall_alignment_score=float(row["similarity_score"]),
                        created_at=row["created_at"],
                        metadata=meta,
                    )
                )
            return results
        finally:
            conn.close()
