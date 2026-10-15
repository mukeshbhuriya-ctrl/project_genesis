"""
Tests for Cross-Modal Representation Alignment Subsystem (Phase 15).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.world_model.alignment import CrossModalAligner


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "align_test.db"
    initialize_database(db_file)
    return db_file


def test_token_extraction():
    """Verify modality token extractors for text, AST, and visual representations."""
    aligner = CrossModalAligner()

    # Text
    text = "Build an interactive memory inspector dashboard table"
    text_tokens = aligner.extract_text_tokens(text)
    assert "interactive" in text_tokens
    assert "memory" in text_tokens
    assert "inspector" in text_tokens
    assert "the" not in text_tokens

    # AST
    code = "class MemoryTable:\n    def render(self, rows):\n        pass\n"
    ast_tokens = aligner.extract_ast_tokens(code)
    assert "class_memorytable" in ast_tokens
    assert "fn_render" in ast_tokens
    assert "arg_rows" in ast_tokens

    # Visual spec
    vis = {"component": "DataTable", "columns": ["id", "confidence", "content"]}
    vis_tokens = aligner.extract_visual_tokens(vis)
    assert "ui_component" in vis_tokens
    assert "ui_elem_id" in vis_tokens


def test_project_tokens_and_cosine_similarity():
    """Verify latent projection is normalized and cosine similarity works correctly."""
    aligner = CrossModalAligner(dim=32)

    emb1 = aligner.project_tokens_to_latent(["memory", "table"], "text")
    assert len(emb1.vector) == 32
    # Verify L2 norm == 1.0
    norm = sum(x * x for x in emb1.vector)
    assert abs(norm - 1.0) < 1e-4

    # Identical tokens yield identical vectors
    emb2 = aligner.project_tokens_to_latent(["memory", "table"], "text")
    assert aligner.cosine_similarity(emb1.vector, emb2.vector) >= 0.999


def test_align_modalities_and_persistence(test_db: Path):
    """Verify end-to-end multi-modal alignment and SQLite storage."""
    aligner = CrossModalAligner(db_path=test_db, dim=64)

    prompt = "Create a hardware VRAM consumption chart monitor"
    code = "def render_vram_chart(gpu_stats):\n    return chart_element\n"
    visual = {"layout": "Card", "chart_type": "line", "metric": "vram_mb"}

    res = aligner.align_modalities(
        text_prompt=prompt,
        ast_code=code,
        visual_spec=visual,
        record_id="align-001",
    )

    assert res.id == "align-001"
    assert -1.0 <= res.overall_alignment_score <= 1.0
    assert -1.0 <= res.text_ast_similarity <= 1.0
    assert -1.0 <= res.ast_visual_similarity <= 1.0
    assert -1.0 <= res.text_visual_similarity <= 1.0

    listed = aligner.list_alignments(limit=10)
    assert len(listed) == 1
    assert listed[0].id == "align-001"


def test_cross_modal_retrieve(test_db: Path):
    """Verify cross-modal retrieval ranks semantically related artifacts highest."""
    aligner = CrossModalAligner(db_path=test_db, dim=64)

    query_text = "Vector memory embedding store"

    candidates = [
        ("ast", "def search_vector_memory(embedding: list[float]):\n    return db.query()\n"),
        ("visual", {"type": "NavBar", "items": ["Home", "Settings"]}),
        ("ast", "def compute_fft(signal: list[float]):\n    return transform(signal)\n"),
    ]

    ranked = aligner.cross_modal_retrieve("text", query_text, candidates)
    assert len(ranked) == 3
    # Top candidate should be the memory search AST function
    top_mod, top_content, top_score = ranked[0]
    assert top_mod == "ast"
    assert "search_vector_memory" in top_content
