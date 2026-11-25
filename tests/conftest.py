"""Test fixtures. Nothing here touches a model, a network, or an API key."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages"))

from app import db  # noqa: E402
from app.config import settings as base_settings  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the metadata layer at a throwaway SQLite file.

    Settings is frozen, so the module-level binding is swapped rather than
    mutated -- dataclasses.replace is the only way to vary it safely.
    """
    monkeypatch.setattr(db, "settings", replace(base_settings, db_path=str(tmp_path / "t.db")))
    db.init()
    return tmp_path
