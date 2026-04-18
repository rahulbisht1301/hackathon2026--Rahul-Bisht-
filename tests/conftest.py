from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.config import settings
from agent.data.loader import init_loader
from agent.data.vector_store import init_vector_store
from agent.tools.failures import simulator


@pytest.fixture(autouse=True)
def reset_failure_injection(monkeypatch, project_root: Path):
    monkeypatch.setattr(settings, "tool_failure_rate", 0.0, raising=False)
    monkeypatch.setattr(settings, "policy_reference_date", "2024-03-15", raising=False)
    settings.data_dir = str(project_root / "data")
    init_loader(settings.data_dir)
    simulator.reset()
    yield
    simulator.reset()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def init_kb(project_root: Path):
    settings.data_dir = str(project_root / "data")
    asyncio.run(init_vector_store())

