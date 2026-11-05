import subprocess
from types import SimpleNamespace

import pytest

import coda


@pytest.fixture(autouse=True)
def fresh_commit():
    coda.get_commit.cache_clear()
    yield
    coda.get_commit.cache_clear()


def label_for(monkeypatch, returncode, stdout):
    monkeypatch.setattr(coda, "__version__", "1.1.0")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=returncode, stdout=stdout))
    return coda.get_version_label()


def test_version_label(monkeypatch):
    assert label_for(monkeypatch, 0, "ab12cd\n") == "1.1.0 (ab12cd)"


def test_version_label_without_git(monkeypatch):
    assert label_for(monkeypatch, 128, "") == "1.1.0"
