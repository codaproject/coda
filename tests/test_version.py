import subprocess
from types import SimpleNamespace

import pytest

import coda


@pytest.fixture(autouse=True)
def fresh_version():
    coda.get_version.cache_clear()
    yield
    coda.get_version.cache_clear()


def fake_git(monkeypatch, returncode, stdout=""):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=returncode, stdout=stdout))


def test_version_from_tag(monkeypatch):
    fake_git(monkeypatch, 0, "v1.2.0-3-gab12cd3-dirty\n")
    assert coda.get_version() == "1.2.0-3-gab12cd3-dirty"


def test_version_without_tag(monkeypatch):
    fake_git(monkeypatch, 128)
    assert coda.get_version() == coda.__version__
