import subprocess
from types import SimpleNamespace

import pytest

import coda


@pytest.fixture(autouse=True)
def fresh_version():
    coda.get_version.cache_clear()
    yield
    coda.get_version.cache_clear()


def version_for(monkeypatch, returncode, described):
    monkeypatch.setattr(coda, "__version__", "1.1.0")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=returncode,
                                        stdout=described + "\n"))
    return coda.get_version()


def test_version(monkeypatch):
    assert version_for(monkeypatch, 0, "1.1.0") == "1.1.0"


def test_version_off_release(monkeypatch):
    assert version_for(monkeypatch, 0, "1.1.0-3-gab12cd3-dirty") == \
        "1.1.0+3-gab12cd3-dirty"
    coda.get_version.cache_clear()
    assert version_for(monkeypatch, 0, "ab12cd3") == "1.1.0+ab12cd3"


def test_version_without_git(monkeypatch):
    assert version_for(monkeypatch, 128, "") == "1.1.0"
