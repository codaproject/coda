from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTUP_SH = ROOT / "startup.sh"
RUNTIME_ENV_VARS = (
    "APP_HOST",
    "APP_PORT",
    "INFERENCE_HOST",
    "INFERENCE_PORT",
    "INFERENCE_URL",
    "CODA_INFERENCE_URL",
    "INFERENCE_LLM_PROVIDER",
    "INFERENCE_LLM_MODEL",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "CODA_KG_URL",
    "RAG_LLM_PROVIDER",
    "RAG_LLM_MODEL",
    "RAG_ONTOLOGY",
    "RAG_USE_RERANKER",
    "CODA_DEVICE",
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_startup_script(tmp_path: Path, env_contents: str, extra_env: dict[str, str] | None = None):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    shutil.copy2(STARTUP_SH, workdir / "startup.sh")
    (workdir / ".env").write_text(env_contents, encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "startup.log"

    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'python|%s|APP_PORT=%s|INFERENCE_PORT=%s|INFERENCE_URL=%s|PROVIDER=%s|MODEL=%s\\n' \
  "$*" "${APP_PORT:-}" "${INFERENCE_PORT:-}" "${INFERENCE_URL:-}" \
  "${INFERENCE_LLM_PROVIDER:-}" "${INFERENCE_LLM_MODEL:-}" >> "${TEST_LOG}"
printf 'runtime|OLLAMA_BASE_URL=%s|CODA_KG_URL=%s|RAG_PROVIDER=%s|RAG_MODEL=%s|RAG_ONTOLOGY=%s|RAG_RERANKER=%s|CODA_DEVICE=%s\\n' \
  "${OLLAMA_BASE_URL:-}" "${CODA_KG_URL:-}" "${RAG_LLM_PROVIDER:-}" \
  "${RAG_LLM_MODEL:-}" "${RAG_ONTOLOGY:-}" "${RAG_USE_RERANKER:-}" \
  "${CODA_DEVICE:-}" >> "${TEST_LOG}"
exit 0
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
last="${@: -1}"
printf 'curl|%s\\n' "${last}" >> "${TEST_LOG}"
exit 0
""",
    )
    _write_executable(
        fake_bin / "sleep",
        """#!/usr/bin/env bash
exit 0
""",
    )

    env = os.environ.copy()
    for env_var in RUNTIME_ENV_VARS:
        env.pop(env_var, None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TEST_LOG"] = str(log_path)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", "startup.sh"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    return result, lines


def test_startup_script_uses_env_contract_and_derived_health_urls(tmp_path):
    result, lines = _run_startup_script(
        tmp_path,
        "\n".join(
            [
                "APP_HOST=127.0.0.2",
                "APP_PORT=8100",
                "INFERENCE_HOST=127.0.0.3",
                "INFERENCE_PORT=6123",
                "INFERENCE_LLM_PROVIDER=ollama",
                "INFERENCE_LLM_MODEL=llamaX.X",
                "",
            ]
        ),
    )

    python_lines = [line for line in lines if line.startswith("python|")]
    curl_lines = [line for line in lines if line.startswith("curl|")]

    assert len(python_lines) == 2
    assert python_lines[0].startswith(
        "python|-m coda.inference.agent|APP_PORT=8100|INFERENCE_PORT=6123|"
        "INFERENCE_URL=http://127.0.0.1:6123|PROVIDER=ollama|"
        "MODEL=llamaX.X"
    )
    assert "--provider" not in python_lines[0]
    assert "--model" not in python_lines[0]
    assert python_lines[1].startswith(
        "python|-m coda.app|APP_PORT=8100|INFERENCE_PORT=6123|"
        "INFERENCE_URL=http://127.0.0.1:6123|PROVIDER=ollama|"
        "MODEL=llamaX.X"
    )

    assert "curl|http://127.0.0.3:6123/health" in curl_lines
    assert "curl|http://127.0.0.2:8100/health" in curl_lines
    assert "CODA is running at http://localhost:8100" in result.stdout


def test_startup_script_allows_shell_env_to_override_dotenv(tmp_path):
    result, lines = _run_startup_script(
        tmp_path,
        "\n".join(
            [
                "APP_PORT=8000",
                "INFERENCE_PORT=5123",
                "",
            ]
        ),
        extra_env={
            "APP_PORT": "9100",
            "INFERENCE_PORT": "7123",
            "INFERENCE_URL": "http://127.0.0.1:7123",
        },
    )

    python_lines = [line for line in lines if line.startswith("python|")]
    curl_lines = [line for line in lines if line.startswith("curl|")]

    assert python_lines[0].startswith(
        "python|-m coda.inference.agent|APP_PORT=9100|INFERENCE_PORT=7123|"
        "INFERENCE_URL=http://127.0.0.1:7123"
    )
    assert python_lines[1].startswith(
        "python|-m coda.app|APP_PORT=9100|INFERENCE_PORT=7123|"
        "INFERENCE_URL=http://127.0.0.1:7123"
    )
    assert "curl|http://127.0.0.1:7123/health" in curl_lines
    assert "curl|http://127.0.0.1:9100/health" in curl_lines
    assert "CODA is running at http://localhost:9100" in result.stdout


def test_startup_script_passes_app_runtime_env_from_dotenv(tmp_path):
    _, lines = _run_startup_script(
        tmp_path,
        "\n".join(
            [
                "OLLAMA_BASE_URL=http://ollama.internal:11434",
                "CODA_KG_URL=bolt://neo4j.internal:7687",
                "RAG_LLM_PROVIDER=ollama",
                "RAG_LLM_MODEL=llama3.2",
                "RAG_ONTOLOGY=icd11",
                "RAG_USE_RERANKER=false",
                "CODA_DEVICE=cuda",
                "",
            ]
        ),
    )

    runtime_lines = [line for line in lines if line.startswith("runtime|")]
    assert len(runtime_lines) == 2
    assert all(
        line == (
            "runtime|OLLAMA_BASE_URL=http://ollama.internal:11434|"
            "CODA_KG_URL=bolt://neo4j.internal:7687|RAG_PROVIDER=ollama|"
            "RAG_MODEL=llama3.2|RAG_ONTOLOGY=icd11|RAG_RERANKER=false|"
            "CODA_DEVICE=cuda"
        )
        for line in runtime_lines
    )


def test_startup_script_supports_legacy_inference_url(tmp_path):
    _, lines = _run_startup_script(
        tmp_path,
        "\n".join(
            [
                "CODA_INFERENCE_URL=http://legacy-inference.internal:5123",
                "",
            ]
        ),
    )

    python_lines = [line for line in lines if line.startswith("python|")]
    assert all(
        "INFERENCE_URL=http://legacy-inference.internal:5123" in line
        for line in python_lines
    )
