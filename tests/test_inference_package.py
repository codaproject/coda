import subprocess
import sys


def test_importing_inference_package_does_not_preload_agent():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import coda.inference; "
                "assert 'coda.inference.agent' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_inference_agent_module_runs_without_runpy_warning():
    result = subprocess.run(
        [sys.executable, "-W", "error", "-m", "coda.inference.agent", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
