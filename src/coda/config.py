"""Global CODA configuration.

Single source of truth for all runtime configuration, backed by Dynaconf and
the YAML files under the repo-root ``config/`` directory:

    config/settings.yaml    modular config, one namespace per module
    config/.secrets.yaml    API keys (gitignored)
    config/prompts/*.yaml   RAG prompt bodies, addressed by key

Read configuration directly off the module-level ``settings`` object using
dotted access, e.g. ``settings.app.host``, ``settings.grounder.rag.retriever
.ontology``.

Override any value at runtime with a ``CODA_``-prefixed environment variable,
using ``__`` to descend into nested keys (``CODA_APP__PORT=9000``).
"""

import os
from pathlib import Path

import yaml
from dynaconf import Dynaconf, Validator


def _resolve_config_dir() -> Path:
    """Locate the repo-root ``config/`` directory.

    Works both from a source checkout (where this file is ``<root>/src/coda/
    config.py``) and from an installed package (where ``__file__`` is under
    site-packages but the app is launched from the repo root / Docker WORKDIR).
    ``CODA_CONFIG_DIR`` overrides the search entirely.
    """
    override = os.getenv("CODA_CONFIG_DIR")
    if override:
        return Path(override)
    # Source/src layout: <root>/config alongside <root>/src/coda/config.py.
    src_layout = Path(__file__).resolve().parents[2] / "config"
    if (src_layout / "settings.yaml").exists():
        return src_layout
    # Installed layout: launched from the repo root / Docker WORKDIR (/app).
    cwd_layout = Path.cwd() / "config"
    if (cwd_layout / "settings.yaml").exists():
        return cwd_layout
    return src_layout


CONFIG_DIR = _resolve_config_dir()
PROMPTS_DIR = CONFIG_DIR / "prompts"

settings = Dynaconf(
    root_path=str(CONFIG_DIR),
    settings_files=["settings.yaml", ".secrets.yaml"],
    envvar_prefix="CODA",       # `CODA_APP__PORT=9000` overrides app.port
    merge_enabled=True,         # deep-merge .secrets.yaml / env-vars over settings.yaml
    load_dotenv=True,
    validators=[
        # Coerce env-var strings (which always arrive as text) to the right type.
        Validator("app.port", "inference.port", cast=int),
        Validator("grounder.rag.retriever.top_k", cast=int),
        Validator("grounder.rag.retriever.min_similarity", cast=float),
        Validator("grounder.rag.reranker.enabled", cast=bool),
    ],
)


def _load_prompts() -> dict:
    """Load every ``config/prompts/*.yaml`` file, keyed by filename stem.

    Prompt bodies are static data (no environment layering or env-var overrides),
    so they are read as plain dicts and exposed via ``PROMPTS`` rather than the
    Dynaconf store (which would drop them on ``settings.reload()``). Look one up
    by its config key, e.g. ``PROMPTS[settings.grounder.rag.extractor.prompt]``.
    """
    prompts = {}
    for path in sorted(PROMPTS_DIR.glob("*.yaml")):
        with open(path) as f:
            prompts[path.stem] = yaml.safe_load(f)
    return prompts


# Prompt configs keyed by filename stem (e.g. "extractor_default").
PROMPTS = _load_prompts()


def inference_url() -> str:
    """Resolve the inference service URL, deriving one from host/port if unset."""
    url = (settings.inference.url or "").strip()
    if url:
        return url
    return f"http://127.0.0.1:{settings.inference.port}"
