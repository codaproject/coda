"""Global CODA configuration.

Single source of truth for all runtime configuration, backed by Dynaconf and
the YAML files under the repo-root ``config/`` directory:

    config/settings.yaml    modular config, one namespace per module
    config/.secrets.yaml    API keys (gitignored)

Read configuration directly off the module-level ``settings`` object using
dotted access, e.g. ``settings.app.host``, ``settings.grounder.rag.retriever
.ontology``.

Override any value at runtime with a ``CODA_``-prefixed environment variable,
using ``__`` to descend into nested keys (``CODA_APP__PORT=9000``).
"""

import logging
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


def site_profile(site, usecase):
    """Return the path of the profile for a site and use case."""
    name = f"{site}-{usecase}" if usecase else site
    return CONFIG_DIR / "sites" / f"{name}.yaml"


def usecase_defaults(usecase):
    """Return the path of the settings shared by a use case's profiles."""
    return CONFIG_DIR / "usecases" / f"{usecase}.yaml"


def _settings_files():
    """Return the settings files to load, in increasing precedence.

    ``CODA_SETTINGS_FILE`` names a machine's own settings file. Its ``site``
    key, or ``CODA_SITE`` when set, and ``CODA_USECASE`` select a site
    profile in ``config/sites/``, which the machine's file overrides. The
    profile overrides the settings its use case shares in
    ``config/usecases/``.
    """
    files = ["settings.yaml"]
    local = os.getenv("CODA_SETTINGS_FILE")
    if local and Path(local).exists():
        site = os.getenv("CODA_SITE") or \
            (yaml.safe_load(Path(local).read_text()) or {}).get("site")
        if site:
            usecase = os.getenv("CODA_USECASE")
            profile = site_profile(site, usecase)
            if not profile.exists():
                raise ValueError(f"No site profile {profile}")
            if usecase and usecase_defaults(usecase).exists():
                files.append(str(usecase_defaults(usecase)))
            files.append(str(profile))
        files.append(local)
    return files + [".secrets.yaml"]


settings = Dynaconf(
    root_path=str(CONFIG_DIR),
    settings_files=_settings_files(),
    envvar_prefix="CODA",       # `CODA_APP__PORT=9000` overrides app.port
    merge_enabled=True,         # deep-merge .secrets.yaml / env-vars over settings.yaml
    load_dotenv=True,
    validators=[
        # Coerce env-var strings (which always arrive as text) to the right type.
        Validator("app.port", "app.setup_port", "inference.port", cast=int),
        Validator("app.tls", cast=bool),
        Validator("inference.agent",
                  is_in=["champs_prompted", "champs_finetuned"]),
        Validator("inference.llm.provider", "grounder.rag.llm.provider",
                  is_in=["openai", "ollama"]),
        Validator("inference.question_style", is_in=["default", "simplified"]),
        Validator("inference.champs_prompted.cohort",
                  is_in=["child", "adult"]),
        Validator("inference.question_selection",
                  is_in=["llm", "voi", "baq"]),
        Validator("app.wording", is_in=["clinical", "interview"]),
        Validator("grounder.type", is_in=["gilda", "rag"]),
        Validator("inference.num_questions", cast=int),
        Validator("inference.question_skip_logic", cast=bool),
        Validator("grounder.rag.retriever.top_k", cast=int),
        Validator("grounder.rag.retriever.min_similarity", cast=float),
        Validator("grounder.rag.reranker.enabled", cast=bool),
        Validator(
            "storage.enabled",
            "storage.browse",
            "storage.store.audio",
            "storage.store.transcripts",
            "storage.store.annotations",
            "storage.store.inference",
            "storage.store.metadata",
            "storage.store.timing",
            "storage.store.notes",
            "storage.store.user",
            cast=bool,
        ),
        # And the other direction: keep these text. Dynaconf infers types, so a
        # date-shaped consent version (`2026-08-06`) would otherwise arrive as a
        # `datetime.date` and break the JSON-encoded version the page embeds.
        Validator(
            "app.onboarding_notice.file",
            "app.onboarding_notice.version",
            "app.tls_dir",
            "storage.output_dir",
            "logging.file",
            cast=str,
        ),
    ],
)


def reload_settings() -> None:
    """Re-read the settings files and re-apply the validators.

    ``settings.reload()`` on its own reloads the raw data but does **not** re-run
    validation, so every ``cast=`` above is silently lost and values fall back to
    whatever type Dynaconf infers from the env-var text. Use this instead of
    ``settings.reload()`` whenever config is reloaded mid-process (which today is
    only the tests) so the result matches what a fresh process would see.
    """
    settings.reload()
    settings.validators.validate()


def inference_url() -> str:
    """Resolve the inference service URL, deriving one from host/port if unset."""
    url = (settings.inference.url or "").strip()
    if url:
        return url
    return f"http://127.0.0.1:{settings.inference.port}"


LOG_FORMAT = '%(levelname)s: [%(asctime)s] %(name)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def configure_logging() -> None:
    """Configure the root logger from the ``logging`` settings section.

    Logs always go to stderr, and additionally to ``logging.file`` when set.
    Replaces any handlers installed earlier (e.g. the import-time default).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = settings.logging.get("file", "")
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        level=str(settings.logging.get("level", "INFO")).upper(),
        handlers=handlers,
        force=True,
    )
