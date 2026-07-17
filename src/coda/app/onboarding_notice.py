"""Optional onboarding notice injection for deployment-specific HTML."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NOTICE_PLACEHOLDER = "<!-- CODA_ONBOARDING_NOTICE -->"
VERSION_PLACEHOLDER = "__CODA_ONBOARDING_NOTICE_VERSION__"


def load_onboarding_notice_html(enabled: bool, file_path: str) -> str:
    """Return trusted admin-authored notice HTML, or empty string if disabled.

    The notice file is intended for deployment-owned content such as demo terms.
    It is inserted as HTML and should not contain user-generated input.
    """
    if not enabled:
        return ""
    if not file_path.strip():
        logger.warning(
            "CODA_ONBOARDING_NOTICE_ENABLED is true but "
            "CODA_ONBOARDING_NOTICE_FILE is unset"
        )
        return ""

    path = Path(file_path)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read onboarding notice file %s: %s", path, e)
        return ""


def render_onboarding_notice(
        html: str,
        *,
        notice_html: str = "",
        notice_version: str = "default",
) -> str:
    """Inject optional onboarding notice HTML and consent version."""
    html = html.replace(NOTICE_PLACEHOLDER, notice_html)
    return html.replace(VERSION_PLACEHOLDER, json.dumps(notice_version))
