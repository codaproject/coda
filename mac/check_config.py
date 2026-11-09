"""Check CODA's settings and print the values the Mac launcher needs.

Prints the app port, whether TLS is on, the setup page port and the
inference port on one line. On a problem, prints it and exits with status 1.
"""
import os
import sys
from pathlib import Path

import yaml


def unknown_keys(values, defaults, prefix=""):
    """Yield the dotted keys of values that defaults does not have."""
    for key, value in values.items():
        if key not in defaults:
            yield prefix + str(key)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            yield from unknown_keys(value, defaults[key], f"{prefix}{key}.")


def main():
    try:
        from coda.config import CONFIG_DIR, settings, site_profile, \
            usecase_defaults
        settings.validators.validate()
        defaults = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text())
        files = [Path(os.environ["CODA_SETTINGS_FILE"])]
        if settings.site:
            files.insert(0, site_profile(settings.site, settings.usecase))
        if settings.site and settings.usecase:
            files.insert(0, usecase_defaults(settings.usecase))
        for path in files:
            values = yaml.safe_load(path.read_text()) or {}
            for key in unknown_keys(values, defaults):
                raise ValueError(f"Unknown setting {key} in {path}")
    except Exception as exc:
        sys.exit(str(exc))
    print(settings.app.port, settings.app.tls, settings.app.setup_port,
          settings.inference.port)


if __name__ == "__main__":
    main()
