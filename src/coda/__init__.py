import logging
import subprocess
from functools import cache
from pathlib import Path

import pystow

# Configure logging
logging.basicConfig(format=('%(levelname)s: [%(asctime)s] %(name)s'
                            ' - %(message)s'),
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


CODA_BASE = pystow.module('coda')

__version__ = '1.0.0'


@cache
def get_commit():
    """Return the short git commit hash of this checkout, None outside git."""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short=6', 'HEAD'],
                             cwd=Path(__file__).parent, capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode:
        return None
    return out.stdout.strip()


def get_version_label():
    """Return the version with the commit hash, e.g. ``1.0.0 (ab12cd)``."""
    commit = get_commit()
    return f"{__version__} ({commit})" if commit else __version__
