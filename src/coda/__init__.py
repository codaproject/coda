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
def get_version():
    """Return the git-described version of this checkout, else __version__.

    On a release tag this is e.g. ``1.0.0``, past it ``1.0.0-3-gab12cd3``,
    with ``-dirty`` appended when tracked files have local changes.
    """
    try:
        out = subprocess.run(
            ['git', 'describe', '--tags', '--match', 'v[0-9]*', '--dirty'],
            cwd=Path(__file__).parent, capture_output=True, text=True,
            timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return __version__
    if out.returncode:
        return __version__
    return out.stdout.strip().removeprefix('v')
