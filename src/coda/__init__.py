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
    """Return __version__, marked when the git checkout is not its release.

    On the clean release tag this is e.g. ``1.0.0``. Otherwise git's
    description is appended, e.g. ``1.0.0+3-gab12cd3`` three commits past the
    tag, ``1.0.0+dirty`` with local changes, or ``1.1.0+ab12cd3`` before the
    tag exists. Outside a git checkout it is ``__version__``.
    """
    try:
        out = subprocess.run(
            ['git', 'describe', '--tags', '--always', '--dirty', '--match',
             __version__], cwd=Path(__file__).parent, capture_output=True,
            text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return __version__
    described = out.stdout.strip()
    if out.returncode or described == __version__:
        return __version__
    return f"{__version__}+{described.removeprefix(__version__ + '-')}"
