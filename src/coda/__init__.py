import logging

import pystow

# Configure logging
logging.basicConfig(format=('%(levelname)s: [%(asctime)s] %(name)s'
                            ' - %(message)s'),
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


CODA_BASE = pystow.module('coda')

__version__ = '0.1.0'