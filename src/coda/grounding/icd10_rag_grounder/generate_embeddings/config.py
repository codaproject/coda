"""Configuration for ICD-10 embedding generation."""

from pathlib import Path

# Default paths
DEFAULT_ICD10_ZIP = Path.home() / '.data' / 'openacme' / 'icd10' / 'icd102019en.xml.zip'
DEFAULT_MRCONSO_PATH = Path('/Users/thomaslim/Downloads/2025AB/META/MRCONSO.RRF')
DEFAULT_MRDEF_PATH = Path('/Users/thomaslim/Downloads/2025AB/META/MRDEF.RRF')

# Source priority for definitions
SOURCE_PRIORITY = ['MSH', 'CSP', 'NCI', 'HPO', 'SNOMEDCT_US', 'MEDLINEPLUS']

"""Configuration for ICD-10 processing."""

# Default model
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

# Default batch size
DEFAULT_BATCH_SIZE = 32
