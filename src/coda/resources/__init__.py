__all__ = ['get_resource_path']

import os

HERE = os.path.dirname(os.path.abspath(__file__))

def get_resource_path(filename: str) -> str:
    """Get the absolute path to a resource file."""
    return os.path.join(HERE, filename)