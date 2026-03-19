import logging
import os

import gilda
import gilda.ner
from gilda.grounder import Grounder

from . import BaseGrounder

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACES = ['MESH', 'DOID', 'HP']


class GildaGrounder(BaseGrounder):
    """Wrapper for using Gilda as the grounding system.

    If ``db_path`` points to an existing sqlite database, the grounder
    uses it for near-instant startup. Otherwise it falls back to
    loading the default TSV terms into memory.
    """

    def __init__(self, namespaces=None, db_path=None):
        super().__init__()
        self.namespaces = DEFAULT_NAMESPACES \
            if namespaces is None else namespaces

        if not db_path:
            db_path = os.environ.get("GILDA_SQLITE_DB")
        if not db_path:
            from gilda.resources import resource_dir
            default_db = os.path.join(resource_dir, "grounding_terms.db")
            if os.path.isfile(default_db):
                db_path = default_db

        if db_path and os.path.isfile(db_path):
            logger.info("Loading Gilda grounder from sqlite: %s", db_path)
            self._grounder = Grounder(db_path)
        else:
            logger.info("Loading Gilda grounder from default terms")
            self._grounder = Grounder()

    def ground(self, text: str) -> list:
        return self._grounder.ground(text, namespaces=self.namespaces)

    def annotate(self, text: str) -> list:
        return gilda.ner.annotate(text, grounder=self._grounder,
                                  namespaces=self.namespaces)
