import logging
import os
import threading

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

    Gilda's SQLite adapter caches a single connection, which is bound to
    the thread that first triggers it.  To avoid cross-thread errors the
    grounder is created lazily on the first call to ``ground`` or
    ``annotate``, so it binds to whichever thread actually uses it (e.g.
    a dedicated executor thread).
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

        self._db_path = db_path
        self._grounder = None
        self._lock = threading.Lock()

    def _get_grounder(self):
        if self._grounder is None:
            with self._lock:
                if self._grounder is None:
                    if self._db_path and os.path.isfile(self._db_path):
                        logger.info("Loading Gilda grounder from sqlite: %s",
                                    self._db_path)
                        self._grounder = Grounder(self._db_path)
                    else:
                        logger.info("Loading Gilda grounder from default terms")
                        self._grounder = Grounder()
        return self._grounder

    def ground(self, text: str) -> list:
        return self._get_grounder().ground(text, namespaces=self.namespaces)

    def annotate(self, text: str) -> list:
        return gilda.ner.annotate(text, grounder=self._get_grounder(),
                                  namespaces=self.namespaces)
