import logging

from coda.config import settings
from coda.inference.agent import InferenceServer
from coda.inference.champs_prompted import create_champs_prompted_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
server = InferenceServer(
    create_champs_prompted_agent(),
    host=settings.inference.host,
    port=settings.inference.port,
)
server.run()
