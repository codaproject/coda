import uvicorn

from .server import app
from coda.config import settings

if __name__ == "__main__":
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)
