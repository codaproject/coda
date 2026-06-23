import uvicorn

from .server import app
from coda.runtime_config import get_app_host, get_app_port

if __name__ == "__main__":
    uvicorn.run(app, host=get_app_host(), port=get_app_port())
