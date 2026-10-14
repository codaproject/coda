import uvicorn

from coda.config import configure_logging, settings

if __name__ == "__main__":
    configure_logging()
    from .server import app

    ssl_kwargs = {}
    if settings.app.tls:
        from .tls import ensure_certs, start_setup_server

        material = ensure_certs(settings.app.tls_dir)
        start_setup_server(
            material.ca_cert,
            host=settings.app.host,
            port=settings.app.setup_port,
            https_port=settings.app.port,
        )
        ssl_kwargs = {
            "ssl_certfile": str(material.leaf_cert),
            "ssl_keyfile": str(material.leaf_key),
        }
    uvicorn.run(
        app,
        host=settings.app.host,
        port=settings.app.port,
        log_config=None,
        **ssl_kwargs,
    )
