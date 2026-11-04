CODA on a Mac
=============

`CODA.app` starts CODA natively on an Apple Silicon Mac, with MLX
transcription and a local Ollama model. `Update CODA.app` moves the laptop to
the newest CODA release. Both open a Terminal window that shows progress, and
closing the CODA window stops CODA.

Installation
------------

1. Install the Xcode command line tools, which provide git:
   `xcode-select --install`
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install the [Ollama app](https://ollama.com/download/mac) and open it once.
4. Check out the newest CODA release and put both apps on the Desktop:

   ```bash
   git clone https://github.com/codaproject/coda.git ~/coda
   cd ~/coda
   git checkout "$(git tag -l 'v[0-9]*' --sort=-v:refname | grep -v -- - | head -1)"
   ln -s ~/coda/mac/CODA.app ~/coda/mac/"Update CODA.app" ~/Desktop/
   ```

5. Put site settings in `~/coda/.env` (see `.env.example`), for example
   `CODA_DIALOGUE__LANGUAGE` or `CODA_STORAGE__ENABLED`. Settings in `.env`
   take precedence over the Mac defaults in `mac/defaults.env`.
6. Start CODA once while online. The first start installs the Python
   packages, downloads the models, prepares the grounding data, and asks to
   trust CODA's local certificate authority so the browser accepts
   `https://localhost:8000`.

Phones and tablets on the same network connect as described in the main
README under "Phone or tablet on the same network".

Updates
-------

When CODA starts on a release, it checks for a newer one and offers to update.
`Update CODA.app` does the same on demand. An update checks out the newest
release tag, then installs its Python packages and models. If that fails, the
laptop goes back to the version it had. Without internet the check is skipped.
`.env`, `certs/` and saved cases are not touched by updates.

The interview page shows the version in its footer. A version such as
`1.0.0-3-gab12cd3` or one ending in `-dirty` is not a release, and is shown in
orange.

Releasing
---------

1. Set `__version__` in `src/coda/__init__.py` and commit.
2. If the dependencies changed, regenerate the lock and commit it:

   ```bash
   uv pip compile pyproject.toml --extra mac --python-version 3.12 \
     --python-platform aarch64-apple-darwin -o mac/requirements.lock
   ```

3. Tag the commit with release notes, which the update dialog shows, and push:

   ```bash
   git tag -a v1.1.0 -m "Adds Xitsonga transcription"
   git push origin v1.1.0
   ```
