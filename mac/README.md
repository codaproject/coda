CODA on a Mac
=============

CODA runs natively on an Apple Silicon Mac, with MLX transcription and a local
Ollama model. Three apps go on the Desktop:

- `CODA` starts CODA in a Terminal window, and closing the window stops it.
- `Update CODA` moves the laptop to the newest CODA release.
- `CODA Settings` opens this laptop's settings file.

Installation
------------

On a new Mac, open Terminal and run, with the laptop's site (e.g. `kintampo`,
see `config/sites/`) in place of `<site>`:

```bash
curl -fsSL https://raw.githubusercontent.com/codaproject/coda/main/mac/install.sh | bash -s <site>
```

The first run installs the Xcode command line tools and asks to be run again
once they are in place. It then installs uv and Ollama, checks out the newest
CODA release into `~/coda`, writes the laptop's settings file, puts the apps
on the Desktop and starts CODA. The first start needs internet. It installs
the Python packages, downloads the models, prepares the grounding data, and
asks to trust CODA's local certificate authority so the browser accepts
`https://localhost:8000`.

At icddr,b, Bangla transcription uses a model that needs a Hugging Face
login. After the first start, sign in once with a Hugging Face token that has
accepted the terms of `ai4bharat/indic-conformer-600m-multilingual`, then
close CODA and start it again:

```bash
~/coda/.venv-mac/bin/hf auth login
```

Phones and tablets on the same network connect as described in the main
README under "Phone or tablet on the same network". CODA prints the setup page
address in its Terminal window.

Configuration
-------------

Settings come from three files, each overriding the one before:

1. `config/settings.yaml`, CODA's defaults, which lists and explains every
   setting.
2. `config/sites/<site>.yaml`, the site profile, such as the UI elements shown,
   storage, language and inference model. Profiles are part of CODA, so an
   update also updates the profile.
3. `~/Library/Application Support/CODA/settings.yaml`, this laptop's file,
   which `CODA Settings` opens. It names the site profile and holds any changes
   made on the laptop. Updates never touch it.

A site has one profile per use case, named `<site>-<use>.yaml`, e.g.
`kintampo-facility.yaml` and `kintampo-community.yaml`. With `site: kintampo`
in the laptop's file, CODA asks at start which use case to run when the site
has more than one, and offers the last choice first. `site:
kintampo-facility` always starts that use case. The interview page footer
shows the profile in use, and each saved case records it.

To start a new site profile, copy `config/sites/mac.yaml`. Before starting,
CODA checks the settings and shows a dialog naming any problem, such as a
misspelled setting or a value that is not allowed.

Updates
-------

When CODA starts on a release, it checks for a newer one and offers to update.
`Update CODA` does the same on demand. An update checks out the newest release
tag, then installs its Python packages and models. If that fails, the laptop
goes back to the version it had. Without internet the check is skipped. The
laptop's settings, `certs/` and saved cases are not touched by updates.

The interview page shows the version and commit in its footer, e.g.
`CODA 1.0.0 (ab12cd)`.

Releasing
---------

1. Set `__version__` in `src/coda/__init__.py` and commit.
2. If the dependencies changed, regenerate the lock and commit it:

   ```bash
   uv pip compile pyproject.toml --extra mac --extra indic-conformer \
     --python-version 3.12 \
     --python-platform aarch64-apple-darwin -o mac/requirements.lock
   ```

3. Tag the commit with a one-line summary, which the update dialog shows, and
   push the tag before creating the GitHub release from it:

   ```bash
   git tag -a 1.1.0 -m "Adds Xitsonga transcription"
   git push origin 1.1.0
   ```
