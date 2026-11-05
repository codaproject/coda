"""Download the models one site profile needs, so CODA can then run offline.

Pulls the profile's Ollama models that are missing and loads its speech model
once, which caches it. With the argument ``ollama``, only pulls the Ollama
models. A ``HF_TOKEN`` in the environment is saved for later starts.
"""
import os
import subprocess
import sys

from coda.config import settings


def ollama_models():
    """Return the Ollama models the profile uses."""
    models = set()
    if settings.inference.llm.provider == "ollama":
        models.add(settings.inference.llm.model)
    if settings.app.show_clinical_note and settings.app.ocr_model:
        models.add(settings.app.ocr_model)
    return models


def main():
    for model in sorted(ollama_models()):
        if subprocess.run(["ollama", "show", model],
                          capture_output=True).returncode:
            print(f"Downloading {model}...", flush=True)
            subprocess.run(["ollama", "pull", model], check=True)
    if sys.argv[1:] == ["ollama"]:
        return
    if os.getenv("HF_TOKEN"):
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    profile = " ".join(filter(None, [settings.site, settings.usecase]))
    print(f"Preparing the {profile} speech model...", flush=True)
    from coda.dialogue import create_transcriber
    create_transcriber(settings.dialogue.transcriber_backend,
                       model=settings.dialogue.transcriber_model or None)


if __name__ == "__main__":
    main()
