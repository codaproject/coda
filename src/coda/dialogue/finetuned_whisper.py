"""Fine-tuned Whisper checkpoints published in CTranslate2 and MLX formats."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    """A Hugging Face model repository pinned to a revision."""
    repo: str
    revision: str


@dataclass(frozen=True)
class FinetunedWhisper:
    """A Whisper fine-tune for one language, in CTranslate2 and MLX formats.

    `decode_language` is the Whisper language token to decode with when
    transcribing `language`, needed when Whisper has no token of its own for it.
    """
    ct2: Checkpoint
    mlx: Checkpoint
    language: str
    decode_language: str
    condition_on_previous_text: bool = True


FINETUNED_MODELS = {
    "anv-tso-turbo": FinetunedWhisper(
        ct2=Checkpoint("gyorilab/whisper-large-v3-turbo-anv-tso-ct2",
                       "7af53f4171cc3a1d6cc72684c976bd2aff8073bc"),
        mlx=Checkpoint("gyorilab/whisper-large-v3-turbo-anv-tso-mlx",
                       "38406654473e82de2c6c066f3b6b45c580e8e253"),
        language="ts",
        decode_language="sw",
        condition_on_previous_text=False,
    ),
    "anv-zul-turbo": FinetunedWhisper(
        ct2=Checkpoint("gyorilab/whisper-large-v3-turbo-anv-zul-ct2",
                       "5f760351c4fe82853b4565ccccec233d1773f77a"),
        mlx=Checkpoint("gyorilab/whisper-large-v3-turbo-anv-zul-mlx",
                       "1cfe913ea3269026a5664ad498a1555af6205981"),
        language="zu",
        decode_language="en",
        condition_on_previous_text=False,
    ),
    "anv-sot-turbo": FinetunedWhisper(
        ct2=Checkpoint("gyorilab/whisper-large-v3-turbo-anv-sot-ct2",
                       "5bdfcfab559a7e86a944d2438112dfe02378d687"),
        mlx=Checkpoint("gyorilab/whisper-large-v3-turbo-anv-sot-mlx",
                       "0d126d19c04c4ebb4d1be6fe63239ab268a58735"),
        language="st",
        decode_language="en",
        condition_on_previous_text=False,
    ),
}

# Languages the fine-tunes add beyond Whisper's own
FINETUNED_LANGUAGES = {"ts": "Tsonga", "zu": "Zulu", "st": "Sesotho"}


def decode_options(name, language):
    """Return (language, condition_on_previous_text) to decode `language` with.

    Models that aren't fine-tunes pass the language through unchanged.
    """
    spec = FINETUNED_MODELS.get(name)
    if spec is None:
        return language, True
    if language == spec.language:
        language = spec.decode_language
    return language, spec.condition_on_previous_text


def get_model_path(name, mlx=False):
    """Return the local directory of a fine-tune, downloading it if needed."""
    from huggingface_hub import snapshot_download

    spec = FINETUNED_MODELS[name]
    checkpoint = spec.mlx if mlx else spec.ct2
    return snapshot_download(checkpoint.repo, revision=checkpoint.revision)
