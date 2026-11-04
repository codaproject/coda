import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)

from coda.inference.champs_finetuned.engine import CausalLM

PREFIX = [5, 6, 7, 8]
SHARED = [10, 11, 12]
END = 99


@pytest.fixture(scope="module")
def causal_lm() -> CausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    # Skip __init__, which downloads the tokenizer and weights
    lm = CausalLM.__new__(CausalLM)
    lm._model = LlamaForCausalLM(config).eval()
    return lm


def reference_scores(
    model: PreTrainedModel, suffixes: list[list[int]], unscored: int
) -> list[float]:
    """Mean log-prob of each suffix past its first `unscored` tokens, from
    one uncached forward pass per candidate."""
    scores = []
    with torch.inference_mode():
        for suffix in suffixes:
            ids = torch.tensor([PREFIX + suffix])
            logprobs = model(input_ids=ids).logits[0, :-1].float().log_softmax(-1)
            token_logprobs = logprobs.gather(1, ids[0, 1:, None])[:, 0]
            scores.append(token_logprobs[-(len(suffix) - unscored) :].mean().item())
    return scores


@pytest.mark.parametrize(
    ("suffixes", "unscored"),
    [
        (
            [
                SHARED + answer + [END]
                for answer in ([20], [21], [22, 23], [24, 25], [26, 27])
            ],
            len(SHARED),
        ),
        ([[20, END], [21, 22, END]], 0),
        ([SHARED + [20, END]], len(SHARED) + 1),
    ],
    ids=["shared_prefix", "no_shared_prefix", "single_candidate"],
)
def test_scores_are_mean_logprob_of_answer_tokens(
    causal_lm: CausalLM, suffixes: list[list[int]], unscored: int
) -> None:
    scores = causal_lm._score_suffixes(
        PREFIX,
        [PREFIX + suffix for suffix in suffixes],
        [len(suffix) for suffix in suffixes],
        batch_size=2,
    )
    expected = reference_scores(causal_lm._model, suffixes, unscored)
    assert scores == pytest.approx(expected, abs=1e-4)
