"""Causal LM wrapper with menu-candidate scoring for the MedGemma classifier."""

import copy
from collections.abc import Callable, Mapping
from typing import cast

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def enable_fast_math() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass


def load_tokenizer(
    model_id: str,
    *,
    padding_side: str | None = None,
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_id, padding_side=padding_side, use_fast=use_fast
        ),
    )
    if tokenizer.pad_token_id is not None:
        return tokenizer
    eos_token = tokenizer.eos_token
    if eos_token is None:
        raise ValueError(
            f"Tokenizer for {model_id!r} has neither pad_token nor eos_token."
        )
    tokenizer.pad_token = eos_token
    return tokenizer


def validate_token_lengths(lengths: list[int], *, max_length: int) -> None:
    if lengths and (longest := max(lengths)) > max_length:
        index = lengths.index(longest)
        raise ValueError(
            f"Input {index} has {longest} tokens, "
            f"exceeding the {max_length} token limit."
        )


class CausalLM:
    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: torch.dtype | str = "auto",
        adapter_path: str | None = None,
        model_kwargs_overrides: dict[str, object] | None = None,
        torch_compile: bool = False,
    ) -> None:
        enable_fast_math()
        self.model_id = model_id
        self._tokenizer = load_tokenizer(model_id, padding_side="left")
        model_kwargs = _build_model_kwargs(
            device_map=device_map, dtype=dtype, overrides=model_kwargs_overrides
        )
        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id, **model_kwargs
        )

        if adapter_path:
            self._model = _merge_lora_adapter(self._model, adapter_path)

        if torch_compile:
            self._model = cast(
                PreTrainedModel, torch.compile(self._model, fullgraph=True)
            )

        self._model.eval()

    def validate_inputs(self, inputs, generation_kwargs) -> None:
        model_text_config = self._model.config.get_text_config()
        context_length = getattr(model_text_config, "max_position_embeddings", None)
        max_new_tokens = generation_kwargs.get("max_new_tokens", 0)

        if not isinstance(context_length, int) or not isinstance(max_new_tokens, int):
            raise ValueError(
                "Model context length and max_new_tokens must be integers."
            )

        validate_token_lengths(
            inputs["attention_mask"].sum(dim=1).tolist(),
            max_length=context_length - max_new_tokens,
        )

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        generation_kwargs: Mapping[str, object],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        prompts = [self._format_prompt(messages) for messages in messages_batch]
        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        self.validate_inputs(inputs, generation_kwargs)

        inputs = inputs.to(self._model.device)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        generate = cast(Callable[..., torch.Tensor], self._model.generate)

        generate_kwargs = {
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            **generation_kwargs,
            **inputs,
        }

        with torch.inference_mode():
            outputs = generate(**generate_kwargs)

        decoded = self._tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=skip_special_tokens
        )

        return [text.strip() for text in decoded]

    def score_candidates(
        self,
        messages: list[dict[str, str]],
        candidates: list[str],
        *,
        batch_size: int,
    ) -> list[float]:
        prefix_ids, input_ids, suffix_lengths = self._encode_candidates(
            messages, candidates
        )

        return self._score_suffixes(
            prefix_ids, input_ids, suffix_lengths, batch_size=batch_size
        )

    def _encode_candidates(
        self,
        messages: list[dict[str, str]],
        candidates: list[str],
    ) -> tuple[list[int], list[list[int]], list[int]]:
        prefix_ids = self._tokenizer(
            self._format_prompt(messages), add_special_tokens=False
        )["input_ids"]
        full_texts = [
            self._format_prompt(
                [*messages, {"role": "assistant", "content": candidate}],
                add_generation_prompt=False,
            ).removesuffix("\n")
            for candidate in candidates
        ]
        input_ids = self._tokenizer(full_texts, add_special_tokens=False)["input_ids"]

        suffix_lengths = []
        for ids in input_ids:
            assert ids[: len(prefix_ids)] == prefix_ids
            suffix_lengths.append(len(ids) - len(prefix_ids))

        return prefix_ids, input_ids, suffix_lengths

    def _score_suffixes(
        self,
        prefix_ids: list[int],
        input_ids: list[list[int]],
        suffix_lengths: list[int],
        *,
        batch_size: int,
    ) -> list[float]:
        device = self._model.device
        prefix_len = len(prefix_ids)
        with torch.inference_mode():
            prefix = torch.tensor([prefix_ids], device=device)
            out = self._model(input_ids=prefix, use_cache=True, logits_to_keep=1)
            base_cache = out.past_key_values
            first_logprobs = out.logits[0, -1].float().log_softmax(-1)

            scores = [0.0] * len(input_ids)
            by_length: dict[int, list[int]] = {}
            for i, length in enumerate(suffix_lengths):
                by_length.setdefault(length, []).append(i)

            for indices in by_length.values():
                for start in range(0, len(indices), batch_size):
                    chunk = indices[start : start + batch_size]
                    suffixes = torch.tensor(
                        [input_ids[i][prefix_len:] for i in chunk], device=device
                    )
                    total = first_logprobs[suffixes[:, 0]]
                    if suffixes.shape[1] > 1:
                        cache = copy.deepcopy(base_cache)
                        cache.batch_repeat_interleave(len(chunk))
                        logits = self._model(
                            input_ids=suffixes[:, :-1], past_key_values=cache
                        ).logits
                        logprobs = logits.float().log_softmax(-1)
                        rest = logprobs.gather(2, suffixes[:, 1:, None])[..., 0]
                        total = total + rest.sum(-1)

                    for i, value in zip(chunk, total.tolist(), strict=True):
                        scores[i] = value

        return scores

    def _format_prompt(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool = True
    ) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                reasoning_effort="high",
            )
        )


class InferenceModel:
    model_id: str = ""
    skip_special_tokens: bool = False
    model_kwargs: dict[str, object] = {}

    def __init__(
        self,
        *,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: torch.dtype | str = "auto",
        torch_compile: bool = False,
    ) -> None:
        if not self.model_id:
            raise ValueError("model_id must be set on the model class.")

        self._model = CausalLM(
            self.model_id,
            device_map=device_map,
            dtype=dtype,
            adapter_path=adapter_path,
            model_kwargs_overrides=dict(self.model_kwargs),
            torch_compile=torch_compile,
        )

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        generation_kwargs: Mapping[str, object],
    ) -> list[str]:
        return self._model.generate_batch(
            messages_batch,
            generation_kwargs=generation_kwargs,
            skip_special_tokens=self.skip_special_tokens,
        )

    def score_candidates(
        self,
        messages: list[dict[str, str]],
        candidates: list[str],
        *,
        batch_size: int,
    ) -> list[float]:
        return self._model.score_candidates(messages, candidates, batch_size=batch_size)

    def parse_response(self, output: str) -> str:
        return output.strip()

    def parse_analysis(self, output: str) -> str | None:
        return None


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype | str:
    if dtype != "auto" or not torch.cuda.is_available():
        return dtype
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return dtype


def _build_model_kwargs(
    *,
    device_map: str,
    dtype: torch.dtype | str,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    model_kwargs: dict[str, object] = {
        "attn_implementation": "sdpa",
        "device_map": device_map,
        "dtype": _resolve_dtype(dtype),
    }
    if overrides:
        model_kwargs.update(overrides)
    return model_kwargs


def _merge_lora_adapter(model: PreTrainedModel, adapter_path: str) -> PreTrainedModel:
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    return cast(PreTrainedModel, peft_model.merge_and_unload())


class MedGemmaModel(InferenceModel):
    model_id = "google/medgemma-4b-it"
    skip_special_tokens = True

    def parse_analysis(self, output: str) -> str | None:
        return output.strip() or None
