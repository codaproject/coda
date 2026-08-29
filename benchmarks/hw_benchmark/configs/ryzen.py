"""Ryzen AI config: Lemonade (llama.cpp/GGUF) over its OpenAI-compatible API.

Only llama.cpp/GGUF models are used: Lemonade's ryzenai-llm (NPU/Hybrid) recipe
does not honor response_format/json_schema, so it can't do CODA's constrained
decoding regardless of speed.

One-time Lemonade-side setup this config assumes is already done:
  lemonade alias add whisper-1 Whisper-Small
  lemonade load <model> --ctx-size 8192 --llamacpp-args "--reasoning off" --save-options
      (--reasoning off keeps models from burning context on chain-of-thought before
      emitting the JSON answer; models over ~10GB OOM on this iGPU-allocatable memory)

Excluded, with reasons:
  Qwen2.5-7B-Instruct-NPU/-Hybrid, gpt-oss-20b-NPU  - ryzenai-llm ignores json_schema (0% valid).
      gpt-oss-20b-NPU also took 242s and got stuck in a reasoning loop
  gpt-oss-20b-mxfp4-GGUF  - Lemonade catalog bug: this id resolves to a speculative-decoding
      draft model, so use gpt-oss-20b-GGUF-MXFP4 (in MODELS below) instead
  Gemma-4-26B-A4B-it-GGUF, Qwen3-30B-A3B-GGUF  - OOM on ~31GB RAM (>~10GB file fails, MoE doesn't help)
  DeepSeek-Qwen3-8B-GGUF  - R1-distill reasons at length (>3min/call) even with --reasoning off
"""

STT_BACKEND = "lemonade-whisper"
LEMONADE_BASE = "http://localhost:13305/api/v1"

MODELS = [
    {"name": "Qwen2.5-7B-Instruct-GGUF", "backend": "lemonade",
     "id": "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"},
    {"name": "Qwen3-8B-GGUF", "backend": "lemonade", "id": "Qwen3-8B-GGUF"},
    {"name": "gpt-oss-20b-GGUF", "backend": "lemonade", "id": "gpt-oss-20b-GGUF-MXFP4"},
    {"name": "Qwen3-14B-GGUF", "backend": "lemonade", "id": "Qwen3-14B-GGUF"},
    {"name": "Gemma-4-12B-it-GGUF", "backend": "lemonade", "id": "Gemma-4-12B-it-GGUF"},
]
