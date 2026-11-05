"""Apple Silicon config: Ollama for qwen2.5/gpt-oss/Qwen3, mlx-openai-server for MLX Qwen3."""

STT_BACKEND = "mlx-whisper"

MODELS = [
    {"name": "qwen2.5-7b-instruct", "backend": "ollama", "id": "qwen2.5:7b-instruct"},
    {"name": "gpt-oss-20b", "backend": "ollama", "id": "gpt-oss:20b"},
    {"name": "qwen3-30b-a3b-instruct-2507-q4_K_M", "backend": "ollama",
     "id": "qwen3:30b-a3b-instruct-2507-q4_K_M"},
    {"name": "Qwen3-30B-A3B-Instruct-2507-4bit", "backend": "mlx",
     "id": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"},
]
