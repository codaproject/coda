"""Per-platform benchmark configs (model list + STT backend).

The rest of the benchmark is shared; only these differ between machines. run_bench
auto-selects mac vs ryzen from the OS, or takes --config to override.
"""
import importlib
import platform


def load(name=None):
    if name is None:
        name = "mac" if platform.system() == "Darwin" else "ryzen"
    mod = importlib.import_module(f"configs.{name}")
    return {
        "name": name,
        "stt_backend": mod.STT_BACKEND,
        "lemonade_base": getattr(mod, "LEMONADE_BASE", None),
        "models": mod.MODELS,
    }
