"""Detect Windows/AMD hardware so a benchmark run self-labels which machine produced it.

Mirrors the Mac hwinfo output shape (chip, ram_gb, bandwidth_gbps, cpu_cores, os)
so Ryzen results plot on the same bandwidth_vs_latency axis as the Macs. Apple and
AMD both hide peak memory bandwidth at runtime, so it is looked up from the chip name.
"""
import json
import platform


# Peak memory bandwidth in GB/s, keyed by substrings of the CPU brand string.
# Strix Halo (Ryzen AI Max/Max+ 300) is 256-bit LPDDR5X-8000 ~= 256 GB/s.
_BANDWIDTH_GBPS = {
    "Ryzen AI Max+ 395": 256,
    "Ryzen AI Max+ 390": 256,
    "Ryzen AI Max 385": 256,
    "Ryzen AI Max": 256,
}


def _bandwidth_for(chip):
    chip_lower = chip.lower()
    for name in sorted(_BANDWIDTH_GBPS, key=len, reverse=True):
        if name.lower() in chip_lower:
            return _BANDWIDTH_GBPS[name]
    return None


def _cpu_brand():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "") or platform.processor()
    except Exception:
        return platform.processor()


def hardware_info():
    chip = _cpu_brand()
    ram_gb = None
    cores = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
        cores = psutil.cpu_count(logical=False)
    except Exception:
        pass
    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "bandwidth_gbps": _bandwidth_for(chip),
        "cpu_cores": cores,
        "os": f"{platform.system()} {platform.release()}",
    }


if __name__ == "__main__":
    print(json.dumps(hardware_info(), indent=2))
