"""Detect Mac hardware so a benchmark run self-labels which machine produced it."""
import json
import subprocess


def _sysctl(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
            text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _macos_version():
    try:
        out = subprocess.run(["sw_vers", "-productVersion"], capture_output=True,
            text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


# Peak unified-memory bandwidth in GB/s, keyed by substrings of the chip name.
# Apple does not expose bandwidth at runtime, so it is looked up from the model.
_BANDWIDTH_GBPS = {
    "M1 Max": 400, "M1 Ultra": 800, "M1 Pro": 200, "M1": 68,
    "M2 Max": 400, "M2 Ultra": 800, "M2 Pro": 200, "M2": 100,
    "M3 Max": 400, "M3 Ultra": 800, "M3 Pro": 150, "M3": 100,
    "M4 Max": 546, "M4 Pro": 273, "M4": 120,
    "M5 Max": 614, "M5 Pro": 307, "M5": 153,
}


def _bandwidth_for(chip):
    for name in sorted(_BANDWIDTH_GBPS, key=len, reverse=True):
        if name in chip:
            return _BANDWIDTH_GBPS[name]
    return None


def hardware_info():
    chip = _sysctl("machdep.cpu.brand_string")
    memsize = _sysctl("hw.memsize")
    ram_gb = round(int(memsize) / (1024 ** 3)) if memsize.isdigit() else None
    ncpu = _sysctl("hw.ncpu")
    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "bandwidth_gbps": _bandwidth_for(chip),
        "cpu_cores": int(ncpu) if ncpu.isdigit() else None,
        "macos": _macos_version(),
    }


if __name__ == "__main__":
    print(json.dumps(hardware_info(), indent=2))
