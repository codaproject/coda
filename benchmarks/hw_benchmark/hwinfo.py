"""Detect the machine's hardware so a benchmark run self-labels which config produced it.

Cross-platform: Apple Silicon Macs (sysctl/sw_vers) and Windows AMD Ryzen AI boxes
(cpuinfo/psutil). Both hide peak memory bandwidth at runtime, so it is looked up
from the chip name. The output shape is identical on both so results plot on one
shared bandwidth axis.
"""
import json
import platform
import subprocess


# Peak memory bandwidth in GB/s, keyed by substrings of the chip name.
# Apple: unified-memory bandwidth per SoC. AMD Strix Halo (Ryzen AI Max/Max+ 300):
# 256-bit LPDDR5X-8000 ~= 256 GB/s.
_BANDWIDTH_GBPS = {
    "M1 Max": 400, "M1 Ultra": 800, "M1 Pro": 200, "M1": 68,
    "M2 Max": 400, "M2 Ultra": 800, "M2 Pro": 200, "M2": 100,
    "M3 Max": 400, "M3 Ultra": 800, "M3 Pro": 150, "M3": 100,
    "M4 Max": 546, "M4 Pro": 273, "M4": 120,
    "M5 Max": 614, "M5 Pro": 307, "M5": 153,
    "Ryzen AI Max+ 395": 256, "Ryzen AI Max+ 390": 256,
    "Ryzen AI Max+ 388": 256, "Ryzen AI Max 385": 256, "Ryzen AI Max": 256,
}


def _bandwidth_for(chip):
    chip_lower = chip.lower()
    for name in sorted(_BANDWIDTH_GBPS, key=len, reverse=True):
        if name.lower() in chip_lower:
            return _BANDWIDTH_GBPS[name]
    return None


def _sysctl(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
            text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _mac_info():
    chip = _sysctl("machdep.cpu.brand_string")
    memsize = _sysctl("hw.memsize")
    ncpu = _sysctl("hw.ncpu")
    try:
        osver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True,
            text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        osver = platform.mac_ver()[0]
    return {
        "chip": chip,
        "ram_gb": round(int(memsize) / (1024 ** 3)) if memsize.isdigit() else None,
        "bandwidth_gbps": _bandwidth_for(chip),
        "cpu_cores": int(ncpu) if ncpu.isdigit() else None,
        "os": f"macOS {osver}".strip(),
    }


def _win_info():
    try:
        import cpuinfo
        chip = cpuinfo.get_cpu_info().get("brand_raw", "") or platform.processor()
    except Exception:
        chip = platform.processor()
    ram_gb = cores = None
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


def hardware_info():
    return _mac_info() if platform.system() == "Darwin" else _win_info()


if __name__ == "__main__":
    print(json.dumps(hardware_info(), indent=2))
