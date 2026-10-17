"""Non-speech bank: sounds a transcriber should treat as silence.

Writes data/non_speech/<name>.wav (16 kHz mono int16, 60 s each) and a
manifest.json describing every clip. Two kinds:

  synthetic  generated here, reproducible from a seed: digital silence, white
             noise, pink noise (broadband ambience), brown noise (ventilation,
             distant traffic rumble), mains hum, and a muffled-conversation mix
             made by low-pass filtering three readings from the speech bank.
  recorded   real environments from DEMAND (Thiemann, Ito & Vincent 2013,
             CC BY-SA 3.0, https://zenodo.org/records/1227121): cafeteria
             babble, a meeting room, a living room, a kitchen, street traffic
             and a park. One channel and one minute of each is kept; the
             ~100 MB archives are deleted after extraction.

Every clip except silence is normalized to the same RMS level so the splicer
can set a signal-to-noise ratio against speech by scaling alone.

    python benchmarks/synthetic_narrative/non_speech_bank.py
"""
import argparse
import json
import urllib.request
import zipfile

import numpy as np
from scipy.signal import butter, sosfilt

from common import HERE, SAMPLE_RATE
import audio

OUT_DIR = HERE / "data" / "non_speech"
SPEECH_DIR = HERE / "data" / "transcript_to_speech"
SECONDS = 60.0
RMS_DBFS = -25.0  # common level for every clip except silence
SEED = 0

DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1"
# name -> DEMAND environment code
RECORDED = {
    "cafeteria": "PCAFETER",
    "meeting_room": "OMEETING",
    "living_room": "DLIVING",
    "kitchen": "DKITCHEN",
    "street_traffic": "STRAFFIC",
    "park": "NPARK",
}
DEMAND_SKIP_S = 30.0  # skip the first half minute of each 5 min recording

# Speech-bank readings mixed into the synthetic muffled conversation.
CONVERSATION_SOURCES = ("en/lri_2", "pt_br/malaria_1", "bn/diarrhea_1")
CONVERSATION_CUTOFF_HZ = 800.0


def normalize(x, rms_dbfs=RMS_DBFS):
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x))
    if rms == 0:
        return np.zeros(len(x), dtype=np.int16)
    x = x / rms * (10 ** (rms_dbfs / 20)) * 32767
    return np.clip(x, -32768, 32767).astype(np.int16)


def colored_noise(rng, n, exponent):
    """Noise with a 1/f**exponent power spectrum: 0 white, 1 pink, 2 brown."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)
    freqs[0] = freqs[1]  # avoid dividing by zero at DC
    spectrum /= freqs ** (exponent / 2)
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n)


def mains_hum(rng, n):
    t = np.arange(n) / SAMPLE_RATE
    hum = sum(a * np.sin(2 * np.pi * 50 * k * t)
              for k, a in ((1, 1.0), (2, 0.5), (3, 0.25)))
    # slow amplitude wobble and a little hiss, as a real buzz has
    hum *= 1 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
    return hum + 0.02 * rng.standard_normal(n)


def fit(pcm, n):
    """Loop or trim to exactly n samples."""
    pcm = np.asarray(pcm)
    reps = int(np.ceil(n / max(len(pcm), 1)))
    return np.tile(pcm, reps)[:n]


def muffled_conversation(n):
    """Three readings overlapping behind a wall: mixed, then low-passed."""
    mix = np.zeros(n, dtype=np.float64)
    for source in CONVERSATION_SOURCES:
        language, case_id = source.split("/")
        pcm = audio.read_wav(SPEECH_DIR / language / f"{case_id}.wav")
        mix += fit(pcm, n).astype(np.float64)
    sos = butter(4, CONVERSATION_CUTOFF_HZ, btype="low", fs=SAMPLE_RATE,
                 output="sos")
    return sosfilt(sos, mix)


def synthetic_clips():
    rng = np.random.default_rng(SEED)
    n = int(SECONDS * SAMPLE_RATE)
    yield "silence", np.zeros(n), "digital zeros"
    yield "white_noise", rng.standard_normal(n), "flat spectrum"
    yield "pink_noise", colored_noise(rng, n, 1), "1/f spectrum, broadband ambience"
    yield "brown_noise", colored_noise(rng, n, 2), "1/f^2 spectrum, low rumble"
    yield "mains_hum", mains_hum(rng, n), "50 Hz hum with harmonics"
    yield ("muffled_conversation", muffled_conversation(n),
           f"three speech-bank readings {CONVERSATION_SOURCES} mixed and "
           f"low-passed at {CONVERSATION_CUTOFF_HZ:.0f} Hz")


def download_demand(env, dest_dir):
    """Fetch one DEMAND environment and return its channel-1 WAV as int16."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"{env}_16k.zip"
    if not archive.exists():
        print(f"  downloading {env} ...", flush=True)
        urllib.request.urlretrieve(DEMAND_URL.format(env=env), archive)
    with zipfile.ZipFile(archive) as z:
        member = next(m for m in z.namelist() if m.lower().endswith("ch01.wav"))
        with z.open(member) as f:
            wav_path = dest_dir / f"{env}_ch01.wav"
            wav_path.write_bytes(f.read())
    pcm = audio.read_wav(wav_path)
    archive.unlink()
    wav_path.unlink()
    return pcm


def build(force=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"seconds": SECONDS, "rms_dbfs": RMS_DBFS, "sample_rate": SAMPLE_RATE,
                "clips": {}}
    n = int(SECONDS * SAMPLE_RATE)

    for name, signal, description in synthetic_clips():
        path = OUT_DIR / f"{name}.wav"
        if force or not path.exists():
            audio.write_wav(path, normalize(signal))
        manifest["clips"][name] = {"file": path.name, "kind": "synthetic",
                                   "description": description}
        print(f"  {name:22s} synthetic  {description}", flush=True)

    for name, env in RECORDED.items():
        path = OUT_DIR / f"{name}.wav"
        if force or not path.exists():
            pcm = download_demand(env, OUT_DIR / "_download")
            start = int(DEMAND_SKIP_S * SAMPLE_RATE)
            audio.write_wav(path, normalize(pcm[start:start + n]))
        manifest["clips"][name] = {
            "file": path.name, "kind": "recorded",
            "description": f"DEMAND {env}, channel 1, from {DEMAND_SKIP_S:.0f} s",
            "source": DEMAND_URL.format(env=env), "license": "CC BY-SA 3.0"}
        print(f"  {name:22s} recorded   DEMAND {env}", flush=True)

    if (OUT_DIR / "_download").exists():
        (OUT_DIR / "_download").rmdir()
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
