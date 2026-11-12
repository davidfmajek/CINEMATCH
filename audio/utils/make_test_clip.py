"""
make_test_clip.py -- Extract a short clip from one of your raw_audio movies,
for use as a test query against the fingerprint database.

Usage:
    1. Edit RAW_AUDIO_DIR, OUTPUT_CLIP_PATH, MOVIE_FILENAME below.
    2. Optionally set START_TIME_SEC (or leave None for a random start).
    3. Run:  python make_test_clip.py
"""

import random
import subprocess
from pathlib import Path
import soundfile as sf

# ---- EDIT THESE ----
RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
MOVIE_FILENAME = "Inception (2010).wav"   # pick any file from raw_audio/
OUTPUT_CLIP_PATH = Path("C:/Users/David/CINEMATCH/data/clips/test_clip_01.wav")
CLIP_DURATION_SEC = 10
START_TIME_SEC = None   # set a number (e.g. 1200) for a specific start, or None for random
# ----------------------


def get_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def main():
    movie_path = RAW_AUDIO_DIR / MOVIE_FILENAME
    if not movie_path.is_file():
        print(f"ERROR: {movie_path} not found")
        return

    duration = get_duration(movie_path)
    print(f"{MOVIE_FILENAME}: {duration/60:.1f} minutes long")

    start = START_TIME_SEC
    if start is None:
        # keep some margin from the very start/end (opening/closing credits
        # tend to be silence or just music that's less representative)
        margin = 120
        start = random.uniform(margin, max(margin, duration - margin - CLIP_DURATION_SEC))

    print(f"Extracting {CLIP_DURATION_SEC}s starting at {start:.1f}s ({start/60:.1f} min)")

    OUTPUT_CLIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(movie_path),
        "-ss", str(start),
        "-t", str(CLIP_DURATION_SEC),
        "-loglevel", "error",
        str(OUTPUT_CLIP_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("FAILED:", result.stderr[:500])
        return

    print(f"Saved clip to {OUTPUT_CLIP_PATH}")
    print(f"(Ground truth: this clip is from '{MOVIE_FILENAME}' at {start:.1f}s)")


if __name__ == "__main__":
    main()