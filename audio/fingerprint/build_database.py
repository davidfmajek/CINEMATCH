"""
build_database.py -- Fingerprint every .wav in RAW_AUDIO_DIR and save
the combined hash database to disk as a pickle.

Usage:
    1. Edit RAW_AUDIO_DIR and DB_OUTPUT_PATH below.
    2. Run:  python build_database.py
"""

import sys
import pickle
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from fingerprint.fingerprint import fingerprint_file


RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
DB_OUTPUT_PATH = Path("C:/Users/David/CINEMATCH/data/fingerprint_db.pkl")


def build_database(raw_audio_dir: Path):
    wav_files = sorted(raw_audio_dir.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {raw_audio_dir}")
        return None

    print(f"Found {len(wav_files)} audio file(s) to fingerprint\n")

    hash_db = defaultdict(list)
    movie_names = {}

    for movie_id, wav_path in enumerate(wav_files):
        movie_title = wav_path.stem  # filename without .wav extension
        movie_names[movie_id] = movie_title

        start = time.time()
        hashes = fingerprint_file(str(wav_path))
        elapsed = time.time() - start

        for h, anchor_time in hashes:
            hash_db[h].append((movie_id, anchor_time))

        print(f"[{movie_id:2d}] {movie_title}: {len(hashes)} hashes ({elapsed:.1f}s)")

    return {
        "hashes": dict(hash_db),  # convert defaultdict back to plain dict for pickling
        "movie_names": movie_names,
    }


def main():
    if not RAW_AUDIO_DIR.is_dir():
        print(f"ERROR: {RAW_AUDIO_DIR} is not a valid directory")
        print("Edit RAW_AUDIO_DIR at the top of this script.")
        return

    db = build_database(RAW_AUDIO_DIR)
    if db is None:
        return

    DB_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_OUTPUT_PATH, "wb") as f:
        pickle.dump(db, f)

    total_hashes = sum(len(v) for v in db["hashes"].values())
    unique_hashes = len(db["hashes"])
    print(f"\nSaved database to {DB_OUTPUT_PATH}")
    print(f"Movies: {len(db['movie_names'])}")
    print(f"Total hash entries: {total_hashes}")
    print(f"Unique hash values: {unique_hashes}")


if __name__ == "__main__":
    main()