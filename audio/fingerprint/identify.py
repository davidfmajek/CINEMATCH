"""
identify.py -- Match ANY audio file against the fingerprint database.
Same logic as match.py, but takes the file path as a command-line argument
so you can quickly test multiple recordings without editing the script.

Usage:
    python identify.py path/to/recording.wav
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from fingerprint.fingerprint import fingerprint_file, HOP_LENGTH, SAMPLE_RATE, compute_spectrogram, load_audio
from fingerprint.match import load_database, match_clip

DB_PATH = Path("C:/Users/David/CINEMATCH/data/fingerprint_db.pkl")
TOP_N_RESULTS = 5


def frames_to_seconds(frames):
    return frames * HOP_LENGTH / SAMPLE_RATE


def main():
    if len(sys.argv) != 2:
        print("Usage: python identify.py path/to/recording.wav")
        sys.exit(1)

    query_path = Path(sys.argv[1])
    if not query_path.is_file():
        print(f"ERROR: {query_path} not found")
        sys.exit(1)

    print(f"Loading database from {DB_PATH} ...")
    hash_db, movie_names = load_database(DB_PATH)
    print(f"Database: {len(movie_names)} movies, {len(hash_db)} unique hashes\n")

    # Show the dB range so I can sanity-check the recording level --
    # mic recordings sit much quieter than digital extractions.
    y, sr = load_audio(str(query_path))
    S_db = compute_spectrogram(y)
    print(f"Query dB range: min={S_db.min():.1f}, max={S_db.max():.1f}, "
          f"median={np.median(S_db):.1f}")

    print(f"\nFingerprinting {query_path} ...")
    query_hashes = fingerprint_file(str(query_path))
    print(f"Query generated {len(query_hashes)} hashes\n")

    if len(query_hashes) == 0:
        print("WARNING: zero hashes generated. The recording is likely too quiet")
        print("for the current MIN_AMPLITUDE_DB threshold -- see fingerprint.py")
        return

    results = match_clip(query_hashes, hash_db, movie_names)

    if not results:
        print("No matches found at all.")
        return

    print(f"Top {min(TOP_N_RESULTS, len(results))} matches:\n")
    print(f"{'Rank':<5}{'Movie':<45}{'Votes':<8}{'TotalHits':<10}{'Confidence':<10}")
    print("-" * 80)

    for rank, (movie_id, title, offset, votes, total_hits) in enumerate(results[:TOP_N_RESULTS], 1):
        confidence = votes / total_hits if total_hits else 0
        offset_sec = frames_to_seconds(offset)
        print(f"{rank:<5}{title:<45}{votes:<8}{total_hits:<10}{confidence:<10.2%}")
        if rank == 1:
            print(f"      -> Best guess: ~{offset_sec:.1f}s ({offset_sec/60:.1f} min) into '{title}'")


if __name__ == "__main__":
    main()