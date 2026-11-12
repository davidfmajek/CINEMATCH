"""
sweep_percentile.py -- Try several PEAK_PERCENTILE values and compare accuracy
across: clean/noise/volume/pitch/speed augmentations AND real mic recordings.

Uses only TEST_MOVIES (not all 30) to build a small in-memory database per
percentile value, so a full sweep takes minutes instead of rebuilding the
30-movie database from scratch for every value tried.

Usage:
    1. Edit RAW_AUDIO_DIR, TEST_MOVIES, MIC_RECORDINGS, PERCENTILES_TO_TRY below.
    2. Run:  python sweep_percentile.py
"""

import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))
from fingerprint.fingerprint import fingerprint_audio, fingerprint_file, SAMPLE_RATE
from fingerprint.match import match_clip
from tools.stress_test import AUGMENTATIONS, extract_random_clip, MARGIN_SEC, CLIP_DURATION_SEC

# ---- EDIT THESE ----
RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")

TEST_MOVIES = [
    "Inception (2010).wav",
    "Whiplash (2014).wav",
    "The Matrix (1999).wav",
    "Fight Club (1999).wav",
    "The Grand Budapest Hotel (2014).wav",
    "The Good, the Bad and the Ugly (1966).wav",
    "Parasite (2019).wav",   # the actual source of the mic recordings -- must be
                              # included or mic-recording accuracy is meaningless
]

# (path_to_mic_recording, expected_movie_title) -- title must match a TEST_MOVIES stem
MIC_RECORDINGS = [
    ("C:/Users/David/CINEMATCH/data/clips/mic_recording_01.wav", "Parasite (2019)"),
    ("C:/Users/David/CINEMATCH/data/clips/mic_recording_02.wav", "Parasite (2019)"),
]

PERCENTILES_TO_TRY = [95, 97]
CLIPS_PER_MOVIE = 2   # fewer than full stress_test for speed; bump up once you've narrowed down
# ----------------------


def build_mini_database(movie_paths, percentile):
    """Build an in-memory hash DB for just the test movies, at a given percentile."""
    hash_db = defaultdict(list)
    movie_names = {}
    for movie_id, path in enumerate(movie_paths):
        movie_names[movie_id] = path.stem
        hashes = fingerprint_file(str(path), percentile=percentile)
        for h, t in hashes:
            hash_db[h].append((movie_id, t))
    return dict(hash_db), movie_names


def evaluate_percentile(percentile, movie_paths, mic_recordings):
    print(f"\n{'='*60}")
    print(f"PERCENTILE = {percentile}")
    print(f"{'='*60}")

    print("Building mini database...")
    hash_db, movie_names = build_mini_database(movie_paths, percentile)
    title_to_id = {title: mid for mid, title in movie_names.items()}

    # --- Augmentation accuracy (reuses stress_test's clip extraction + augmentations) ---
    aug_results = defaultdict(list)  # aug_name -> list of (correct, confidence)

    for movie_path in movie_paths:
        expected_id = title_to_id[movie_path.stem]
        for _ in range(CLIPS_PER_MOVIE):
            y_clean, _ = extract_random_clip(movie_path)
            for aug_name, aug_fn in AUGMENTATIONS.items():
                try:
                    y_aug = aug_fn(y_clean.copy(), SAMPLE_RATE)
                except Exception:
                    continue
                query_hashes = fingerprint_audio(y_aug, SAMPLE_RATE, percentile=percentile)
                if not query_hashes:
                    aug_results[aug_name].append((False, 0.0))
                    continue
                results = match_clip(query_hashes, hash_db, movie_names)
                if not results:
                    aug_results[aug_name].append((False, 0.0))
                    continue
                movie_id, title, offset, votes, total_hits = results[0]
                confidence = votes / total_hits if total_hits else 0.0
                aug_results[aug_name].append((movie_id == expected_id, confidence))

    print("\nAugmentation accuracy:")
    for aug_name in AUGMENTATIONS:
        rows = aug_results[aug_name]
        if not rows:
            continue
        correct = sum(1 for c, _ in rows if c)
        avg_conf = np.mean([conf for _, conf in rows])
        print(f"  {aug_name:<22} {correct}/{len(rows)} correct   avg confidence={avg_conf:.1%}")

    # --- Mic recording accuracy ---
    print("\nMic recordings:")
    mic_results = []
    for mic_path_str, expected_title in mic_recordings:
        mic_path = Path(mic_path_str)
        if not mic_path.is_file():
            print(f"  WARNING: {mic_path} not found, skipping")
            continue
        query_hashes = fingerprint_file(str(mic_path), percentile=percentile)
        results = match_clip(query_hashes, hash_db, movie_names)
        if not results:
            print(f"  {mic_path.name}: NO MATCH")
            mic_results.append((False, 0.0))
            continue
        movie_id, title, offset, votes, total_hits = results[0]
        confidence = votes / total_hits if total_hits else 0.0
        correct = (title == expected_title)
        status = "OK  " if correct else "FAIL"
        print(f"  {mic_path.name}: {status} top={title} votes={votes} conf={confidence:.1%}")
        mic_results.append((correct, confidence))

    return aug_results, mic_results


def main():
    movie_paths = [RAW_AUDIO_DIR / name for name in TEST_MOVIES]
    missing = [p for p in movie_paths if not p.is_file()]
    if missing:
        print("ERROR: missing movie files:")
        for p in missing:
            print(f"  {p}")
        return

    summary = {}
    for pct in PERCENTILES_TO_TRY:
        aug_results, mic_results = evaluate_percentile(pct, movie_paths, MIC_RECORDINGS)
        total_correct = sum(sum(1 for c, _ in rows if c) for rows in aug_results.values())
        total_attempts = sum(len(rows) for rows in aug_results.values())
        mic_correct = sum(1 for c, _ in mic_results if c)
        mic_total = len(mic_results)
        summary[pct] = (total_correct, total_attempts, mic_correct, mic_total)

    print(f"\n\n{'='*70}")
    print("OVERALL SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Percentile':<12}{'Aug Accuracy':<18}{'Mic Accuracy':<15}")
    for pct, (correct, attempts, mic_correct, mic_total) in summary.items():
        aug_acc = f"{correct}/{attempts} ({correct/attempts:.1%})" if attempts else "n/a"
        mic_acc = f"{mic_correct}/{mic_total}" if mic_total else "n/a"
        print(f"{pct:<12}{aug_acc:<18}{mic_acc:<15}")


if __name__ == "__main__":
    main()