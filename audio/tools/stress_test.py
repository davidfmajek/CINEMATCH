"""
stress_test.py -- Test matching accuracy across multiple movies AND multiple
audio degradations (noise, volume change, pitch shift, speed change).

This answers two questions at once:
    1. Does matching work consistently across DIFFERENT movies (not just Inception)?
    2. How much real-world distortion can the algorithm tolerate before it
       misidentifies or fails to match at all?

Usage:
    1. Edit DB_PATH, RAW_AUDIO_DIR, TEST_MOVIES below.
    2. Run:  python stress_test.py

Output: a results table printed to console, plus a CSV saved to disk.
"""

import sys
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))
from fingerprint.fingerprint import fingerprint_audio, SAMPLE_RATE
from fingerprint.match import match_clip, load_database

# ---- EDIT THESE ----
DB_PATH = Path("C:/Users/David/CINEMATCH/data/fingerprint_db.pkl")
RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
RESULTS_CSV_PATH = Path("C:/Users/David/CINEMATCH/data/stress_test_results.csv")

# Pick a handful of movies with varied audio character. Filenames must match
# what's in RAW_AUDIO_DIR exactly.
TEST_MOVIES = [
    "Inception (2010).wav",              # dense orchestral score
    "Whiplash (2014).wav",               # percussive, transient-heavy
    "The Matrix (1999).wav",             # dialogue + action SFX + score
    "Fight Club (1999).wav",             # dialogue-heavy, sparser audio
    "The Grand Budapest Hotel (2014).wav",  # distinctive quirky score
    "The Good, the Bad and the Ugly (1966).wav",  # older recording quality
]

CLIPS_PER_MOVIE = 3       # how many random clips to test per movie
CLIP_DURATION_SEC = 10
MARGIN_SEC = 120          # avoid the very start/end of each movie
# ----------------------


# ---- Audio degradation functions ----

def augment_clean(y, sr):
    return y

def augment_white_noise(y, sr, snr_db):
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)

def augment_volume(y, sr, db_change):
    factor = 10 ** (db_change / 20)
    return (y * factor).astype(np.float32)

def augment_pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_time_stretch(y, sr, rate):
    return librosa.effects.time_stretch(y, rate=rate)


AUGMENTATIONS = {
    "clean":            lambda y, sr: augment_clean(y, sr),
    "noise_20db_snr":   lambda y, sr: augment_white_noise(y, sr, 20),
    "noise_10db_snr":   lambda y, sr: augment_white_noise(y, sr, 10),
    "noise_0db_snr":    lambda y, sr: augment_white_noise(y, sr, 0),
    "volume_-10db":     lambda y, sr: augment_volume(y, sr, -10),
    "volume_-20db":     lambda y, sr: augment_volume(y, sr, -20),
    "pitch_+1_semitone": lambda y, sr: augment_pitch_shift(y, sr, 1),
    "pitch_-1_semitone": lambda y, sr: augment_pitch_shift(y, sr, -1),
    "speed_1.05x":      lambda y, sr: augment_time_stretch(y, sr, 1.05),
    "speed_0.95x":      lambda y, sr: augment_time_stretch(y, sr, 0.95),
}


def get_duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def extract_random_clip(movie_path: Path, sr=SAMPLE_RATE):
    """Load a random CLIP_DURATION_SEC window from a movie. Returns (y, start_sec)."""
    duration = get_duration(movie_path)
    max_start = max(MARGIN_SEC, duration - MARGIN_SEC - CLIP_DURATION_SEC)
    start = random.uniform(MARGIN_SEC, max_start)
    y, _ = librosa.load(str(movie_path), sr=sr, mono=True,
                         offset=start, duration=CLIP_DURATION_SEC)
    return y, start


def run_single_test(y_clip, sr, hash_db, movie_names, expected_movie_id):
    query_hashes = fingerprint_audio(y_clip, sr)
    if not query_hashes:
        return {"correct": False, "top_movie": None, "votes": 0,
                "total_hits": 0, "confidence": 0.0, "n_query_hashes": 0}

    results = match_clip(query_hashes, hash_db, movie_names)
    if not results:
        return {"correct": False, "top_movie": None, "votes": 0,
                "total_hits": 0, "confidence": 0.0, "n_query_hashes": len(query_hashes)}

    top = results[0]  # (movie_id, title, offset, votes, total_hits)
    movie_id, title, offset, votes, total_hits = top
    confidence = votes / total_hits if total_hits else 0.0

    return {
        "correct": movie_id == expected_movie_id,
        "top_movie": title,
        "votes": votes,
        "total_hits": total_hits,
        "confidence": confidence,
        "n_query_hashes": len(query_hashes),
    }


def main():
    print(f"Loading database from {DB_PATH} ...")
    hash_db, movie_names = load_database(DB_PATH)
    # reverse lookup: filename stem -> movie_id
    title_to_id = {title: mid for mid, title in movie_names.items()}
    print(f"Database: {len(movie_names)} movies\n")

    rows = []

    for movie_filename in TEST_MOVIES:
        movie_path = RAW_AUDIO_DIR / movie_filename
        if not movie_path.is_file():
            print(f"WARNING: {movie_path} not found, skipping")
            continue

        expected_title = movie_path.stem
        expected_movie_id = title_to_id.get(expected_title)
        if expected_movie_id is None:
            print(f"WARNING: '{expected_title}' not found in database, skipping")
            continue

        for clip_num in range(CLIPS_PER_MOVIE):
            print(f"Extracting clip {clip_num + 1}/{CLIPS_PER_MOVIE} from {expected_title} ...")
            y_clean, start_sec = extract_random_clip(movie_path)

            for aug_name, aug_fn in AUGMENTATIONS.items():
                try:
                    y_aug = aug_fn(y_clean.copy(), SAMPLE_RATE)
                except Exception as e:
                    print(f"  [{aug_name}] augmentation failed: {e}")
                    continue

                result = run_single_test(y_aug, SAMPLE_RATE, hash_db, movie_names, expected_movie_id)

                status = "OK  " if result["correct"] else "FAIL"
                print(f"  [{aug_name:<20}] {status}  "
                      f"top={result['top_movie']!s:<40} "
                      f"votes={result['votes']:<5} conf={result['confidence']:.1%}")

                rows.append({
                    "movie": expected_title,
                    "clip_num": clip_num,
                    "start_sec": round(start_sec, 1),
                    "augmentation": aug_name,
                    **result,
                })

        print()

    # Save CSV
    if rows:
        RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved detailed results to {RESULTS_CSV_PATH}\n")

    # Summary: accuracy per augmentation type across all movies/clips
    print("=" * 60)
    print("SUMMARY: Accuracy by augmentation type")
    print("=" * 60)
    for aug_name in AUGMENTATIONS:
        aug_rows = [r for r in rows if r["augmentation"] == aug_name]
        if not aug_rows:
            continue
        correct = sum(1 for r in aug_rows if r["correct"])
        avg_conf = np.mean([r["confidence"] for r in aug_rows])
        print(f"{aug_name:<22} {correct}/{len(aug_rows)} correct   avg confidence={avg_conf:.1%}")


if __name__ == "__main__":
    main()