"""
test_pitch_invariance.py -- The whole point of building the neural
fingerprinter was pitch invariance. This checks if it actually delivers
on that, where the hash-based system scored ~0/18 (see FINDINGS.md).

Usage:
    1. Edit RAW_AUDIO_DIR below.
    2. Run:  python test_pitch_invariance.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dataset import WINDOW_SEC, load_window, to_spectrogram_tensor
from fingerprint.fingerprint import SAMPLE_RATE
from model.model import NeuralFingerprinter
from model.match_embedding import load_db, match, CHECKPOINT_PATH

RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
TEST_MOVIES = [
    "Inception (2010).wav",
    "Whiplash (2014).wav",
    "The Matrix (1999).wav",
    "Fight Club (1999).wav",
    "The Grand Budapest Hotel (2014).wav",
    "The Good, the Bad and the Ugly (1966).wav",
]
CLIP_DURATION_SEC = 10
N_CLIPS_PER_MOVIE = 3
PITCH_SHIFTS = [-1.0, 1.0]  # semitones, matching the stress_test.py conditions


def extract_and_pitch_shift(movie_path, start_sec, n_steps, sr=SAMPLE_RATE):
    y, file_sr = sf.read(
        str(movie_path),
        start=int(start_sec * sf.info(str(movie_path)).samplerate),
        frames=int(CLIP_DURATION_SEC * sf.info(str(movie_path)).samplerate),
        dtype="float32",
    )
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    if n_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32)


def embed_clip_array(model, device, y, sr=SAMPLE_RATE):
    """Slide windows across an in-memory audio array and embed each one."""
    hop_samples = int(0.5 * sr)
    window_samples = int(WINDOW_SEC * sr)
    starts_samples = range(0, max(1, len(y) - window_samples), hop_samples)

    tensors = []
    starts_sec = []
    for s in starts_samples:
        window = y[s:s + window_samples]
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)))
        tensors.append(to_spectrogram_tensor(window))
        starts_sec.append(s / sr)

    if not tensors:
        return np.zeros((0, 64)), np.array([])

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeddings = model(batch).cpu().numpy()
    return embeddings, np.array(starts_sec)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = NeuralFingerprinter().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    db_embeddings, db_movie_ids, db_start_times, movie_names = load_db()
    title_to_id = {title: mid for mid, title in movie_names.items()}
    print(f"Database: {len(movie_names)} movies, {len(db_embeddings)} windows\n")

    results_log = []

    for movie_name in TEST_MOVIES:
        movie_path = RAW_AUDIO_DIR / movie_name
        expected_id = title_to_id.get(movie_path.stem)
        if expected_id is None:
            print(f"WARNING: {movie_path.stem} not in database, skipping")
            continue

        duration = sf.info(str(movie_path)).frames / sf.info(str(movie_path)).samplerate
        margin = 120

        for clip_num in range(N_CLIPS_PER_MOVIE):
            start_sec = np.random.uniform(margin, duration - margin - CLIP_DURATION_SEC)

            for n_steps in PITCH_SHIFTS:
                y = extract_and_pitch_shift(movie_path, start_sec, n_steps)
                query_emb, query_starts = embed_clip_array(model, device, y)

                if len(query_emb) == 0:
                    continue

                results = match(query_emb, query_starts, db_embeddings, db_movie_ids,
                                 db_start_times, movie_names)
                if not results:
                    correct = False
                    top_title = "NO MATCH"
                    votes = 0
                else:
                    movie_id, top_title, offset, votes, total_hits = results[0]
                    correct = (movie_id == expected_id)

                status = "OK  " if correct else "FAIL"
                print(f"  [{movie_path.stem[:30]:<30} pitch={n_steps:+.1f}] {status} "
                      f"top={top_title[:30]:<30} votes={votes}")
                results_log.append(correct)

    n_correct = sum(results_log)
    n_total = len(results_log)
    print(f"\n{'='*60}")
    print(f"PITCH-SHIFT ACCURACY: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
    print(f"(Hash-based system scored 0-2/18 on the same test -- see FINDINGS.md)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()