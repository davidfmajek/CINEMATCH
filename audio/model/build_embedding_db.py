"""
build_embedding_db.py -- Run every movie through the trained NeuralFingerprinter
in sliding 1s windows, and save the resulting embeddings as a database for
nearest-neighbor retrieval.

Usage:
    1. Edit RAW_AUDIO_DIR, CHECKPOINT_PATH, DB_OUTPUT_PATH below.
    2. Run:  python build_embedding_db.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dataset import WINDOW_SEC, WINDOW_SAMPLES, load_window, to_spectrogram_tensor
from fingerprint.fingerprint import SAMPLE_RATE
from model.model import NeuralFingerprinter

# ---- EDIT THESE ----
RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
CHECKPOINT_PATH = Path("C:/Users/David/CINEMATCH/data/checkpoints/fingerprinter_r2_latest.pt")
DB_OUTPUT_PATH = Path("C:/Users/David/CINEMATCH/data/embedding_db_r2.npz")
# ----------------------

HOP_SEC = 0.5   # paper uses ~1s cadence; 0.5s gives denser coverage which helps at this scale
BATCH_SIZE = 256  # windows per GPU batch during embedding extraction


def embed_movie(model, device, movie_path: Path):
    """Slide a window across the whole movie, embed each one, return (embeddings, start_times)."""
    duration = sf.info(str(movie_path)).frames / sf.info(str(movie_path)).samplerate
    starts = np.arange(0, duration - WINDOW_SEC, HOP_SEC)

    all_embeddings = []
    batch_tensors = []
    batch_starts = []

    def flush_batch():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            emb = model(batch).cpu().numpy()
        all_embeddings.append(emb)
        batch_tensors.clear()

    for start in starts:
        y = load_window(movie_path, start)
        tensor = to_spectrogram_tensor(y)
        batch_tensors.append(tensor)
        batch_starts.append(start)
        if len(batch_tensors) >= BATCH_SIZE:
            flush_batch()
    flush_batch()  # remaining partial batch

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 64))
    return embeddings, np.array(batch_starts)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NeuralFingerprinter().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    movie_paths = sorted(RAW_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(movie_paths)} movies\n")

    all_embeddings = []
    all_movie_ids = []
    all_start_times = []
    movie_names = {}

    for movie_id, movie_path in enumerate(movie_paths):
        movie_names[movie_id] = movie_path.stem
        print(f"[{movie_id:2d}] Embedding {movie_path.stem} ...")

        embeddings, starts = embed_movie(model, device, movie_path)
        all_embeddings.append(embeddings)
        all_movie_ids.append(np.full(len(starts), movie_id, dtype=np.int32))
        all_start_times.append(starts)

        print(f"      {len(starts)} windows embedded")

    embeddings = np.concatenate(all_embeddings, axis=0)
    movie_ids = np.concatenate(all_movie_ids, axis=0)
    start_times = np.concatenate(all_start_times, axis=0)

    DB_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DB_OUTPUT_PATH,
        embeddings=embeddings.astype(np.float32),
        movie_ids=movie_ids,
        start_times=start_times.astype(np.float32),
        movie_names=np.array(list(movie_names.items()), dtype=object),
    )

    print(f"\nSaved embedding database to {DB_OUTPUT_PATH}")
    print(f"Total windows: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Database size: {embeddings.nbytes / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()