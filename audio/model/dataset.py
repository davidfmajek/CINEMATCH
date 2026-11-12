"""
dataset.py -- Triplet sampling for training the neural fingerprinter.

Each triplet is (anchor, positive, negative):
    anchor   = clean 1s window from some movie
    positive = same window with random augmentation (noise/pitch/volume/speed)
    negative = different window, either different movie or far away in the same one

Augmentation happens on-the-fly -- nothing gets written to disk. I use
soundfile's partial read so I'm only loading 1s at a time, not entire movies.
"""

import sys
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from fingerprint.fingerprint import SAMPLE_RATE, compute_spectrogram
from tools.stress_test import (
    augment_white_noise, augment_volume, augment_pitch_shift, augment_time_stretch
)

WINDOW_SEC = 1.0          # length of each training window (paper uses ~1s cadence)
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
MARGIN_SEC = 60            # avoid the very start/end of each movie


def random_augment(y, sr):
    """Pick and apply a random augmentation.

    Pitch and speed changes use a phase vocoder which alters the array length
    and occasionally produces NaN on short windows. I clamp back to exactly
    WINDOW_SAMPLES and sanitize after.
    """
    choice = random.choice(["noise", "volume", "pitch", "speed", "none"])

    if choice == "noise":
        snr_db = random.uniform(0, 25)
        y_aug = augment_white_noise(y, sr, snr_db)
    elif choice == "volume":
        db_change = random.uniform(-25, -3)
        y_aug = augment_volume(y, sr, db_change)
    elif choice == "pitch":
        n_steps = random.uniform(-1.5, 1.5)
        y_aug = augment_pitch_shift(y, sr, n_steps)
    elif choice == "speed":
        rate = random.uniform(0.9, 1.1)
        y_aug = augment_time_stretch(y, sr, rate)
    else:
        y_aug = y  # occasionally train on a "clean" positive too

    # Guard against NaN/Inf from phase vocoder instability on short windows
    y_aug = np.nan_to_num(y_aug, nan=0.0, posinf=0.0, neginf=0.0)

    # Force back to exactly WINDOW_SAMPLES -- speed/pitch changes alter length
    if len(y_aug) < WINDOW_SAMPLES:
        y_aug = np.pad(y_aug, (0, WINDOW_SAMPLES - len(y_aug)))
    else:
        y_aug = y_aug[:WINDOW_SAMPLES]

    return y_aug.astype(np.float32)


def load_window(path: Path, start_sec: float, duration_sec=WINDOW_SEC, sr=SAMPLE_RATE):
    """Efficiently load just a short window of audio from a (potentially huge) file."""
    y, file_sr = sf.read(
        str(path),
        start=int(start_sec * sf.info(str(path)).samplerate),
        frames=int(duration_sec * sf.info(str(path)).samplerate),
        dtype="float32",
    )
    if y.ndim > 1:
        y = y.mean(axis=1)  # ensure mono
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    # Pad if the read was short (hit end of file)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    return y[:WINDOW_SAMPLES]


def to_spectrogram_tensor(y):
    """Audio window -> normalized spectrogram tensor, ready for the CNN."""
    S_db = compute_spectrogram(y)
    # Normalize per-window so absolute loudness doesn't affect the input.
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    S_db = np.nan_to_num(S_db, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(S_db).float().unsqueeze(0)  # add channel dim: (1, freq, time)


class TripletAudioDataset(Dataset):
    """
    Each __getitem__ call returns one (anchor, positive, negative) triplet
    of spectrogram tensors, sampled fresh (with fresh random augmentation)
    every time -- so effectively infinite training data from a fixed set
    of movies.
    """

    def __init__(self, movie_paths, samples_per_epoch=5000, margin_sec=MARGIN_SEC):
        self.movie_paths = list(movie_paths)
        self.samples_per_epoch = samples_per_epoch
        self.margin_sec = margin_sec
        self.durations = {
            p: sf.info(str(p)).frames / sf.info(str(p)).samplerate
            for p in self.movie_paths
        }

    def __len__(self):
        return self.samples_per_epoch

    def _random_window_start(self, path):
        duration = self.durations[path]
        max_start = max(self.margin_sec, duration - self.margin_sec - WINDOW_SEC)
        return random.uniform(self.margin_sec, max_start)

    def _pick_anchor_start(self, path, hard_example_prob=0.5, n_candidates=4):
        """
        Pick a window start. Half the time, pick the quietest of a few
        candidates instead of a random one -- forces training on the content
        the model was actually struggling with (see FINDINGS.md).
        """
        if random.random() > hard_example_prob:
            return self._random_window_start(path)

        candidates = [self._random_window_start(path) for _ in range(n_candidates)]
        energies = []
        for start in candidates:
            y = load_window(path, start)
            rms = np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)
            energies.append(rms)

        quietest_idx = int(np.argmin(energies))
        return candidates[quietest_idx]

    def __getitem__(self, idx):
        # --- Anchor ---
        anchor_movie = random.choice(self.movie_paths)
        anchor_start = self._pick_anchor_start(anchor_movie)
        y_anchor = load_window(anchor_movie, anchor_start)

        # --- Positive: same window, randomly augmented ---
        y_positive = random_augment(y_anchor.copy(), SAMPLE_RATE)

        # --- Negative: different window (different movie, or far away in same movie) ---
        if random.random() < 0.5 or len(self.movie_paths) == 1:
            # same movie, far away in time
            negative_movie = anchor_movie
            negative_start = self._random_window_start(anchor_movie)
            # ensure it's not accidentally overlapping the anchor
            while abs(negative_start - anchor_start) < 5.0:
                negative_start = self._random_window_start(anchor_movie)
        else:
            negative_movie = random.choice([p for p in self.movie_paths if p != anchor_movie])
            negative_start = self._random_window_start(negative_movie)
        y_negative = load_window(negative_movie, negative_start)
        # Negatives can also be augmented sometimes, so the model doesn't
        # learn "augmented = positive" as a shortcut.
        if random.random() < 0.3:
            y_negative = random_augment(y_negative, SAMPLE_RATE)

        return (
            to_spectrogram_tensor(y_anchor),
            to_spectrogram_tensor(y_positive),
            to_spectrogram_tensor(y_negative),
        )


if __name__ == "__main__":
    # Quick smoke test
    import sys
    from pathlib import Path

    RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
    movie_paths = sorted(RAW_AUDIO_DIR.glob("*.wav"))[:3]  # just a few for the smoke test
    print(f"Testing with {len(movie_paths)} movies: {[p.stem for p in movie_paths]}")

    ds = TripletAudioDataset(movie_paths, samples_per_epoch=5)
    for i in range(3):
        a, p, n = ds[i]
        print(f"Sample {i}: anchor={a.shape}, positive={p.shape}, negative={n.shape}")