"""
explore_spectrogram.py -- Visualize spectrograms and peak-picking to get a
feel for what the fingerprinter is actually working with.

Saves three images: waveform, spectrogram, and the constellation map (peaks
overlaid). Useful for tuning neighborhood_size and min_amplitude_db before
committing to a value.

Note: this uses ref=np.max in amplitude_to_db (normalized per-file), which
is fine here because I'm just looking at one file at a time. The production
fingerprinter uses ref=1.0 so clips and movies end up on the same scale.
"""

import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter


def load_audio(path, sr=22050):
    """Load audio, resampled to sr (mono)."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    print(f"Loaded {path}: {len(y)/sr:.2f}s, sr={sr}")
    return y, sr


def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """Compute magnitude spectrogram in dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, hop_length


def find_peaks(S_db, neighborhood_size=(20, 20), min_amplitude_db=-40):
    """Find local maxima above min_amplitude_db. Returns (freq_bin, time_bin) pairs."""
    # A point is a peak if it equals the max within its neighborhood
    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db

    # Filter out quiet regions (silence/noise floor shouldn't count as peaks)
    loud_enough = S_db > min_amplitude_db

    peak_mask = local_max & loud_enough
    freq_bins, time_bins = np.where(peak_mask)

    peaks = list(zip(freq_bins, time_bins))
    print(f"Found {len(peaks)} peaks")
    return peaks


def plot_waveform(y, sr):
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig("1_waveform.png", dpi=120)
    plt.close()
    print("Saved 1_waveform.png")


def plot_spectrogram(S_db, sr, hop_length):
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                              x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig("2_spectrogram.png", dpi=120)
    plt.close()
    print("Saved 2_spectrogram.png")


def plot_constellation(S_db, peaks, sr, hop_length):
    """Overlay detected peaks on top of the spectrogram."""
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                              x_axis="time", y_axis="log", cmap="gray_r")

    freq_bins = np.array([p[0] for p in peaks])
    time_bins = np.array([p[1] for p in peaks])

    # Convert bin indices to actual time (s) and frequency (Hz) for plotting
    times = librosa.frames_to_time(time_bins, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(S_db.shape[0] - 1) * 2)[freq_bins]

    plt.scatter(times, freqs, color="red", s=8, marker="x")
    plt.title(f"Constellation Map ({len(peaks)} peaks)")
    plt.tight_layout()
    plt.savefig("3_constellation.png", dpi=120)
    plt.close()
    print("Saved 3_constellation.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_spectrogram.py path/to/audio.wav")
        sys.exit(1)

    path = sys.argv[1]
    y, sr = load_audio(path)

    plot_waveform(y, sr)

    S_db, hop_length = compute_spectrogram(y, sr)
    plot_spectrogram(S_db, sr, hop_length)

    # Tune these two params -- this is where you'll spend real experimentation time:
    # - neighborhood_size controls how "sparse" the peaks are (bigger = fewer, more dominant peaks)
    # - min_amplitude_db filters out quiet noise-floor peaks
    peaks = find_peaks(S_db, neighborhood_size=(20, 20), min_amplitude_db=-40)
    plot_constellation(S_db, peaks, sr, hop_length)

    print("\nDone. Look at 3_constellation.png -- the red x's are what")
    print("Shazam-style hashing will later pair up into fingerprints.")


if __name__ == "__main__":
    main()