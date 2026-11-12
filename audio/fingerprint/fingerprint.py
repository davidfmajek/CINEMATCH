"""
fingerprint.py -- Core Shazam-style fingerprinting logic.

Pipeline for one audio file:
    audio -> spectrogram -> peaks -> paired hashes

A "hash" encodes (freq1, freq2, delta_time) as a compact int.
Each hash maps to (movie_id, anchor_time) in the database.

Building the full multi-movie database is in build_database.py, which
imports this.
"""

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from itertools import combinations

# ---- Tunable constants (same intuition as explore_spectrogram.py) ----
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
NEIGHBORHOOD_SIZE = (20, 20)   # (freq_bins, time_bins) for local-max peak picking
# ---- Peak-picking threshold ----
# Instead of a fixed dB cutoff, I use the percentile of THIS signal's own
# loudness distribution. That way volume changes don't matter -- shifting
# all samples up or down just moves the whole distribution, and the
# percentile threshold moves with it. It also handles the gap between clean
# digital extractions and quieter mic recordings, which the fixed cutoff
# couldn't do (see FINDINGS.md).
PEAK_PERCENTILE = 95            # keep only the loudest 5% of bins as peak candidates
USE_ADAPTIVE_THRESHOLD = True   # set False to fall back to the fixed MIN_AMPLITUDE_DB below
MIN_AMPLITUDE_DB = -10          # only used if USE_ADAPTIVE_THRESHOLD is False

# ---- Peak-pairing (fingerprint) constants ----
FAN_OUT = 5                    # how many target peaks to pair with each anchor peak
MIN_TIME_DELTA = 0             # frames
MAX_TIME_DELTA = 200           # frames (~4.6s at hop_length=512, sr=22050), pairing window
FREQ_BITS = 11                 # bits used to encode each frequency bin in the hash
                                # (n_fft=2048 -> 1025 freq bins, needs 11 bits: 2^10=1024 is NOT enough)
TIME_DELTA_BITS = 12           # bits used to encode delta_time in the hash


def load_audio(path, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def compute_spectrogram(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # ref=1.0 instead of ref=np.max. np.max normalizes against the loudest
    # moment in this specific file, so a 2hr movie and a 10s clip end up on
    # completely different dB scales even when the underlying audio is
    # identical. Fixed reference means the same amplitude always maps to the
    # same dB value, regardless of what else is in the file.
    # This was the most impactful bug fix in the whole project (see FINDINGS.md).
    S_db = librosa.amplitude_to_db(S, ref=1.0)
    return S_db


def find_peaks(S_db, neighborhood_size=NEIGHBORHOOD_SIZE,
                use_adaptive_threshold=USE_ADAPTIVE_THRESHOLD,
                percentile=PEAK_PERCENTILE, min_amplitude_db=MIN_AMPLITUDE_DB):
    """
    Find local maxima in the spectrogram. Returns (freq_bin, time_bin) pairs.

    Threshold is either the percentile of this signal's own dB distribution
    (adaptive, default) or a fixed dB cutoff (use_adaptive_threshold=False).
    """
    # A point is a peak if it equals the max within its neighborhood
    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db

    if use_adaptive_threshold:
        threshold = np.percentile(S_db, percentile)
    else:
        threshold = min_amplitude_db

    loud_enough = S_db > threshold

    peak_mask = local_max & loud_enough
    freq_bins, time_bins = np.where(peak_mask)

    peaks = list(zip(freq_bins, time_bins))
    peaks.sort(key=lambda p: p[1])  # sort by time so pairing window logic is simple
    return peaks


def pair_peaks_to_hashes(peaks, fan_out=FAN_OUT,
                          min_time_delta=MIN_TIME_DELTA, max_time_delta=MAX_TIME_DELTA):
    """
    For each anchor peak, pair with up to `fan_out` nearby peaks that occur
    later in time (within the delta_time window). Each pair becomes a hash.

    Returns list of (hash_value, anchor_time_bin) tuples.
    """
    hashes = []
    n = len(peaks)

    for i in range(n):
        freq1, t1 = peaks[i]
        paired = 0

        for j in range(i + 1, n):
            freq2, t2 = peaks[j]
            delta_time = t2 - t1

            if delta_time < min_time_delta:
                continue
            if delta_time > max_time_delta:
                break  # peaks are sorted by time, so no more candidates for this anchor

            h = create_hash(freq1, freq2, delta_time)
            hashes.append((h, t1))

            paired += 1
            if paired >= fan_out:
                break

    return hashes


def create_hash(freq1, freq2, delta_time):
    """
    Pack (freq1, freq2, delta_time) into a single integer hash.
    Bit-packing keeps the hash compact and hashable as a dict key.
    """
    freq1 = int(freq1) & ((1 << FREQ_BITS) - 1)
    freq2 = int(freq2) & ((1 << FREQ_BITS) - 1)
    delta_time = int(delta_time) & ((1 << TIME_DELTA_BITS) - 1)

    h = (freq1 << (FREQ_BITS + TIME_DELTA_BITS)) | (freq2 << TIME_DELTA_BITS) | delta_time
    return h


def fingerprint_audio(y, sr=SAMPLE_RATE, percentile=PEAK_PERCENTILE):
    """
    Same pipeline as fingerprint_file, but takes an in-memory audio array
    instead of a file path. Used for stress-testing augmented clips without
    writing them to disk first, and for sweeping PEAK_PERCENTILE values.
    """
    S_db = compute_spectrogram(y)
    peaks = find_peaks(S_db, percentile=percentile)
    hashes = pair_peaks_to_hashes(peaks)
    return hashes


def fingerprint_file(path, percentile=PEAK_PERCENTILE):
    """
    Full pipeline for one audio file: load -> spectrogram -> peaks -> hashes.
    Returns list of (hash_value, anchor_time_bin) tuples.
    """
    y, sr = load_audio(path)
    return fingerprint_audio(y, sr, percentile=percentile)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fingerprint.py path/to/audio.wav")
        sys.exit(1)

    path = sys.argv[1]

    # Quick dB range check -- useful context, though the threshold is now
    # adaptive (percentile-based) by default rather than a fixed cutoff.
    y, sr = load_audio(path)
    S_db = compute_spectrogram(y)
    print(f"Spectrogram dB range: min={S_db.min():.1f}, max={S_db.max():.1f}, "
          f"median={np.median(S_db):.1f}")
    if USE_ADAPTIVE_THRESHOLD:
        adaptive_cutoff = np.percentile(S_db, PEAK_PERCENTILE)
        print(f"Adaptive threshold ({PEAK_PERCENTILE}th percentile): {adaptive_cutoff:.1f} dB")
    else:
        print(f"Fixed MIN_AMPLITUDE_DB threshold: {MIN_AMPLITUDE_DB}")
        print(f"Fraction of bins above threshold: {(S_db > MIN_AMPLITUDE_DB).mean():.2%}\n")

    print(f"Fingerprinting {path} ...")
    hashes = fingerprint_file(path)
    print(f"Generated {len(hashes)} hashes")
    print("Sample hashes (hash_value, anchor_time_bin):")
    for h, t in hashes[:10]:
        print(f"  {h}, {t}")