"""
diagnose_failure.py -- Figure out WHY mic_recording_02 failed to match.

Two possible causes:
  (A) Bad embedding: the model's embeddings for this clip just aren't close
      to the right region in Parasite's embedding space. The model didn't
      recognize the content at all.
  (B) Bad matching: the embeddings are close enough, but the offset voting
      scattered the signal too thin. A tuning problem, not a representation
      problem.

I already know from the hash system where the true match is (Parasite,
~3087.7s), so I can check similarity directly against that region without
going through the voting step.

Usage:
    python diagnose_failure.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dataset import WINDOW_SEC, load_window, to_spectrogram_tensor
from model.model import NeuralFingerprinter
from model.match_embedding import load_db, embed_query, CHECKPOINT_PATH

QUERY_PATH_01 = Path("C:/Users/David/CINEMATCH/data/clips/mic_recording_01.wav")
QUERY_PATH_02 = Path("C:/Users/David/CINEMATCH/data/clips/mic_recording_02.wav")

# Known ground truth from the hash-based system's results (see FINDINGS.md)
KNOWN_OFFSET_01_SEC = 1905.0
KNOWN_OFFSET_02_SEC = 3087.7
PARASITE_TITLE = "Parasite (2019)"


def analyze_recording(model, device, query_path, known_offset_sec, db_embeddings,
                       db_movie_ids, db_start_times, movie_names, label):
    print(f"\n{'='*70}")
    print(f"ANALYZING: {label} ({query_path.name})")
    print(f"{'='*70}")

    # --- Basic audio stats ---
    info = sf.info(str(query_path))
    duration = info.frames / info.samplerate
    y_full, sr = sf.read(str(query_path))
    if y_full.ndim > 1:
        y_full = y_full.mean(axis=1)
    rms = np.sqrt(np.mean(y_full.astype(np.float64) ** 2))
    print(f"Duration: {duration:.1f}s | RMS energy: {rms:.4f}")

    # --- Embed the query ---
    query_embeddings, query_starts = embed_query(model, device, query_path)
    print(f"Query windows: {len(query_embeddings)}")

    # --- Find the Parasite movie_id ---
    title_to_id = {t: mid for mid, t in movie_names.items()}
    parasite_id = title_to_id.get(PARASITE_TITLE)
    if parasite_id is None:
        print(f"ERROR: '{PARASITE_TITLE}' not found in database")
        return

    # --- Get all DB embeddings belonging to Parasite, near the known offset ---
    parasite_mask = db_movie_ids == parasite_id
    parasite_embeddings = db_embeddings[parasite_mask]
    parasite_starts = db_start_times[parasite_mask]

    near_mask = np.abs(parasite_starts - known_offset_sec) < 15.0  # +/- 15s window
    near_embeddings = parasite_embeddings[near_mask]
    near_starts = parasite_starts[near_mask]

    print(f"\nGround-truth region: Parasite @ ~{known_offset_sec:.1f}s "
          f"({len(near_embeddings)} DB windows within +/-15s)")

    if len(near_embeddings) == 0:
        print("WARNING: no DB windows found near the known offset -- can't compare")
        return

    # Direct similarity against the known true region, bypassing voting
    # entirely -- this tells me if the embedding itself is the problem.
    sims_to_truth = query_embeddings @ near_embeddings.T  # (Q, near_N)
    best_sim_per_query = sims_to_truth.max(axis=1)

    print(f"\nSimilarity of each query window to its TRUE matching region:")
    print(f"  Mean: {best_sim_per_query.mean():.3f}")
    print(f"  Max:  {best_sim_per_query.max():.3f}")
    print(f"  Min:  {best_sim_per_query.min():.3f}")

    # Rank ALL movies, not just one random comparison -- a single movie can
    # be an "embedding hub" that looks falsely similar to everything.
    print(f"\nRanking ALL {len(movie_names)} movies by mean max-similarity:")
    movie_scores = []
    for mid in sorted(movie_names.keys()):
        mask = db_movie_ids == mid
        movie_embs = db_embeddings[mask]
        sims = query_embeddings @ movie_embs.T
        mean_best_sim = sims.max(axis=1).mean()
        movie_scores.append((mid, movie_names[mid], mean_best_sim))

    movie_scores.sort(key=lambda x: x[2], reverse=True)
    parasite_rank = next(i for i, (mid, _, _) in enumerate(movie_scores, 1) if mid == parasite_id)

    for rank, (mid, title, score) in enumerate(movie_scores[:5], 1):
        marker = "  <-- TRUE MATCH" if mid == parasite_id else ""
        print(f"  #{rank}: {title:<40} {score:.3f}{marker}")
    if parasite_rank > 5:
        true_score = next(s for mid, t, s in movie_scores if mid == parasite_id)
        print(f"  ...")
        print(f"  #{parasite_rank}: {PARASITE_TITLE:<40} {true_score:.3f}  <-- TRUE MATCH")

    print(f"\n--- DIAGNOSIS ---")
    print(f"True match ranks #{parasite_rank} out of {len(movie_names)} movies by embedding similarity.")
    if parasite_rank == 1:
        print("Embedding correctly ranks the true match FIRST.")
        print("If final matching still failed, this points to a VOTING/tolerance issue (B).")
    elif parasite_rank <= 3:
        print("Embedding ranks the true match near the top, but not clearly first.")
        print("This is a BORDERLINE case -- the representation has signal, but it's not decisive.")
    else:
        print(f"Embedding does NOT clearly favor the true match (rank #{parasite_rank}).")
        print("This points to a genuine EMBEDDING problem (A) for this specific query.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralFingerprinter().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    db_embeddings, db_movie_ids, db_start_times, movie_names = load_db()
    print(f"Database: {len(movie_names)} movies, {len(db_embeddings)} windows")

    analyze_recording(model, device, QUERY_PATH_01, KNOWN_OFFSET_01_SEC,
                       db_embeddings, db_movie_ids, db_start_times, movie_names,
                       "mic_recording_01 (SUCCEEDED)")
    analyze_recording(model, device, QUERY_PATH_02, KNOWN_OFFSET_02_SEC,
                       db_embeddings, db_movie_ids, db_start_times, movie_names,
                       "mic_recording_02 (FAILED)")


if __name__ == "__main__":
    main()