"""
match_embedding.py -- Identify a query clip using the neural fingerprinter.

Same offset-consistency voting as match.py, but in embedding space instead
of hash space. For each query window, find its nearest neighbors in the
embedding database (cosine similarity), then vote on (movie, offset) pairs.

Usage:
    python match_embedding.py path/to/query_clip.wav
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dataset import WINDOW_SEC, load_window, to_spectrogram_tensor
from model.model import NeuralFingerprinter
import soundfile as sf

CHECKPOINT_PATH = Path("C:/Users/David/CINEMATCH/data/checkpoints/fingerprinter_r2_latest.pt")
DB_PATH = Path("C:/Users/David/CINEMATCH/data/embedding_db_r2.npz")

QUERY_HOP_SEC = 0.5     # how densely to sample windows from the query clip
TOP_K = 3                # nearest neighbors to consider per query window
OFFSET_TOLERANCE = 1.0   # seconds -- offsets within this range count as "the same" offset
TOP_N_RESULTS = 5


def load_db():
    data = np.load(DB_PATH, allow_pickle=True)
    embeddings = data["embeddings"]          # (N, dim)
    movie_ids = data["movie_ids"]            # (N,)
    start_times = data["start_times"]        # (N,)
    movie_names = dict(data["movie_names"])  # {movie_id: title}
    return embeddings, movie_ids, start_times, movie_names


def embed_query(model, device, query_path: Path):
    duration = sf.info(str(query_path)).frames / sf.info(str(query_path)).samplerate
    starts = np.arange(0, max(0.01, duration - WINDOW_SEC), QUERY_HOP_SEC)
    if len(starts) == 0:
        starts = np.array([0.0])

    tensors = []
    for start in starts:
        y = load_window(query_path, start)
        tensors.append(to_spectrogram_tensor(y))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeddings = model(batch).cpu().numpy()
    return embeddings, starts


def match(query_embeddings, query_starts, db_embeddings, db_movie_ids, db_start_times, movie_names):
    """
    For each query window, find its TOP_K nearest neighbors in the DB by
    cosine similarity (just a dot product since embeddings are L2-normalized).
    Each neighbor votes for (movie_id, rounded_offset).
    """
    similarities = query_embeddings @ db_embeddings.T  # (Q, N) cosine similarities

    offset_votes = defaultdict(Counter)  # movie_id -> Counter(rounded_offset -> count)
    total_hits = defaultdict(int)

    for q_idx in range(similarities.shape[0]):
        query_start = query_starts[q_idx]
        top_k_idx = np.argpartition(-similarities[q_idx], TOP_K)[:TOP_K]

        for db_idx in top_k_idx:
            movie_id = int(db_movie_ids[db_idx])
            db_start = db_start_times[db_idx]
            offset = db_start - query_start
            # Round offset to the nearest OFFSET_TOLERANCE bucket so that
            # near-identical offsets (e.g. 31.5s vs 31.7s) count as the same vote
            rounded_offset = round(offset / OFFSET_TOLERANCE) * OFFSET_TOLERANCE
            offset_votes[movie_id][rounded_offset] += 1
            total_hits[movie_id] += 1

    results = []
    for movie_id, votes in offset_votes.items():
        best_offset, best_count = votes.most_common(1)[0]
        results.append((
            movie_id,
            movie_names[movie_id],
            best_offset,
            best_count,
            total_hits[movie_id],
        ))
    results.sort(key=lambda r: r[3], reverse=True)
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python match_embedding.py path/to/query_clip.wav")
        sys.exit(1)

    query_path = Path(sys.argv[1])
    if not query_path.is_file():
        print(f"ERROR: {query_path} not found")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NeuralFingerprinter().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    print(f"Loading embedding database from {DB_PATH} ...")
    db_embeddings, db_movie_ids, db_start_times, movie_names = load_db()
    print(f"Database: {len(movie_names)} movies, {len(db_embeddings)} windows\n")

    print(f"Embedding query: {query_path}")
    query_embeddings, query_starts = embed_query(model, device, query_path)
    print(f"Query produced {len(query_embeddings)} window embeddings\n")

    results = match(query_embeddings, query_starts, db_embeddings, db_movie_ids,
                     db_start_times, movie_names)

    if not results:
        print("No matches found.")
        return

    print(f"Top {min(TOP_N_RESULTS, len(results))} matches:\n")
    print(f"{'Rank':<5}{'Movie':<45}{'Votes':<8}{'TotalHits':<10}{'Confidence':<10}")
    print("-" * 80)
    for rank, (movie_id, title, offset, votes, total_hits) in enumerate(results[:TOP_N_RESULTS], 1):
        confidence = votes / total_hits if total_hits else 0
        print(f"{rank:<5}{title:<45}{votes:<8}{total_hits:<10}{confidence:<10.2%}")
        if rank == 1:
            print(f"      -> Best guess: ~{offset:.1f}s ({offset/60:.1f} min) into '{title}'")


if __name__ == "__main__":
    main()