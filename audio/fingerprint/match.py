"""
match.py -- Match a query clip against the fingerprint database.

The key idea (from Wang 2003): if a clip is really from a movie, all its
matching hashes should agree on the same time offset (db_time - query_time).
Random collisions scatter across random offsets. So I just find the offset
with the most votes -- that's the spike, and the movie it points to is the
match.

Usage:
    1. Edit DB_PATH and QUERY_CLIP_PATH below.
    2. Run:  python match.py
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from fingerprint.fingerprint import fingerprint_file, HOP_LENGTH, SAMPLE_RATE

# ---- EDIT THESE ----
DB_PATH = Path("C:/Users/David/CINEMATCH/data/fingerprint_db.pkl")
QUERY_CLIP_PATH = Path("C:/Users/David/CINEMATCH/data/clips/test_clip_01.wav")
TOP_N_RESULTS = 5
# ----------------------


def load_database(db_path: Path):
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    return db["hashes"], db["movie_names"]


def match_clip(query_hashes, hash_db, movie_names):
    """
    For each query hash, look it up in the database. For every hit, compute
    the time offset and tally votes per (movie_id, offset) pair.

    Returns a list of (movie_id, movie_title, best_offset, vote_count,
    total_matching_hashes) sorted by vote_count descending.
    """
    # offset_votes[movie_id][offset] = count
    offset_votes = defaultdict(Counter)
    total_hash_hits = defaultdict(int)

    for query_hash, query_time in query_hashes:
        matches = hash_db.get(query_hash)
        if not matches:
            continue

        for movie_id, db_time in matches:
            offset = db_time - query_time
            offset_votes[movie_id][offset] += 1
            total_hash_hits[movie_id] += 1

    results = []
    for movie_id, votes in offset_votes.items():
        best_offset, best_count = votes.most_common(1)[0]
        results.append((
            movie_id,
            movie_names[movie_id],
            best_offset,
            best_count,
            total_hash_hits[movie_id],
        ))

    results.sort(key=lambda r: r[3], reverse=True)  # sort by vote_count (spike height)
    return results


def frames_to_seconds(frames):
    return frames * HOP_LENGTH / SAMPLE_RATE


def main():
    print(f"Loading database from {DB_PATH} ...")
    hash_db, movie_names = load_database(DB_PATH)
    print(f"Database: {len(movie_names)} movies, {len(hash_db)} unique hashes\n")

    print(f"Fingerprinting query clip: {QUERY_CLIP_PATH}")
    query_hashes = fingerprint_file(str(QUERY_CLIP_PATH))
    print(f"Query generated {len(query_hashes)} hashes\n")

    results = match_clip(query_hashes, hash_db, movie_names)

    if not results:
        print("No matches found at all. Clip may not be in the database,")
        print("or peak-picking params are too different between full movies and clips.")
        return

    print(f"Top {min(TOP_N_RESULTS, len(results))} matches:\n")
    print(f"{'Rank':<5}{'Movie':<45}{'Votes':<8}{'TotalHits':<10}{'Confidence':<10}")
    print("-" * 80)

    for rank, (movie_id, title, offset, votes, total_hits) in enumerate(results[:TOP_N_RESULTS], 1):
        confidence = votes / total_hits if total_hits else 0
        offset_sec = frames_to_seconds(offset)
        print(f"{rank:<5}{title:<45}{votes:<8}{total_hits:<10}{confidence:<10.2%}")
        if rank == 1:
            print(f"      -> Best guess: clip starts at ~{offset_sec:.1f}s "
                  f"({offset_sec/60:.1f} min) into '{title}'")

    print("\nInterpretation:")
    print("- 'Votes' = size of the largest offset spike (real match strength)")
    print("- 'TotalHits' = all raw hash collisions, including scattered noise")
    print("- 'Confidence' = Votes/TotalHits -- closer to 1.0 means a sharper, more")
    print("  convincing spike. A true match should have MUCH higher Votes than")
    print("  runner-ups, and high confidence. A close race between top movies")
    print("  suggests either an ambiguous clip or params that need tuning.")


if __name__ == "__main__":
    main()