"""
Batch-extract full audio tracks from movie files for the fingerprint database.

Setup:
    1. Install ffmpeg (must be on PATH):
         macOS:  brew install ffmpeg
         Ubuntu: sudo apt install ffmpeg

Usage:
    1. Edit the MOVIES_DIR and OUTPUT_DIR paths below.
    2. Run:  python extract_audio.py

    Movies may be loose files or one-per-folder (Title (Year)/video.mkv).
    Nested files are named after their parent folder in OUTPUT_DIR.
    Output is mono 22.05kHz WAV (librosa.load() defaults).

    Skips a file if the output .wav already exists, so it's safe to re-run
    after adding new movies.
"""

import subprocess
import shutil
import sys
from pathlib import Path


MOVIES_DIR = Path("C:/Users/David/OneDrive/Documents/movies")
OUTPUT_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".wmv", ".flv"}
SAMPLE_RATE = 22050


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it first:")
        print("  macOS:  brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        sys.exit(1)


def extract_audio(input_path: Path, output_path: Path) -> bool:
    """Extract mono 22.05kHz WAV audio from a video file. Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",                  # overwrite output if present
        "-i", str(input_path),
        "-vn",                 # no video
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",            # mono
        "-loglevel", "error",  # keep output quiet, only show real errors
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {input_path.name}")
        print(f"  ffmpeg error: {result.stderr.strip()[:300]}")
        return False
    return True


def main():
    movies_dir = MOVIES_DIR
    output_dir = OUTPUT_DIR

    if not movies_dir.is_dir():
        print(f"ERROR: {movies_dir} is not a valid directory")
        print("Edit MOVIES_DIR at the top of this script.")
        sys.exit(1)

    check_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)

    movie_files = sorted(
        f for f in movies_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not movie_files:
        print(f"No video files found in {movies_dir} (including subfolders)")
        print(f"Looked for extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        sys.exit(1)

    print(f"Found {len(movie_files)} movie file(s) under {movies_dir}\n")

    succeeded, skipped, failed = [], [], []

    for movie_path in movie_files:
        # Loose file -> stem; nested -> parent folder name (e.g. "Barbie (2023)")
        label = (
            movie_path.stem
            if movie_path.parent == movies_dir
            else movie_path.parent.name
        )
        output_path = output_dir / f"{label}.wav"
        display = f"{movie_path.parent.name}/{movie_path.name}" if movie_path.parent != movies_dir else movie_path.name

        if output_path.exists():
            print(f"SKIP (already exists): {display}")
            skipped.append(display)
            continue

        print(f"Extracting: {display} -> {output_path.name}")
        if extract_audio(movie_path, output_path):
            succeeded.append(display)
        else:
            failed.append(display)

    print("\n--- Summary ---")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Skipped (already existed): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()