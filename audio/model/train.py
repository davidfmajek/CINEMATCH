"""
train.py -- Train the NeuralFingerprinter with triplet loss.

Triplet loss (from FaceNet, also used in the Now Playing paper): for each
(anchor, positive, negative) triplet, push
    distance(anchor, positive) + margin < distance(anchor, negative)
Same audio (clean or distorted) should cluster together; different audio
should stay apart.

Usage:
    1. Edit RAW_AUDIO_DIR and CHECKPOINT_DIR below.
    2. Run:  python train.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dataset import TripletAudioDataset
from model.model import NeuralFingerprinter

# ---- EDIT THESE ----
RAW_AUDIO_DIR = Path("C:/Users/David/CINEMATCH/data/raw_audio")
CHECKPOINT_DIR = Path("C:/Users/David/CINEMATCH/data/checkpoints")
# ----------------------

BATCH_SIZE = 64
SAMPLES_PER_EPOCH = 15000    # increased from 5000 -- previous run plateaued too early,
                               # and diagnostics showed the model needs more exposure to
                               # harder (quieter) content, which is now oversampled in
                               # dataset.py's hard-example biasing
NUM_EPOCHS = 60               # increased from 30 for the same reason
LEARNING_RATE = 1e-3
MARGIN = 0.3                 # triplet loss margin -- tune if training collapses or plateaus early
NUM_WORKERS = 4               # parallel data loading; lower this if you hit CPU/IO bottlenecks

RESUME_FROM_CHECKPOINT = Path("C:/Users/David/CINEMATCH/data/checkpoints/fingerprinter_latest.pt")
# Set to None to train from scratch instead of continuing from round 1


def check_gpu():
    print("=" * 50)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU DETECTED: {gpu_name}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Total VRAM: {total_mem_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected -- training will run on CPU.")
        print("This will be MUCH slower (likely 10-50x). If you have an")
        print("NVIDIA GPU, check that torch was installed with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 50)
    return device


def train():
    device = check_gpu()

    movie_paths = sorted(RAW_AUDIO_DIR.glob("*.wav"))
    if not movie_paths:
        print(f"ERROR: no .wav files found in {RAW_AUDIO_DIR}")
        return
    print(f"Training on {len(movie_paths)} movies")

    dataset = TripletAudioDataset(movie_paths, samples_per_epoch=SAMPLES_PER_EPOCH)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
    )

    model = NeuralFingerprinter().to(device)

    if RESUME_FROM_CHECKPOINT is not None and RESUME_FROM_CHECKPOINT.is_file():
        model.load_state_dict(torch.load(RESUME_FROM_CHECKPOINT, map_location=device))
        print(f"Resumed from checkpoint: {RESUME_FROM_CHECKPOINT}")
    else:
        print("Training from scratch (no checkpoint loaded)")

    print(f"Model device: {next(model.parameters()).device}")
    if device.type == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct_margin = 0  # how often positive really is closer than negative
        n_batches = 0

        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = triplet_loss_fn(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                dist_pos = (emb_anchor - emb_positive).pow(2).sum(dim=1)
                dist_neg = (emb_anchor - emb_negative).pow(2).sum(dim=1)
                total_correct_margin += (dist_pos < dist_neg).sum().item()

        avg_loss = total_loss / n_batches
        accuracy = total_correct_margin / (n_batches * BATCH_SIZE)

        gpu_info = ""
        if device.type == "cuda":
            gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
            gpu_info = f" | GPU mem={gpu_mem:.0f}MB"

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | loss={avg_loss:.4f} | "
              f"pos<neg accuracy={accuracy:.1%}{gpu_info}")

        # Save a checkpoint every few epochs, plus always overwrite "latest"
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            checkpoint_path = CHECKPOINT_DIR / f"fingerprinter_r2_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    final_path = CHECKPOINT_DIR / "fingerprinter_r2_latest.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train()