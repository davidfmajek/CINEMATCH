"""
model.py -- CNN encoder for audio fingerprinting, based on the architecture
in the Now Playing paper (Google, NeurIPS 2017).

Input: normalized spectrogram (1, 1025, 44). Output: 64-dim L2-normalized
embedding. Trained with triplet loss so the same audio (clean or degraded)
maps to nearby embeddings, and different audio maps to distant ones.

I simplified quite a bit vs. the paper since I'm working with 30 movies on
a desktop GPU, not 10k songs on a mobile chip -- no divide-and-encode block,
fewer layers, no quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 64  # smaller than the paper's 96/128, appropriate for a 30-movie dataset


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ELU -> (optional) downsample via stride."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)))


class NeuralFingerprinter(nn.Module):
    """
    Input:  (batch, 1, 1025, 44)  -- normalized dB spectrogram
    Output: (batch, EMBEDDING_DIM) -- L2-normalized embedding
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()

        # Each ConvBlock halves both spatial dimensions (stride=2).
        # 1025 x 44 -> 513x22 -> 257x11 -> 129x6 -> 65x3 -> 33x2
        self.conv_stack = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

        # Adaptive pooling so I'm not hardcoding a flatten size that breaks
        # if WINDOW_SEC or HOP_LENGTH changes.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        flat_size = 128 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ELU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        # L2-normalize so all embeddings live on the unit hypersphere --
        # cosine similarity then equals dot product, and the triplet margin
        # has a consistent scale.
        x = F.normalize(x, p=2, dim=1)
        return x


if __name__ == "__main__":
    # Smoke test: confirm shapes flow through correctly end-to-end.
    model = NeuralFingerprinter()
    dummy_input = torch.randn(8, 1, 1025, 44)  # batch of 8
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output L2 norms (should all be ~1.0): {output.norm(dim=1)[:5]}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")