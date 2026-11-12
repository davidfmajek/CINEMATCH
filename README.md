# CINEMATCH

Shazam exists for music. I wanted to build the same thing for movies: point a phone at a TV and find out what's playing and where you are in it. That's the goal.

The project is as much about learning as it is about building. I didn't start by grabbing a fingerprinting library. I started by reading Avery Wang's original Shazam paper from 2003, implementing it from scratch, then reading how Google tackled a harder version of the same problem in 2017, and implementing that too. Each phase is meant to actually understand the technique, not just get it working.

The longer-term direction is to move beyond audio entirely, identifying a movie from a still frame or a short 3-15 second video clip using visual embeddings rather than audio fingerprints. That's not built yet, but it's where this is heading.

---

## What I built so far

### Phase 1: Hash-based audio fingerprinting

The same approach Shazam uses, implemented from Wang's paper.

The idea: convert audio to a spectrogram, find the loudest distinctive peaks, and pair them into compact hashes. Each hash encodes two frequencies and the time gap between them. Store all those hashes for every movie in a database. When you want to identify something, fingerprint the query clip and look up its hashes. If the clip really is from a movie, all the matching hashes should agree on the same time offset into that movie. Random collisions scatter randomly and cancel out. The movie with the sharpest spike in the offset histogram wins.

It's elegant because it's robust to noise by design and can match in milliseconds with a simple dictionary lookup.

```
audio file
     |
     v
  [ STFT ]  ->  magnitude spectrogram  (1025 freq bins x time frames)
     |
     v
  [ peak picking ]  ->  local maxima above the 95th-percentile threshold
     |
     v
  [ pair peaks ]  ->  hash(freq1, freq2, delta_time)  for each anchor peak
     |
     v
  [ database lookup ]  ->  list of (movie_id, db_time) per hash hit
     |
     v
  [ offset voting ]  ->  offset = db_time - query_time, tally per movie
     |
     v
  movie with the tallest offset spike  +  timestamp
```

### Phase 2: Neural audio fingerprinting

After getting the hash system working and understanding its limits (pitch shift breaks it completely (see `FINDINGS.md`)), I wanted to see how a learned approach handles the same problem.

I based this on Google's "Now Playing" paper (NeurIPS 2017), which uses a CNN to map short audio windows to compact embedding vectors. The model is trained with triplet loss: embeddings of the *same* audio (whether clean or degraded) should be close together; embeddings of *different* audio should be far apart. Once trained, identification works the same way as Phase 1, just nearest-neighbor lookup in embedding space instead of exact hash lookup.

I trained two rounds. Round 2 added **hard-example mining** after I noticed the model was coasting on easy, loud content and failing on quieter recordings.

```
                         -- training --
  (anchor, positive, negative) triplets
  anchor   = clean 1s window from a movie
  positive = same window + random augmentation (noise / pitch / speed / volume)
  negative = different window (different movie or far-away in the same one)
       |
       v
  [ CNN encoder ]  (same weights for all three)
       |
       v
  triplet loss: distance(anchor, positive) + margin < distance(anchor, negative)


                         -- inference --
  query clip
       |
       v
  sliding 1s windows  (0.5s hop)
       |
       v
  [ CNN encoder ]
       |
       v
  64-dim L2-normalized embeddings
       |
       v
  nearest-neighbor search in embedding database  (cosine similarity)
       |
       v
  offset voting  ->  movie + timestamp


                         -- CNN encoder --
  input  (1, 1025, 44)
       |
  Conv(1   -> 32,  stride 2)  + BN + ELU  ->  (32,  513, 22)
  Conv(32  -> 64,  stride 2)  + BN + ELU  ->  (64,  257, 11)
  Conv(64  -> 128, stride 2)  + BN + ELU  ->  (128, 129,  6)
  Conv(128 -> 128, stride 2)  + BN + ELU  ->  (128,  65,  3)
  Conv(128 -> 128, stride 2)  + BN + ELU  ->  (128,  33,  2)
       |
  AdaptiveAvgPool  ->  (128, 4, 4)
  Flatten          ->  2048
  FC(2048 -> 256)  + ELU
  FC(256  ->  64)
  L2 normalize
       |
  output  64-dim embedding  (~930K parameters total)
```

### Phase 3: Visual identification (not started)

The next step is identifying movies from images or short clips rather than audio. A still frame or a few seconds of video should be enough to identify a movie. The visual style, color grading, and scene composition are as distinctive as the audio. This phase will likely use visual embeddings trained similarly to Phase 2, but the input changes from spectrograms to frames.

```
                         -- training --
  (anchor, positive, negative) triplets
  anchor   = frame or short clip from a movie
  positive = same frame/clip + visual augmentation (color jitter, brightness,
             crop, compression artifacts to simulate a phone pointed at a screen)
  negative = frame/clip from a different movie or distant timestamp
       |
       v
  [ visual encoder ]
       |
       v
  triplet loss: same idea as Phase 2, different input modality


                         -- inference (still frame) --
  single frame
       |
       v
  resize + normalize
       |
       v
  [ visual encoder ]  ->  embedding
       |
       v
  nearest-neighbor search in frame database
       |
       v
  movie + approximate timestamp


                         -- inference (short video clip) --
  3-15 second clip
       |
       v
  sample N frames  (e.g. 1 per second)
       |
       v
  [ visual encoder ]  ->  embedding per frame
       |
       v
  aggregate / vote across frames
       |
       v
  movie + timestamp


                         -- visual encoder (planned) --
  input  (3, H, W)  RGB frame
       |
  pretrained CNN backbone  (e.g. ResNet or EfficientNet)
  fine-tuned with triplet loss on movie frames
       |
  FC projection  ->  128-dim L2-normalized embedding
       |
  output  128-dim embedding
```

---

## Results

### Classical system

I ran a stress test across 6 movies (Inception, Whiplash, The Matrix, Fight Club, The Grand Budapest Hotel, The Good the Bad and the Ugly), 3 random 10-second clips per movie, 10 augmentation types. 180 trials total.

| Augmentation | Accuracy |
|---|---|
| Clean | 83% |
| Volume −10 dB | 83% |
| Volume −20 dB | 83% |
| Noise 20 dB SNR | 83% |
| Noise 10 dB SNR | 83% |
| Noise 0 dB SNR | 83% |
| Speed 0.95x | 78% |
| Speed 1.05x | 67% |
| Pitch +1 semitone | 11% |
| Pitch −1 semitone | 0% |

The volume invariance is exact, it's a mathematical consequence of how the peak threshold works (see below). The noise tolerance surprised me; at 0 dB SNR the signal and noise have equal power, it sounds awful, and it still matched correctly 83% of the time.

Pitch shift completely breaks it, which I expected once I understood why (see `FINDINGS.md`).

### Neural system

I tested on two real microphone recordings of Parasite (2019):

| Test | Result |
|---|---|
| mic_recording_01 | Correct (Parasite ranked #1) |
| mic_recording_02 (quieter scene) | Failed |

Round 2 hard-example mining is what got mic_recording_01 working; Round 1 missed it. mic_recording_02 is a quieter scene and I haven't solved it yet. I wrote a diagnostic script (`diagnose_failure.py`) to figure out if the problem was in the embeddings or the matching step. Turns out the embeddings themselves are off, not the voting. See `FINDINGS.md`.

---

## Things I had to figure out

**The normalization bug (`ref=np.max`):** This took me an embarrassingly long time to find. `librosa.amplitude_to_db(S, ref=np.max)` normalizes the spectrogram against *its own* loudest moment, so a 2-hour movie and a 10-second clip from it end up on completely different dB scales, even though the audio is identical. The fix is `ref=1.0`, a fixed reference so the same absolute amplitude always maps to the same dB value. Match confidence improved dramatically after this one change.

**Adaptive peak threshold:** I started with a fixed dB cutoff to decide which peaks to keep. That broke completely on mic recordings because they're quieter than digital extractions. The fix: use the 95th percentile of *this specific signal's* dB distribution as the threshold. It self-calibrates, so volume changes are irrelevant, which is exactly why the stress test shows identical accuracy for clean and ±20 dB volume.

**On-the-fly augmentation:** For neural training, I never write augmented audio to disk. Each triplet is generated fresh: load a short window via soundfile (which does partial file reads, so I'm not loading a 2-hour movie into RAM) and apply a random augmentation in memory. Effectively infinite training variety from 20 movie files.

**Shortcut learning:** After Round 1, the model worked fine on clean audio but struggled on mic recordings. The issue was that it had learned to identify loud, easy-to-distinguish content well while performing poorly on quieter windows, which is exactly what mic recordings of movies tend to have more of. It had picked up "loud = identifiable" as a shortcut rather than genuinely learning to match content across distortions. Round 2 hard-example mining directly targeted this.

---

## File layout

```
audio/
  fingerprint/
    fingerprint.py        Core fingerprinting (STFT -> peaks -> hashes)
    build_database.py     Build the hash database from raw audio
    match.py              Matching logic (offset-consistency voting)
    identify.py           CLI: identify a recording using the hash system

  model/
    model.py              NeuralFingerprinter CNN (~930K params)
    dataset.py            Triplet sampling with hard-example mining
    train.py              Training script (two rounds)
    build_embedding_db.py Build the embedding database from a checkpoint
    match_embedding.py    Neural matching (nearest-neighbor + offset voting)

  tools/
    stress_test.py        Automated accuracy test across movies and augmentations
    diagnose_failure.py   Figure out if a failure is an embedding or voting problem
    sweep_percentile.py   Sweep PEAK_PERCENTILE to find the best threshold
    test_pitch_invariance.py  Compare hash vs. neural on pitch-shifted audio

  utils/
    extract_audio.py      Extract audio from video files via ffmpeg
    make_test_clip.py     Make test clips from raw audio
    explore_spectrogram.py    Visualize spectrograms and peak-picking

data/
  raw_audio/              Full-movie WAV files (not in repo)
  clips/                  Test recordings and mic recordings
  fingerprint_db.pkl      Hash database
  embedding_db_r2.npz     Neural embedding database (Round 2)
  checkpoints/            Model checkpoints
  stress_test_results.csv Per-clip results for the classical system
```

---

## Running it

**Identify something (classical):**
```
cd audio
python fingerprint/identify.py path/to/recording.wav
```

**Identify something (neural):**
```
cd audio
python model/match_embedding.py path/to/recording.wav
```

**Re-run the stress test:**
```
cd audio
python tools/stress_test.py
```

**Re-train:**
Edit `RAW_AUDIO_DIR` and `CHECKPOINT_DIR` in `model/train.py`, then:
```
cd audio
python model/train.py
```

---

## References

- Wang, A. (2003). *An Industrial-Strength Audio Search Algorithm.* ISMIR.
- Google. (2017). *Now Playing: Continuous Low-Power Music Recognition.* NeurIPS.
- Schroff, F., Kalenichenko, D., Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR.
