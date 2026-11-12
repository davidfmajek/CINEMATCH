# FINDINGS

Notes on what I tried, what broke, what I fixed, and what I measured. Written as I went so I'd remember the reasoning later.

---

## Phase 1: Classical Fingerprinting

### How it works

Standard Shazam pipeline (Wang 2003):

1. STFT (n_fft=2048, hop_length=512, sr=22050)
2. Convert amplitude spectrogram to dB
3. Peak-pick: find local maxima in the spectrogram
4. Pair peaks into (freq1, freq2, delta_time) hashes (the "constellation map")
5. Query time: look up hashes in the database, compute time offsets, vote on (movie, offset) pairs

The voting is the key idea: a genuine match means all the matching hash pairs agree on the same offset. Random collisions scatter. So you're looking for a spike.

### Bug 1: `ref=np.max` in `amplitude_to_db`

This was the most impactful fix in Phase 1 and took me too long to find.

The default librosa call is `librosa.amplitude_to_db(S, ref=np.max)`, which normalizes the spectrogram against the loudest sample in *this specific signal*. The problem: a 2-hour movie and a 10-second clip from it have different loudest moments, so they land on completely different dB scales, even though the underlying audio is identical. Peak-picking thresholds that work on the full movie fail on the clip.

The fix is `ref=1.0`, a fixed reference so the same absolute amplitude always maps to the same dB value no matter what else is in the file. After this one change, match confidence improved dramatically. It's one of those bugs where the code looks fine and the docs don't warn you, but it breaks the whole thing in a subtle way.

**Where:** `audio/fingerprint.py:compute_spectrogram`

### Bug 2: Fixed peak threshold

My first version used a fixed dB cutoff (e.g. `-10 dB`) to decide which spectrogram peaks to keep. This worked fine on clean digital extractions but fell apart on mic recordings, which sit at a much lower absolute loudness. The same cutoff that picks a healthy number of peaks on a clean file leaves almost nothing on a quiet mic recording, so you get basically zero hashes and matching fails.

The fix: compute the threshold as the 95th percentile of *this specific signal's* dB distribution. That way the same fraction of spectrogram bins is always eligible as peak candidates regardless of how loud the input is. As a bonus, volume changes become irrelevant. Multiplying all samples by a constant shifts the whole dB distribution up or down, but the 95th percentile moves with it, so the same peaks get picked.

**Where:** `audio/fingerprint.py:find_peaks`, constant `PEAK_PERCENTILE = 95`

### Stress test results

6 movies with varied audio (Inception, Whiplash, The Matrix, Fight Club, The Grand Budapest Hotel, The Good the Bad and the Ugly) × 3 random 10-second clips × 10 augmentation types = 180 trials.

| Augmentation | Accuracy | Notes |
|---|---|---|
| Clean | 83% (15/18) | 3 failures are content-specific, not systematic |
| Volume −10 dB | 83% (15/18) | Identical to clean, invariant by design |
| Volume −20 dB | 83% (15/18) | Same |
| Noise 20 dB SNR | 83% (15/18) | No degradation |
| Noise 10 dB SNR | 83% (15/18) | Still no degradation |
| Noise 0 dB SNR | 83% (15/18) | Signal power = noise power. I didn't expect this to hold |
| Speed 0.95x | 78% (14/18) | Makes sense, time-stretching shifts the delta_time values in the hashes |
| Speed 1.05x | 67% (12/18) | Worse than 0.95x, not sure why the asymmetry is this large |
| Pitch +1 semitone | 11% (2/18) | Near-total failure, expected |
| Pitch −1 semitone | 0% (0/18) | Total failure, expected |

The 0 dB SNR result was the most surprising to me. At that noise level the audio sounds terrible, but the hash-based system still matched correctly most of the time. My guess is that even under heavy noise the highest-energy peaks in the spectrogram still survive, and those are the ones the peak-picker keeps, so the constellation map stays mostly intact.

The 3 clean failures aren't the same movies every time. They're just clips that happen not to have enough distinctive audio structure to generate a confident match. Not something I can really fix without changing the fundamental approach.

### Why pitch shift completely breaks it

Classical fingerprints encode raw frequency bins. When you shift pitch, every peak moves to a different bin and you get completely different hashes with zero overlap with the original. It's not a threshold or tuning issue, it's structural.

The obvious fix would be encoding frequency *ratios* instead of absolute bins, so pitch shift becomes a constant factor rather than a displacement. I didn't implement this. The neural approach handles pitch shift as a training-time augmentation instead.

---

## Phase 2: Neural Fingerprinting

### Model

Scaled-down version of Google's "Now Playing" architecture:

- **Input:** 1×1025×44 normalized dB spectrogram (1-second window at sr=22050, n_fft=2048)
- **Encoder:** 5-layer conv stack (32→64→128→128→128 channels, stride-2 ConvBlocks), adaptive avg pool to 4×4, two FC layers
- **Output:** 64-dim L2-normalized embedding (~930K parameters)
- **Loss:** Triplet loss, margin=0.3

Triplets: anchor = clean window, positive = same window augmented, negative = different window (different movie or far-away timestamp in the same movie). Augmentation is on-the-fly, no pre-generated files stored to disk.

20 movies total. Each epoch draws 15,000 fresh triplets.

### Round 1

30 epochs, 5,000 samples/epoch, batch 64, lr=1e-3.

Worked on clean audio and digital extractions, but failed on the mic recordings. I wrote `diagnose_failure.py` to figure out what was going wrong.

The diagnosis: the model had learned to identify loud, tonally distinctive content reliably while performing poorly on quieter windows. It's a shortcut: it effectively learned "loud = identifiable" rather than genuinely learning to match content across distortions. The kind of quiet content it struggled with is exactly what you get in a real mic recording of a movie playing in a room.

### Round 2: Hard-example mining

The fix: when sampling training windows, 50% of the time deliberately pick the *quietest* of 4 candidates by RMS energy. This oversamples the hard cases and forces the model to learn them.

Also bumped the training budget: 60 epochs, 15,000 samples/epoch.

This got mic_recording_01 working. mic_recording_02 still fails.

### The competing invariances problem

Round 2 improved real-mic performance, but I noticed it hurt pitch-shift accuracy compared to Round 1. I think this is a genuine tension:

- Hard-example mining pushes toward *loudness* invariance, learning to match the same content at different energy levels
- Pitch-shift invariance requires *frequency-shift* invariance, learning to match the same content at different pitches

These two objectives pull in different directions, and with the current architecture and training budget you can't fully satisfy both at once. So Round 2 is a trade-off: better on real mics, worse on pitch shift. I'd rather document this clearly than pretend it isn't there.

### Mic recording results (Round 2)

Both recordings are of Parasite (2019).

| Recording | RMS energy | Result |
|---|---|---|
| mic_recording_01 | higher | PASS (Parasite ranked #1) |
| mic_recording_02 | lower | FAIL (Parasite not ranked #1) |

For mic_recording_02, `diagnose_failure.py` confirmed the problem is in the embeddings themselves. The model's embeddings for this clip simply aren't close to the right region in Parasite's embedding space. That's a Type A failure (bad embedding), not a Type B failure (bad voting). No amount of tuning the matching step would fix it; the representation for this quieter content is still weak.

### Checkpoint note

Two checkpoint series:
- `fingerprinter_latest.pt` (Round 1, epoch 30)
- `fingerprinter_r2_latest.pt` (Round 2, epoch 60)

Always use `fingerprinter_r2_latest.pt` + `embedding_db_r2.npz` for current results. Round 1 files kept for comparison.

---

## Still open

- I haven't run a systematic stress test on the neural system yet, it's only been tested on two mic recordings. That's the obvious next thing to do.
- mic_recording_02's quieter-content problem isn't solved. Possible next steps: more training rounds, log-mel input instead of linear spectrogram, more model capacity.
- The speed asymmetry (0.95x better than 1.05x in Phase 1) is unexplained and worth looking into.
