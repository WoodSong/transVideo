# transVideo

An automated pipeline for dubbing English videos into Chinese with synchronized audio.

## Overview

6 sequential standalone scripts, each handling one stage:

```
URL → [1-Acquisition] → <id>/<id>.mp4
                             ↓
                    [2-Vocal-Isolation] → <id>_(Vocals).wav + <id>_(Instrumental).wav
                             ↓ (Vocals)
                    [3-asr-diarization] → <id>_transcription.json
                             ↓
                    [4-translation] → <id>_transcription_translated.json
                             ↓
                    [5-tts] → <id>_transcription_translated_audio/seg_N.wav
                             ↓
                    [6-merge] ← (also takes original .mp4 and Instrumental.wav)
                             ↓
                         <id>_final.mp4
```

## Setup

**Dependencies:**

```bash
uv sync
```

**System tools required:** `ffmpeg`, `ffprobe`

**Environment variables** (create a `.env` file):

```
OPENAI_API_KEY=...        # Required for step 4 (translation)
OPENAI_BASE_URL=...       # Optional: custom OpenAI-compatible endpoint
LLM_MODEL=gpt-4.1-mini   # Optional: model override
HF_TOKEN=...              # Required for step 3 (HuggingFace diarization token)
```

## Running the Pipeline

```bash
# Step 1: Download video
uv run python 1-Acquisition.py <url>
uv run python 1-Acquisition.py -a urls.txt --profile Product

# Step 2: Separate vocals from background music
uv run python 2-Vocal-Isolation.py <video_id>/<video_id>.mp4

# Step 3: ASR + speaker diarization
uv run python 3-asr-diarization.py <audio_file> --device mps
# HF_TOKEN is read from .env automatically

# Step 4: Translate segments to Chinese dubbing scripts
uv run python 4-translation.py <transcription.json>

# Step 5: Generate TTS audio per segment
uv run python 5-tts.py <translated.json> --voice zh-CN-XiaoxiaoNeural

# Step 6: Stitch audio and mix with video
uv run python 6-merge.py <translated.json> <video.mp4> <instrumental.wav>
```

## How It Works

**Step 3 (ASR):** WhisperX transcription forced to CPU/int8 for stability. Alignment and diarization use the `--device` flag (`mps` on Apple Silicon, `cuda` on NVIDIA).

**Step 4 (Translation):** Segments are merged by speaker continuity, then sent to an LLM in batches of 10. Each segment produces two fields: `translation` (literal, for QA) and `dubbing` (colloquial, for TTS). Target speech rate: 4–5 Chinese characters/sec.

**Step 5 (TTS):** Uses Microsoft Edge TTS. After generation, actual audio duration is measured and compared to the target. If the atempo ratio falls outside `[0.85, 1.20]`, the LLM rewrites the dubbing and TTS is regenerated (up to 3 retries). After all segments are processed, a neighbor-smoothing pass detects adjacent segments with large ratio differences and rewrites the worse one to reduce jarring speed contrasts.

**Step 6 (Merge):** Per-segment WAVs are overlaid at their original timestamps onto a silent timeline using pydub. Final mix uses FFmpeg sidechain compression to duck background music under vocals.

## Models

The `./models/` directory stores downloaded ML models:
- Whisper models (downloaded by whisperx on first run)
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` — vocal separation model for step 2
