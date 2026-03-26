# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

transVideo is a video dubbing pipeline that translates English videos into Chinese with synchronized audio. It consists of 6 sequential standalone scripts, each handling one stage of the pipeline.

## Environment Setup

```bash
# Install dependencies (uses uv)
uv sync

# Required system tools
# ffmpeg and ffprobe must be installed (used by steps 2, 5, 6)

# Environment variables (set in .env)
OPENAI_API_KEY=...       # Required for step 4 (translation)
OPENAI_BASE_URL=...      # Optional: custom OpenAI-compatible endpoint
LLM_MODEL=gpt-4.1-mini  # Optional: model override
```

## Running the Pipeline

Each script is run independently in order. Output from one step is input to the next.

```bash
# Step 1: Download video (outputs to <video_id>/<video_id>.mp4)
uv run python 1-Acquisition.py <url>
uv run python 1-Acquisition.py -a urls.txt --profile Product

# Step 2: Separate vocals/instruments (outputs *_(Vocals).wav and *_(Instrumental).wav)
uv run python 2-Vocal-Isolation.py <video_id>/<video_id>.mp4

# Step 3: ASR + speaker diarization (outputs <audio>_transcription.json)
uv run python 3-asr-diarization.py <audio_file> --device mps
# Requires HuggingFace token for diarization (--token flag)

# Step 4: Translate English segments to Chinese dubbing scripts (outputs *_translated.json)
uv run python 4-translation.py <transcription.json>

# Step 5: Generate TTS audio per segment (outputs seg_<id>.wav files in *_audio/ dir)
uv run python 5-tts.py <translated.json> --voice zh-CN-XiaoxiaoNeural

# Step 6: Stitch audio and mix with video (outputs *_final.mp4)
uv run python 6-merge.py <translated.json> <video.mp4> <instrumental.wav>
```

## Pipeline Data Flow

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

## Key Architecture Decisions

**Step 3 (ASR):** ASR (Whisper) is forced to CPU/int8 for compatibility. Alignment and diarization use the device flag (mps on Apple Silicon, cuda on NVIDIA). The script patches `torch.load` to bypass PyTorch 2.6+ security defaults for model loading.

**Step 4 (Translation):** Segments are merged by speaker continuity before being sent to the LLM in batches of 10. Each segment gets two outputs: `translation` (literal, for QA) and `dubbing` (colloquial, for TTS). A refinement pass adjusts `dubbing` length based on `duration` (target: 3-4 chars/sec).

**Step 5 (TTS):** Uses Microsoft Edge TTS (`edge-tts`). Audio is time-stretched with FFmpeg's `atempo` filter to match the original segment's timestamp duration. `atempo` is limited to 0.5–2.0x, so the filter is chained when needed.

**Step 6 (Merge):** Uses pydub to overlay per-segment WAVs at their original timestamps onto a silent timeline. Final mix applies FFmpeg sidechain compression so background music ducks under vocals.

## Models Directory

The `./models/` directory stores downloaded ML models:
- Whisper models (downloaded by whisperx on first run)
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` — default vocal separation model for step 2
