# Design: TTS Closed-Loop Duration Alignment

**Date:** 2026-03-26
**Scope:** Step 5 (`5-tts.py`) only
**Problem:** English-to-Chinese dubbing segments frequently have large duration mismatches, causing `atempo` to stretch/compress audio beyond audible quality limits (too fast or too slow speech in the final video).

---

## Root Cause

The current pipeline estimates Chinese character count to predict TTS duration, then applies `atempo` to force-fit the result. This breaks down because:

1. Character count → TTS duration estimation has ~20–30% error
2. LLM dubbing length in Step 4 is not calibrated against real TTS output
3. `atempo` ratios beyond ~1.5x or below ~0.75x produce audible distortion

---

## Solution: Iterative Closed-Loop in Step 5

After generating TTS, **measure actual audio duration**, compute the required `atempo` ratio, and if it falls outside the safe range, **call the LLM to rewrite the dubbing text** and regenerate TTS. Repeat up to 3 times.

### Flow

```
generate_tts(dubbing)
    → measure actual_duration
    → ratio = actual_duration / target_duration
    → if ratio in [0.75, 1.50]: apply atempo → done
    → else: rewrite_dubbing(LLM) → regenerate TTS → repeat
    → after 3 retries: pick best attempt, mark atempo_warning=true, force atempo
```

---

## Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `ATEMPO_MIN` | 0.75 | Below this, speech sounds unnaturally slow |
| `ATEMPO_MAX` | 1.50 | Above this, speech sounds rushed/distorted |
| `MAX_RETRIES` | 3 | Balances quality vs. LLM cost |

---

## New Function: `rewrite_dubbing`

Added to `5-tts.py`. Accepts:
- `client`: OpenAI client (initialized from `.env` same as Step 4)
- `seg`: the current segment dict (contains `dubbing`, `translation`, `duration`)
- `actual_duration`: measured TTS duration in seconds
- `target_duration`: original segment duration in seconds

Behavior:
- Calculates how many characters too long/short the dubbing is, based on actual TTS speed
- Sends a concise prompt to the LLM: provide current dubbing, translation (for meaning anchor), and the character adjustment needed
- Returns new dubbing string (plain text, not JSON)
- System prompt: "你是专业配音导演。直接输出优化后的配音口语稿，不需要任何开场白或格式。"

---

## Retry Logic

Each iteration:
1. Generate TTS to `seg_{id}_temp.mp3`
2. Measure `actual_duration` via `ffprobe`
3. Compute `ratio = actual_duration / target_duration`
4. If `ATEMPO_MIN <= ratio <= ATEMPO_MAX`: proceed to `atempo` alignment
5. Else: call `rewrite_dubbing`, update `seg["dubbing"]`, go to step 1
6. Track all (attempt, ratio, temp_path) tuples

After `MAX_RETRIES` exhausted without success:
- Select the attempt with ratio closest to 1.0
- Apply `atempo` with that attempt's audio
- Set `seg["atempo_warning"] = True` in the output JSON

---

## LLM Integration in Step 5

Step 5 currently has no LLM dependency. This change adds:
- `from openai import OpenAI` and `from dotenv import load_dotenv`
- Client initialization at startup (only if any segment needs rewriting — lazy init)
- Uses same env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`
- If `OPENAI_API_KEY` is not set, skip rewriting and always force-atempo (fallback to current behavior)

---

## Output JSON Changes

Segments that required retries will have additional fields:
```json
{
  "id": 42,
  "dubbing": "...(final version used for TTS)...",
  "atempo_warning": true,
  "atempo_retries": 3
}
```

Segments that aligned cleanly on first try: no new fields added.

---

## What Is NOT Changed

- Step 4 (`4-translation.py`): no changes. `refine_segment` still runs as a first-pass character-count filter.
- Step 6 (`6-merge.py`): no changes. Uses `start` timestamps from JSON, unaffected.
- The `atempo` implementation in Step 5: no changes to chaining logic.

---

## Trade-offs

| Concern | Mitigation |
|---------|------------|
| Extra LLM API calls | Only triggered when ratio is out of range; most segments won't need it |
| Longer processing time | Retries are per-segment and sequential; tolerable for offline pipeline |
| LLM may not fix in 3 tries | Best-attempt fallback + `atempo_warning` flag for human review |
