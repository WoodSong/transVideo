# TTS Closed-Loop Duration Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `5-tts.py` so that after generating TTS audio, if the required `atempo` ratio is outside [0.75, 1.5], the LLM rewrites the dubbing text and TTS is regenerated — up to 3 retries — before falling back to forced alignment with a warning flag.

**Architecture:** Add a `rewrite_dubbing()` function that calls the OpenAI API with the current dubbing + translation and a character-adjustment instruction. The `process_segments()` loop becomes an iterative retry loop that tracks all attempts and selects the best one on failure. LLM client is initialized lazily from `.env`.

**Tech Stack:** Python 3.13, `edge-tts`, `ffmpeg`/`ffprobe`, `openai` SDK, `python-dotenv`

---

## File Structure

- **Modify:** `5-tts.py` — all changes live here
- **No other files change**

---

### Task 1: Add constants and LLM imports

**Files:**
- Modify: `5-tts.py:1-8`

- [ ] **Step 1: Add imports and constants at the top of the file**

Replace the existing imports block (lines 1–8) with:

```python
import asyncio
import json
import os
import argparse
import subprocess
import re
import edge_tts
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ATEMPO_MIN = 0.75
ATEMPO_MAX = 1.50
MAX_RETRIES = 3
```

- [ ] **Step 2: Verify the file still runs without error**

```bash
cd /Users/i062843/development/transVideo
uv run python 5-tts.py --help
```

Expected output: argparse help text, no import errors.

- [ ] **Step 3: Commit**

```bash
git add 5-tts.py
git commit -m "feat: add LLM imports and atempo constants to 5-tts.py"
```

---

### Task 2: Add `rewrite_dubbing` function

**Files:**
- Modify: `5-tts.py` — insert after `generate_tts` function (after line 65)

- [ ] **Step 1: Insert `rewrite_dubbing` function after `generate_tts`**

Add this function between `generate_tts` and `process_segments`:

```python
def rewrite_dubbing(client, seg, actual_duration, target_duration):
    """
    Ask the LLM to rewrite dubbing text to better fit the target duration.
    Returns new dubbing string, or original if LLM call fails.
    """
    current_dubbing = seg.get("dubbing", "").strip()
    translation = seg.get("translation", current_dubbing)

    # Estimate current TTS speed (chars/sec) from actual measurement
    char_count = len(re.findall(r'[\u4e00-\u9fff0-9]', current_dubbing))
    if char_count == 0 or actual_duration == 0:
        return current_dubbing

    chars_per_sec = char_count / actual_duration
    target_chars = int(target_duration * chars_per_sec)
    current_chars = len(current_dubbing)

    if current_chars > target_chars:
        instruction = f"太长了。请缩减到约 {target_chars} 字以内，保留核心意思。"
    else:
        instruction = f"太短了。请扩展到约 {target_chars} 字以上，保留核心意思，可以增加自然的口语填充词。"

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "你是专业配音导演。直接输出优化后的配音口语稿，不需要任何开场白或格式。"
                },
                {
                    "role": "user",
                    "content": f"原意：{translation}\n当前配音稿：{current_dubbing}\n{instruction}"
                }
            ]
        )
        new_dubbing = response.choices[0].message.content.strip()
        new_dubbing = new_dubbing.replace("`", "").strip('"').strip("'")
        return new_dubbing
    except Exception as e:
        print(f"  LLM rewrite failed: {e}")
        return current_dubbing
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/i062843/development/transVideo
uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('tts', '5-tts.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add 5-tts.py
git commit -m "feat: add rewrite_dubbing function to 5-tts.py"
```

---

### Task 3: Rewrite `process_segments` with iterative retry loop

**Files:**
- Modify: `5-tts.py` — replace `process_segments` function (lines 67–127)

- [ ] **Step 1: Replace `process_segments` with the new iterative version**

Replace the entire `process_segments` function with:

```python
async def process_segments(data, voice, output_dir, limit=None, offset=0):
    segments = data.get("segments", [])
    if offset > 0:
        segments = segments[offset:]
    if limit:
        segments = segments[:limit]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lazy-initialize LLM client only if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    llm_client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

    for i, seg in enumerate(segments):
        seg_id = seg.get("id", i)
        text = seg.get("dubbing", seg.get("translation", ""))
        if not text:
            print(f"Skipping segment {seg_id}: No text found.")
            continue

        t_target = seg.get("duration", 0)
        if t_target <= 0:
            print(f"Skipping segment {seg_id}: Zero duration.")
            continue

        final_path = os.path.join(output_dir, f"seg_{seg_id}.wav")
        current_dubbing = text

        # Track all attempts: list of (ratio, temp_path, dubbing_used)
        attempts = []

        for attempt in range(MAX_RETRIES + 1):
            temp_path = os.path.join(output_dir, f"seg_{seg_id}_temp_{attempt}.mp3")

            # Determine TTS rate: pre-estimate to nudge speed
            char_count = len(re.findall(r'[\u4e00-\u9fff0-9]', current_dubbing))
            eng_word_count = len(re.findall(r'[a-zA-Z]+', current_dubbing))
            total_estimate = char_count + (eng_word_count * 0.5)
            t_base = total_estimate / 4.0
            r_pre = t_base / t_target if t_target > 0 else 1.0
            rate_param = "+20%" if r_pre > 1.05 else "+0%"

            print(f"  Attempt {attempt + 1}/{MAX_RETRIES + 1} for seg {seg_id}: '{current_dubbing[:20]}...'")
            await generate_tts(current_dubbing, voice, rate_param, temp_path)
            t_actual = get_audio_duration(temp_path)

            if t_actual == 0:
                print(f"  Warning: Could not measure duration for attempt {attempt + 1}. Skipping.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                break

            ratio = t_actual / t_target
            print(f"  Actual: {t_actual:.2f}s, Target: {t_target:.2f}s, Ratio: {ratio:.2f}")
            attempts.append((ratio, temp_path, current_dubbing))

            if ATEMPO_MIN <= ratio <= ATEMPO_MAX:
                # Good ratio — apply atempo and finish
                success = apply_atempo(temp_path, final_path, t_target)
                if success:
                    t_final = get_audio_duration(final_path)
                    print(f"  Aligned duration: {t_final:.2f}s")
                else:
                    print(f"  Failed to align segment {seg_id}")
                os.remove(temp_path)
                break

            # Ratio out of range
            if attempt < MAX_RETRIES and llm_client:
                print(f"  Ratio {ratio:.2f} out of range [{ATEMPO_MIN}, {ATEMPO_MAX}]. Rewriting dubbing...")
                seg["dubbing"] = current_dubbing  # ensure seg has current value for rewrite_dubbing
                new_dubbing = rewrite_dubbing(llm_client, seg, t_actual, t_target)
                print(f"  Rewritten: '{new_dubbing[:40]}...' ({len(new_dubbing)} chars)")
                current_dubbing = new_dubbing
                os.remove(temp_path)
            else:
                # No LLM or retries exhausted — pick best attempt
                best_ratio, best_temp, best_dubbing = min(attempts, key=lambda x: abs(x[0] - 1.0))
                print(f"  Retries exhausted. Using best attempt (ratio={best_ratio:.2f}). Marking atempo_warning.")
                seg["atempo_warning"] = True
                seg["atempo_retries"] = attempt + 1
                seg["dubbing"] = best_dubbing
                # Clean up all other temp files
                for r, tp, _ in attempts:
                    if tp != best_temp and os.path.exists(tp):
                        os.remove(tp)
                success = apply_atempo(best_temp, final_path, t_target)
                if success:
                    t_final = get_audio_duration(final_path)
                    print(f"  Forced aligned duration: {t_final:.2f}s")
                else:
                    print(f"  Failed to align segment {seg_id}")
                if os.path.exists(best_temp):
                    os.remove(best_temp)
                break
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/i062843/development/transVideo
uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('tts', '5-tts.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run with `--limit 1` on an existing translated JSON to confirm it runs end-to-end**

```bash
cd /Users/i062843/development/transVideo
# Replace <path_to_translated.json> with an actual file from a prior pipeline run
uv run python 5-tts.py <path_to_translated.json> --limit 1 --output_dir /tmp/tts_test
```

Expected: Output shows `Attempt 1/4`, actual duration, and either "Aligned duration" or "Rewriting dubbing". No Python exceptions.

- [ ] **Step 4: Commit**

```bash
git add 5-tts.py
git commit -m "feat: replace process_segments with iterative closed-loop atempo alignment"
```

---

### Task 4: Persist updated `dubbing` field back to JSON output

**Context:** The current script reads a JSON file but never writes back updated `dubbing` values. When `rewrite_dubbing` rewrites text, the final `seg["dubbing"]` should be saved so downstream review is possible.

**Files:**
- Modify: `5-tts.py` — update `main()` to save updated segments

- [ ] **Step 1: Update `main()` to write back the JSON after processing**

Replace the last 3 lines of `main()`:

```python
    await process_segments(data, args.voice, output_dir, args.limit, args.offset)
    print(f"Finished processing all segments. Audio saved in {output_dir}")
```

With:

```python
    await process_segments(data, args.voice, output_dir, args.limit, args.offset)
    print(f"Finished processing all segments. Audio saved in {output_dir}")

    # Write back updated dubbing/warning fields to the JSON
    with open(args.input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Updated segment data written back to {args.input_file}")
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/i062843/development/transVideo
uv run python 5-tts.py --help
```

Expected: help text, no errors.

- [ ] **Step 3: Commit**

```bash
git add 5-tts.py
git commit -m "feat: write back updated dubbing and atempo_warning fields to input JSON"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `ATEMPO_MIN=0.75`, `ATEMPO_MAX=1.50`, `MAX_RETRIES=3` | Task 1 |
| `rewrite_dubbing` function with actual TTS speed calibration | Task 2 |
| Iterative retry loop tracking all attempts | Task 3 |
| Best-attempt fallback with `atempo_warning=True` and `atempo_retries` | Task 3 |
| Lazy LLM client init, fallback if no API key | Task 3 |
| Updated `dubbing` written back to JSON | Task 4 |

**Placeholder scan:** None found.

**Type consistency:** `rewrite_dubbing` returns `str`. `seg["dubbing"]` is `str` throughout. `attempts` is `list[tuple[float, str, str]]`. `best_ratio, best_temp, best_dubbing` destructures correctly.
