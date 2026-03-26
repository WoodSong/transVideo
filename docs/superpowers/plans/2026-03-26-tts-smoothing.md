# TTS Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve perceptual smoothness of dubbed audio by tightening the per-segment atempo acceptance window and adding a neighbor-smoothing pass that rewrites segments whose ratios differ too much from adjacent segments.

**Architecture:** Two changes to `5-tts.py`. First: tighten `ATEMPO_MIN/MAX` constants so the LLM rewrite loop fires sooner, reducing extreme stretching. Second: after all segments are processed, a synchronous neighbor-smoothing pass iterates over the recorded ratios, identifies pairs where the ratio difference exceeds `NEIGHBOR_RATIO_DIFF`, rewrites the worse offender via `rewrite_dubbing`, regenerates its TTS with the same closed-loop retry logic, and records the new ratio. The smoothing pass runs only once (no recursive re-smoothing). Ratios are persisted in the segment dict under `"ratio"` so the pass can read them without re-measuring audio files.

**Tech Stack:** Python, edge-tts, FFmpeg atempo, OpenAI-compatible LLM API (same as existing code)

---

### Task 1: Tighten atempo thresholds and add NEIGHBOR_RATIO_DIFF constant

**Files:**
- Modify: `5-tts.py:14-16`

- [ ] **Step 1: Change the three constants**

Replace lines 14-16 in `5-tts.py`:

```python
ATEMPO_MIN = 0.85
ATEMPO_MAX = 1.20
MAX_RETRIES = 3
NEIGHBOR_RATIO_DIFF = 0.30
```

- [ ] **Step 2: Verify the file looks right**

Run:
```bash
head -20 5-tts.py
```

Expected output includes:
```
ATEMPO_MIN = 0.85
ATEMPO_MAX = 1.20
MAX_RETRIES = 3
NEIGHBOR_RATIO_DIFF = 0.30
```

- [ ] **Step 3: Commit**

```bash
git add 5-tts.py
git commit -m "feat: tighten atempo thresholds [0.85,1.20] and add NEIGHBOR_RATIO_DIFF=0.30"
```

---

### Task 2: Record ratio on each segment after processing

**Files:**
- Modify: `5-tts.py` — inside `process_segments`, after a segment finishes

After `process_segments` produces a final WAV for a segment, we need to record the ratio that was used so the smoothing pass can read it. The ratio to record is the one from the accepted attempt (or best attempt on fallback).

- [ ] **Step 1: Record ratio in the success branch**

In `process_segments`, find the block that handles a good ratio (around line 180-189):

```python
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
```

Add `seg["ratio"] = ratio` before the `break`:

```python
            if ATEMPO_MIN <= ratio <= ATEMPO_MAX:
                # Good ratio — apply atempo and finish
                success = apply_atempo(temp_path, final_path, t_target)
                if success:
                    t_final = get_audio_duration(final_path)
                    print(f"  Aligned duration: {t_final:.2f}s")
                else:
                    print(f"  Failed to align segment {seg_id}")
                os.remove(temp_path)
                seg["ratio"] = ratio
                break
```

- [ ] **Step 2: Record ratio in the fallback branch**

Find the fallback block (around line 200-218):

```python
                best_ratio, best_temp, best_dubbing = min(attempts, key=lambda x: abs(x[0] - 1.0))
                print(f"  Retries exhausted. Using best attempt (ratio={best_ratio:.2f}). Marking atempo_warning.")
                seg["atempo_warning"] = True
                seg["atempo_retries"] = attempt + 1
                seg["dubbing"] = best_dubbing
```

Add `seg["ratio"] = best_ratio` after `seg["dubbing"] = best_dubbing`:

```python
                best_ratio, best_temp, best_dubbing = min(attempts, key=lambda x: abs(x[0] - 1.0))
                print(f"  Retries exhausted. Using best attempt (ratio={best_ratio:.2f}). Marking atempo_warning.")
                seg["atempo_warning"] = True
                seg["atempo_retries"] = attempt + 1
                seg["dubbing"] = best_dubbing
                seg["ratio"] = best_ratio
```

- [ ] **Step 3: Commit**

```bash
git add 5-tts.py
git commit -m "feat: record atempo ratio per segment for neighbor smoothing"
```

---

### Task 3: Add smooth_neighbors function

**Files:**
- Modify: `5-tts.py` — add new async function after `process_segments`

The function takes the full `segments` list (post-processing), `llm_client`, `voice`, and `output_dir`. It iterates over consecutive pairs, finds pairs where `abs(ratio_a - ratio_b) > NEIGHBOR_RATIO_DIFF`, picks the segment farther from 1.0, rewrites its dubbing via `rewrite_dubbing`, regenerates TTS with the same closed-loop retry logic as `process_segments`, and updates `seg["ratio"]` with the new value.

The smoothing pass runs once — it does not recurse or re-check already-fixed pairs.

- [ ] **Step 1: Add the function**

Insert the following function after the closing of `process_segments` (before `async def main()`):

```python
async def smooth_neighbors(segments, llm_client, voice, output_dir):
    """
    One-pass neighbor smoothing: if two adjacent segments have ratio difference
    > NEIGHBOR_RATIO_DIFF, rewrite the one farther from 1.0 and regenerate its TTS.
    Only runs when an LLM client is available.
    """
    if not llm_client:
        return

    # Build list of (index, seg) for segments that have a recorded ratio
    indexed = [(i, seg) for i, seg in enumerate(segments) if "ratio" in seg]

    for k in range(len(indexed) - 1):
        idx_a, seg_a = indexed[k]
        idx_b, seg_b = indexed[k + 1]

        ratio_a = seg_a["ratio"]
        ratio_b = seg_b["ratio"]

        if abs(ratio_a - ratio_b) <= NEIGHBOR_RATIO_DIFF:
            continue

        # Pick the segment farther from 1.0
        if abs(ratio_a - 1.0) >= abs(ratio_b - 1.0):
            target_seg = seg_a
        else:
            target_seg = seg_b

        seg_id = target_seg.get("id", "?")
        t_target = target_seg.get("duration", 0)
        if t_target <= 0:
            continue

        print(f"  [smooth] seg {seg_id}: neighbor ratio diff "
              f"{ratio_a:.2f} vs {ratio_b:.2f}. Rewriting seg {seg_id} (ratio={target_seg['ratio']:.2f})...")

        final_path = os.path.join(output_dir, f"seg_{seg_id}.wav")
        current_dubbing = target_seg.get("dubbing", "")
        attempts = []

        for attempt in range(MAX_RETRIES + 1):
            temp_path = os.path.join(output_dir, f"seg_{seg_id}_smooth_{attempt}.mp3")

            char_count = len(re.findall(r'[\u4e00-\u9fff0-9]', current_dubbing))
            eng_word_count = len(re.findall(r'[a-zA-Z]+', current_dubbing))
            total_estimate = char_count + (eng_word_count * 0.5)
            t_base = total_estimate / 4.0
            r_pre = t_base / t_target if t_target > 0 else 1.0
            rate_param = "+20%" if r_pre > 1.05 else "+0%"

            print(f"    Attempt {attempt + 1}/{MAX_RETRIES + 1}: '{current_dubbing[:20]}...'")
            await generate_tts(current_dubbing, voice, rate_param, temp_path)
            t_actual = get_audio_duration(temp_path)

            if t_actual <= 0:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                for _, tp, _ in attempts:
                    if os.path.exists(tp):
                        os.remove(tp)
                break

            ratio = t_actual / t_target
            print(f"    Actual: {t_actual:.2f}s, Target: {t_target:.2f}s, Ratio: {ratio:.2f}")
            attempts.append((ratio, temp_path, current_dubbing))

            if ATEMPO_MIN <= ratio <= ATEMPO_MAX:
                success = apply_atempo(temp_path, final_path, t_target)
                if success:
                    t_final = get_audio_duration(final_path)
                    print(f"    Smoothed aligned duration: {t_final:.2f}s")
                os.remove(temp_path)
                target_seg["dubbing"] = current_dubbing
                target_seg["ratio"] = ratio
                # Update indexed list so subsequent pairs use the new ratio
                indexed[k if target_seg is seg_a else k + 1] = (
                    idx_a if target_seg is seg_a else idx_b, target_seg
                )
                break

            if attempt < MAX_RETRIES:
                target_seg["dubbing"] = current_dubbing
                new_dubbing = rewrite_dubbing(llm_client, target_seg, t_actual, t_target)
                print(f"    Rewritten: '{new_dubbing[:40]}...' ({len(new_dubbing)} chars)")
                current_dubbing = new_dubbing
            else:
                best_ratio, best_temp, best_dubbing = min(attempts, key=lambda x: abs(x[0] - 1.0))
                for r, tp, _ in attempts:
                    if tp != best_temp and os.path.exists(tp):
                        os.remove(tp)
                apply_atempo(best_temp, final_path, t_target)
                if os.path.exists(best_temp):
                    os.remove(best_temp)
                target_seg["dubbing"] = best_dubbing
                target_seg["ratio"] = best_ratio
                print(f"    Smooth retries exhausted. Best ratio={best_ratio:.2f}")
```

- [ ] **Step 2: Verify function is syntactically valid**

```bash
python3 -c "import ast; ast.parse(open('5-tts.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add 5-tts.py
git commit -m "feat: add smooth_neighbors one-pass neighbor ratio smoothing"
```

---

### Task 4: Call smooth_neighbors from main and skip smoothing when --limit/--offset is used

**Files:**
- Modify: `5-tts.py` — `main()` function, after `await process_segments(...)`

The smoothing pass only makes sense when the full segment list was processed. When `--limit` or `--offset` is active (partial run), skip it to avoid false positives from missing neighbors.

- [ ] **Step 1: Update main() to call smooth_neighbors**

Find in `main()`:

```python
    await process_segments(data, args.voice, output_dir, args.limit, args.offset)
    print(f"Finished processing all segments. Audio saved in {output_dir}")
```

Replace with:

```python
    await process_segments(data, args.voice, output_dir, args.limit, args.offset)
    print(f"Finished processing all segments. Audio saved in {output_dir}")

    # Neighbor smoothing — only when processing the full segment list
    partial_run = args.limit is not None or args.offset > 0
    if not partial_run:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        llm_client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        segments = data.get("segments", [])
        print(f"Running neighbor smoothing pass over {len(segments)} segments...")
        await smooth_neighbors(segments, llm_client, args.voice, output_dir)
        print("Neighbor smoothing complete.")
    else:
        print("Skipping neighbor smoothing (partial run).")
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('5-tts.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Smoke test with --limit to confirm smoothing is skipped**

```bash
uv run python 5-tts.py --help 2>&1 | head -5
```

Expected: no import errors, help text printed.

- [ ] **Step 4: Commit**

```bash
git add 5-tts.py
git commit -m "feat: call smooth_neighbors after full processing run in main()"
```
