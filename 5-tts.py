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

ATEMPO_MIN = 0.85
ATEMPO_MAX = 1.20
MAX_RETRIES = 3
NEIGHBOR_RATIO_DIFF = 0.30

def get_audio_duration(file_path):
    """
    Get the duration of an audio file using ffprobe.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0.0

def apply_atempo(input_path, output_path, target_duration):
    """
    Apply atempo filter to match the target duration exactly.
    """
    actual_duration = get_audio_duration(input_path)
    if actual_duration == 0:
        return False
    
    tempo = actual_duration / target_duration
    
    # atempo filter limit is 0.5 to 2.0. We can chain them if needed.
    filters = []
    t = tempo
    while t > 2.0:
        filters.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        filters.append("atempo=0.5")
        t /= 0.5
    filters.append(f"atempo={t}")
    
    filter_str = ",".join(filters)
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:a", filter_str,
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Error applying atempo: {e}")
        return False

async def generate_tts(text, voice, rate_str, output_path):
    """
    Generate TTS using edge-tts.
    """
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(output_path)

def rewrite_dubbing(client, seg, actual_duration, target_duration):
    """
    Ask the LLM to rewrite dubbing text to better fit the target duration.
    Returns new dubbing string, or original if LLM call fails.
    """
    current_dubbing = seg.get("dubbing", "").strip()
    translation = seg.get("translation", current_dubbing)

    # Estimate current TTS speed (chars/sec) from actual measurement
    char_count = len(re.findall(r'[\u4e00-\u9fff0-9]', current_dubbing))
    if char_count == 0 or actual_duration <= 0:
        return current_dubbing

    chars_per_sec = char_count / actual_duration
    target_chars = int(target_duration * chars_per_sec)
    current_chars = char_count

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
        if not new_dubbing: return current_dubbing
        new_dubbing = new_dubbing.replace("`", "").strip('"').strip("'")
        return new_dubbing
    except Exception as e:
        print(f"  LLM rewrite failed: {e}")
        return current_dubbing

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

            if t_actual <= 0:
                print(f"  Warning: Could not measure duration for attempt {attempt + 1}. Skipping.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                for _, tp, _ in attempts:
                    if os.path.exists(tp):
                        os.remove(tp)
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
                # Keep temp_path on disk — fallback cleanup will handle it
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

async def main():
    parser = argparse.ArgumentParser(description="Generate TTS for translated segments.")
    parser.add_argument("input_file", help="Path to the translated JSON file.")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="Edge-TTS voice to use.")
    parser.add_argument("--output_dir", help="Directory to save generated audio segments.")
    parser.add_argument("--limit", type=int, help="Limit the number of segments to process.")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start processing segments from.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        return
        
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    output_dir = args.output_dir
    if not output_dir:
        base, _ = os.path.splitext(args.input_file)
        output_dir = f"{base}_audio"
        
    await process_segments(data, args.voice, output_dir, args.limit, args.offset)
    print(f"Finished processing all segments. Audio saved in {output_dir}")

    # Write back updated dubbing/warning fields to the JSON
    with open(args.input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Updated segment data written back to {args.input_file}")

if __name__ == "__main__":
    asyncio.run(main())
