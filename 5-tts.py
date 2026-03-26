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

async def process_segments(data, voice, output_dir, limit=None, offset=0):
    segments = data.get("segments", [])
    if offset > 0:
        segments = segments[offset:]
    if limit:
        segments = segments[:limit]
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
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
            
        # 1.2 Estimate base duration (4 chars/s)
        # Count Chinese characters and digits
        char_count = len(re.findall(r'[\u4e00-\u9fff0-9]', text))
        # Add some buffer for English words if any
        eng_word_count = len(re.findall(r'[a-zA-Z]+', text))
        total_estimate_count = char_count + (eng_word_count * 0.5) # English words are usually faster
        
        t_base = total_estimate_count / 4.0
        
        # 1.3 Compute required speed-up ratio
        r_need = t_base / t_target
        
        # 1.4 Inject TTS parameters
        rate_param = "+0%"
        if r_need > 1.05:
            rate_param = "+20%"
            
        print(f"Processing segment {seg_id}: '{text[:20]}...'")
        print(f"  Target: {t_target}s, Base: {t_base:.2f}s, Ratio: {r_need:.2f}, TTS Rate: {rate_param}")
        
        temp_path = os.path.join(output_dir, f"seg_{seg_id}_temp.mp3")
        final_path = os.path.join(output_dir, f"seg_{seg_id}.wav")
        
        # 1.5 Generate & validate
        await generate_tts(text, voice, rate_param, temp_path)
        t_actual = get_audio_duration(temp_path)
        print(f"  Actual TTS duration: {t_actual:.2f}s")
        
        # 1.6 Final alignment (FFmpeg)
        success = apply_atempo(temp_path, final_path, t_target)
        if success:
            t_final = get_audio_duration(final_path)
            print(f"  Final aligned duration: {t_final:.2f}s")
        else:
            print(f"  Failed to align segment {seg_id}")
            
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

if __name__ == "__main__":
    asyncio.run(main())
