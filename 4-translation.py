import json
import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

SYSTEM_PROMPT = """
# Role
你是一位拥有20年经验的资深视频译制导演和字幕专家。你的专长是将英文视频内容重写为地道的、符合中文口语习惯的配音稿。

# Goal
接收一段英文视频片段（JSON格式），输出对应的中文翻译。

# Key Guidelines (核心准则)
1. **口语化 (Conversational)**: 
   - 严禁“翻译腔”。不要说“这是一个很好的主意”，要说“这主意不错”。
   - 不要直译从句，要把长句拆成短句。
   - 像真人说话一样自然，使用“咱们”、“大家”、“其实”等连接词。

2. **时长对齐 (Duration Constraint) - CRITICAL**:
   - 输入数据中包含 `duration` (秒)。
   - **中文语速标准**: 这里的 TTS 语速约为 **每秒 4-5 个汉字**。
   - 必须根据 `duration` 控制 `dubbing` 的字数：
     - 如果 duration = 1.5s -> 中文控制在 6-8 字左右。
     - 如果 duration = 5.0s -> 中文控制在 20-25 字左右。
   - 如果原文很长 but duration 很短，**必须意译/精简**，保留核心意思即可。
   - 如果原文很短 but duration 很长，适当增加语气词或填充词（如“那么...”、“也就是说...”）来填满时间。

3. **上下文连贯 (Context)**:
   - 结合提供的“前情提提要”确保指代清楚（例如 "It" 指代什么）。

# Output Format
必须返回严格合法的 JSON 格式，不包含 markdown 标记，结构如下：
{
  "results": [
    {
      "id": <对应输入的id>,
      "translation": "<精准直译，用于校对>",
      "dubbing": "<用于配音的口语化重写，严格遵循时长约束>"
    }
  ]
}

# Example
Input:
[{"id": 10, "text": "I mean, it's really complicated if you look at the details.", "duration": 1.8, "speaker": "S1"}]

Correct Output:
{
  "results": [
    {
      "id": 10,
      "translation": "我的意思是，如果你看细节的话，它真的很复杂。",
      "dubbing": "这个细节嘛，其实挺复杂的。"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """
# Context (前情提要 - 上一句话)
{previous_context}

# Input Data (当前待处理片段)
{current_batch_json}

# Instructions
请处理上述 Input Data。确保 `dubbing` 字段的内容读起来大约耗时 `duration` 秒。
直接返回 JSON。
"""

def merge_segments(segments, max_length=150, min_interval=0.5):
    """
    Merges segments based on speaker and interval.
    """
    merged_segments = []
    if not segments:
        return merged_segments

    current_seg = None

    for i, seg in enumerate(segments):
        # Prepare segment data
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker", "Unknown")
        duration = round(end - start, 3)

        new_seg = {
            "id": i,
            "text": text,
            "start": start,
            "end": end,
            "duration": duration,
            "speaker": speaker
        }

        if current_seg is None:
            current_seg = new_seg
            continue

        # Check for merge criteria
        interval = new_seg["start"] - current_seg["end"]
        same_speaker = new_seg["speaker"] == current_seg["speaker"]
        short_interval = interval < min_interval
        can_fit = (len(current_seg["text"]) + len(new_seg["text"])) < max_length

        if same_speaker and short_interval and can_fit:
            # Merge
            current_seg["text"] += " " + new_seg["text"]
            current_seg["end"] = new_seg["end"]
            current_seg["duration"] = round(current_seg["end"] - current_seg["start"], 3)
        else:
            # Save current and start new
            merged_segments.append(current_seg)
            current_seg = new_seg

    if current_seg:
        merged_segments.append(current_seg)

    return merged_segments

def translate_batch(client, model, batch, previous_context):
    """
    Sends a batch of segments to the LLM for translation.
    """
    input_json = json.dumps(batch, ensure_ascii=False)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        previous_context=previous_context,
        current_batch_json=input_json
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result_data = json.loads(content)
        return result_data.get("results", [])
    except Exception as e:
        print(f"Error during translation: {e}")
        return []

def refine_segment(client, model, seg):
    """
    Refines the dubbing text if it's too long or too short.
    """
    duration = seg.get("duration", 0)
    max_chars = int(duration * 5.0)
    min_chars = int(duration * 3.5)
    
    dubbing = seg.get("dubbing", "").strip()
    translation = seg.get("translation", "").strip()
    
    if not dubbing:
        return

    prompt = ""
    if len(dubbing) > max_chars:
        prompt = f"原意：{translation}。太长了，请必须缩减到 {max_chars} 字以内，保留核心意思即可。"
    elif len(dubbing) < min_chars:
        prompt = f"原意：{translation}。太短了，请必须增加到 {min_chars} 字以上，保留核心意思即可。"
        
    if prompt:
        print(f"  Refining segment {seg['id']} (len: {len(dubbing)}, target: {min_chars}-{max_chars})...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一位专业的配音导演。请根据指令直接输出优化后的配音口语稿（dubbing），不需要 JSON 格式，不需要任何开场白。"},
                    {"role": "user", "content": prompt}
                ]
            )
            refined_dubbing = response.choices[0].message.content.strip()
            # Clean up potential LLM artifacts
            refined_dubbing = refined_dubbing.replace("`", "").strip('"').strip("'")
            seg["dubbing"] = refined_dubbing
            print(f"  Fixed: {len(dubbing)} -> {len(refined_dubbing)} characters.")
        except Exception as e:
            print(f"  Error refining segment {seg['id']}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge and translate transcription segments.")
    parser.add_argument("input_file", help="Path to the transcription JSON file.")
    parser.add_argument("--output", help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of segments per LLM call.")
    parser.add_argument("--max_length", type=int, default=150, help="Max characters for merging.")
    parser.add_argument("--interval", type=float, default=0.5, help="Min interval for merging.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get("segments", [])
    print(f"Loaded {len(segments)} segments from {args.input_file}")

    merged = merge_segments(segments, max_length=args.max_length, min_interval=args.interval)
    print(f"Merged into {len(merged)} segments")

    # LLM Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    if not api_key:
        print("Warning: OPENAI_API_KEY not found in .env. Translation will be skipped.")
        # Just save the merged segments without translation for now
        output_data = {"segments": merged}
    else:

        client = OpenAI(api_key=api_key, base_url=base_url)
        final_results = []
        previous_context = "No previous context."

        for i in range(0, len(merged), args.batch_size):
            batch = merged[i : i + args.batch_size]
            print(f"Translating batch {i//args.batch_size + 1}/{(len(merged)-1)//args.batch_size + 1}...")
            
            translated_batch = translate_batch(client, model, batch, previous_context)
            
            # Map translations back to segments
            id_to_translated = {item["id"]: item for item in translated_batch}
            
            for seg in batch:
                translated_item = id_to_translated.get(seg["id"])
                if translated_item:
                    seg["translation"] = translated_item.get("translation", "")
                    seg["dubbing"] = translated_item.get("dubbing", "")
                    # Refine if needed based on character count
                    refine_segment(client, model, seg)
                else:
                    seg["translation"] = ""
                    seg["dubbing"] = ""
                final_results.append(seg)
                
            # Update context with the last segment's text for the next batch
            if batch:
                previous_context = batch[-1]["text"]

        output_data = {"segments": final_results}

    output_file = args.output
    if not output_file:
        base, _ = os.path.splitext(args.input_file)
        output_file = f"{base}_translated.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
