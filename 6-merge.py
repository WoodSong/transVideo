import json
import os
import argparse
import subprocess
from pydub import AudioSegment

def get_audio_duration(file_path):
    """
    Get the duration of an audio/video file using ffprobe.
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

def stitch_audio(json_file, audio_dir, total_duration, output_wav):
    """
    Stitch small wav files into a single full_vocals.wav based on timeline.
    """
    print(f"Stitching audio chunks from {audio_dir}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    # Create silent background of the full duration
    full_vocals = AudioSegment.silent(duration=int(total_duration * 1000))
    
    for seg in segments:
        seg_id = seg.get("id")
        start_time = seg.get("start")
        audio_path = os.path.join(audio_dir, f"seg_{seg_id}.wav")
        
        if os.path.exists(audio_path):
            chunk = AudioSegment.from_wav(audio_path)
            # Overlay chunk at the specified start time (in ms)
            full_vocals = full_vocals.overlay(chunk, position=int(start_time * 1000))
        else:
            print(f"Warning: Audio file not found: {audio_path}")
            
    full_vocals.export(output_wav, format="wav")
    print(f"Stitched audio saved to {output_wav}")

def final_mix(video_file, full_vocals_wav, bg_music_wav, output_video):
    """
    Mix full_vocals.wav, bg_music.wav, and original video with sidechain compression.
    """
    print("Mixing final video with sidechain compression (ducking)...")
    
    # FFmpeg filter:
    # [2:a] is bg_music, [1:a] is full_vocals
    # 1. sidechaincompress: music volume drops when vocals are present. 
    #    Output [ducked] is JUST the compressed music.
    # 2. amix: mix the [ducked] music back with the [1:a] vocals.
    filter_complex = (
        "[2:a][1:a]sidechaincompress=threshold=0.03:ratio=20:attack=20:release=200[ducked];"
        "[ducked][1:a]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[mixed]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,       # Input 0: Original Video
        "-i", full_vocals_wav,  # Input 1: Stitched Vocals
        "-i", bg_music_wav,     # Input 2: Background Music/Instrumental
        "-filter_complex", filter_complex,
        "-map", "0:v",          # Map video from original
        "-map", "[mixed]",       # Map mixed audio
        "-c:v", "copy",         # Copy video stream (no re-encoding)
        "-c:a", "aac",           # Encode audio as AAC
        "-shortest",            # Match shortest duration (usually the video)
        output_video
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Final video saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge stitched vocals with background music and original video.")
    parser.add_argument("input_json", help="Path to the translated JSON file.")
    parser.add_argument("video_file", help="Path to the original video file.")
    parser.add_argument("bg_music", help="Path to the instrumental/background audio file.")
    parser.add_argument("--output", help="Path to the final output video file.")
    
    args = parser.parse_args()
    
    # Derive paths
    base_dir = os.path.dirname(args.input_json)
    json_name = os.path.basename(args.input_json)
    audio_dir = os.path.join(base_dir, json_name.replace(".json", "_audio"))
    full_vocals_wav = os.path.join(base_dir, "full_vocals.wav")
    
    if not args.output:
        video_base, _ = os.path.splitext(os.path.basename(args.video_file))
        args.output = os.path.join(base_dir, f"{video_base}_final.mp4")
    
    # 1. Get total duration
    total_duration = get_audio_duration(args.video_file)
    if total_duration == 0:
        print("Failed to get video duration. Exiting.")
        return

    # 2. Timeline Stitching
    stitch_audio(args.input_json, audio_dir, total_duration, full_vocals_wav)
    
    # 3. Final Mixing
    final_mix(args.video_file, full_vocals_wav, args.bg_music, args.output)

if __name__ == "__main__":
    main()
