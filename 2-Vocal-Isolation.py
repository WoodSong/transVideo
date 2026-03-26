import argparse
import os
import sys
import subprocess
from audio_separator.separator import Separator

def extract_audio(input_file):
    """
    Extracts audio from a video file using ffmpeg to ensure compatibility.
    """
    base, _ = os.path.splitext(input_file)
    output_audio = f"{base}_temp_audio.wav"
    
    print(f"Extracting audio from {input_file} to {output_audio}...")
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_audio
    ]
    
    try:
        # Run ffmpeg, suppress output unless there's an error
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}", file=sys.stderr)
        return input_file # Fallback to original file

def separate_audio(input_file, model_name="model_bs_roformer_ep_317_sdr_12.9755.ckpt"):
    """
    Separates vocals and instruments from a video/audio file using audio-separator.
    """
    # Extract audio first to avoid format recognition issues
    processing_file = extract_audio(input_file)
    is_temp = processing_file != input_file

    # Create an output directory based on the input file's name (minus extension)
    input_dir = os.path.dirname(os.path.abspath(input_file))
    input_filename = os.path.basename(input_file)
    input_basename = os.path.splitext(input_filename)[0]
    
    # We want to keep the output in the same directory as the input
    output_dir = input_dir
    
    print(f"Initializing separator with model: {model_name}")
    
    # Initialize the Separator
    separator = Separator(
        model_file_dir="./models",
        output_dir=output_dir,
        output_format="WAV",
        normalization_threshold=0.9,
    )

    # Load the model
    print(f"Loading model {model_name}...")
    separator.load_model(model_name)

    # Perform separation
    print(f"Starting separation for: {input_filename}")
    output_files = separator.separate(processing_file)

    print(f"Separation completed. Output files: {output_files}")
    
    # Cleanup temporary file
    if is_temp and os.path.exists(processing_file):
        os.remove(processing_file)
        print(f"Cleaned up temporary file: {processing_file}")

    for file in output_files:
        if "(Vocals)" in file:
            print(f"Vocals saved to: {file}")
        elif "(Instrumental)" in file:
            print(f"Instrumental saved to: {file}")

def main():
    parser = argparse.ArgumentParser(description="Separate vocals and instruments from a video file.")
    parser.add_argument("input", help="Path to the input video or audio file.")
    parser.add_argument("--model", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt", help="Model to use for separation (default: model_bs_roformer_ep_317_sdr_12.9755.ckpt).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        separate_audio(args.input, args.model)
    except Exception as e:
        import traceback
        print("\n--- ERROR DETAILS ---", file=sys.stderr)
        traceback.print_exc()
        print("---------------------\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
