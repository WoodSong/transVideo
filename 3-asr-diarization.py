import torch
import functools

# Fix for PyTorch 2.6+ weights_only=True security defaults
_original_load = torch.load

@functools.wraps(_original_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load
# End of fix

import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

import gc
import os
import argparse
import json


def transcribe_and_diarize(audio_file, device="cpu", model_name="large-v3", hf_token=None):
    """
    Performs ASR, alignment, and diarization on an audio file.
    Note: ASR is forced to CPU/int8 for stability/support, while
    Alignment and Diarization use the specified device (e.g., mps or cuda).
    """
    print(f"call transcribe_and_diarize with {audio_file}, {device}, {model_name}, {hf_token}   ")

    # 1. Transcribe with original whisper (forced to CPU and int8 as requested)
    asr_device = "cpu"
    asr_compute_type = "int8"
    
    print(f"Loading Whisper model: {model_name} on {asr_device} with {asr_compute_type}...")
    model = whisperx.load_model(model_name, asr_device, compute_type=asr_compute_type, download_root="./models")

    print(f"Transcribing {audio_file}...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)
    
    # Free ASR model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Align whisper output (using specified device, e.g., mps)
    print(f"Aligning transcription on {device}...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Free alignment model memory
    del model_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Diarize (using specified device, e.g., mps)
    print(f"Starting diarization on {device}...")
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    
    # Add diarization to aligned segments
    diarize_segments = diarize_model(audio)
    result = assign_word_speakers(diarize_segments, result)

    return result

def save_results(result, output_file):
    """
    Saves the result to a JSON file.
    """
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="WhisperX ASR, alignment, and diarization.")
    parser.add_argument("audio_file", help="Path to the audio file to process.")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, mps). Default: cpu.")
    parser.add_argument("--model", default="large-v3", help="Whisper model to use. Default: large-v3.")
    parser.add_argument("--token", default="HUGGINGFACE_TOKEN_REMOVED", help="Hugging Face access token for diarization.")
    parser.add_argument("--output", help="Path to the output JSON file. Default: [audio_file]_transcription.json")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        return

    # Determine device automatically if possible
    device = args.device
    if device == "cpu" and torch.backends.mps.is_available():
        print("MPS compatible device detected. Using 'mps' for acceleration.")
        device = "mps"

    try:
        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)

        result = transcribe_and_diarize(
            args.audio_file, 
            device=device, 
            model_name=args.model, 
            hf_token=args.token
        )

        output_file = args.output
        if not output_file:
            base, _ = os.path.splitext(args.audio_file)
            output_file = f"{base}_transcription.json"

        save_results(result, output_file)
        print("Process completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
