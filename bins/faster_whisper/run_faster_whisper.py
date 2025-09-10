#!/usr/bin/env python3
"""
Faster Whisper CLI Tool for audio transcription
"""

# TODO: must load model from local
# TODO: create GitHub action to distribute binaries
# TODO: make use of version.py similar to main repo for binary naming

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Union

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper package not found. Please install it with: pip install faster-whisper")
    sys.exit(1)


class FasterWhisperCLI:
    def __init__(self):
        self.model = None

    def load_model(
        self,
        model_size_or_path: str,
        device: str = "auto",
        cpu_threads: int = 0,
        download_root: Optional[str] = None,
        local_files_only: bool = True
    ) -> WhisperModel:
        """Load the Whisper model with specified parameters"""
        try:
            print(f"Loading model: {model_size_or_path}")
            print(f"Device: {device}")
            print(f"CPU threads: {cpu_threads}")
            print(f"Download root: {download_root}")
            print(f"Local files only: {local_files_only}")

            # Prepare model initialization parameters
            model_kwargs = {
                "device": device,
                "local_files_only": local_files_only
            }

            if cpu_threads > 0:
                model_kwargs["cpu_threads"] = cpu_threads

            if download_root:
                model_kwargs["download_root"] = download_root

            self.model = WhisperModel(model_size_or_path, **model_kwargs)
            print("Model loaded successfully!")
            return self.model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    def transcribe_to_file(
        self,
        input_file: str,
        output_file: str,
        model_size_or_path: str = "base",
        device: str = "auto",
        cpu_threads: int = 0,
        download_root: Optional[str] = None,
        local_files_only: bool = True
    ) -> None:
        """
        Transcribe an audio file and save the result to a text file
        """
        # Validate input file
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            sys.exit(1)

        # Load model if not already loaded
        if self.model is None:
            self.load_model(
                model_size_or_path=model_size_or_path,
                device=device,
                cpu_threads=cpu_threads,
                download_root=download_root,
                local_files_only=local_files_only
            )

        try:
            print(f"Transcribing: {input_file}")

            # Perform transcription
            segments, info = self.model.transcribe(input_file)

            # Prepare output
            output_lines = []
            output_lines.append(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            output_lines.append(f"Duration: {info.duration:.2f} seconds")
            output_lines.append("=" * 50)
            output_lines.append("")

            # Process segments
            for segment in segments:
                timestamp = f"[{segment.start:.2f} -> {segment.end:.2f}]"
                text = segment.text.strip()
                output_lines.append(f"{timestamp} {text}")

            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))

            print(f"Transcription completed successfully!")
            print(f"Output saved to: {output_file}")

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            sys.exit(1)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Faster Whisper CLI Tool for audio transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python faster_whisper.py --function transcribe_to_file --input audio.wav --output transcript.txt
  python faster_whisper.py --function transcribe_to_file --input audio.mp3 --output result.txt --model_size_or_path large-v2 --device cuda
        """
    )

    # Required arguments
    parser.add_argument(
        "--function",
        required=True,
        choices=["transcribe_to_file"],
        help="Function to execute"
    )

    # Function-specific arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input audio file"
    )

    parser.add_argument(
        "--output",
        dest="output_file",
        required=True,
        help="Path to output text file"
    )

    # Model configuration
    parser.add_argument(
        "--model_size_or_path",
        default="base",
        help="Model size (tiny, base, small, medium, large, large-v2, large-v3) or path to model directory (default: base)"
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for inference (default: auto)"
    )

    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="Number of CPU threads to use (0 = auto, default: 0)"
    )

    parser.add_argument(
        "--download_root",
        help="Directory to save downloaded models"
    )

    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=True,
        help="Use only local files, don't download from internet (default: True)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize CLI
    cli = FasterWhisperCLI()

    # Execute requested function
    if args.function == "transcribe_to_file":
        cli.transcribe_to_file(
            input_file=args.input,
            output_file=args.output_file,
            model_size_or_path=args.model_size_or_path,
            device=args.device,
            cpu_threads=args.cpu_threads,
            download_root=args.download_root,
            local_files_only=args.local_files_only
        )
    else:
        print(f"Error: Unknown function '{args.function}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
