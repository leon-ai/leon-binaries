#!/usr/bin/env python3
"""
ECAPA Voice Gender Classifier CLI Tool
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import torch
    import torchaudio
    import soundfile as sf
    from safetensors.torch import load_file
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    sys.exit(1)


# --- FIX: Monkeypatch torchaudio.load ---
# This forces the code to use 'soundfile' (which has bundled binaries)
# instead of trying to find system FFmpeg/TorchCodec.
def patched_load(filepath, **kwargs):
    """
    Patched version of torchaudio.load that uses soundfile backend.
    This avoids dependency on system FFmpeg or TorchCodec.
    """
    # Read file using soundfile (returns: data, samplerate)
    data, sr = sf.read(filepath, dtype='float32')

    # Convert numpy array to torch Tensor
    tensor = torch.from_numpy(data)

    # Soundfile returns (Time, Channels), but Torchaudio expects (Channels, Time)
    if tensor.ndim == 2:
        tensor = tensor.t()  # Transpose
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # Add channel dimension

    return tensor, sr


# Apply the patch
torchaudio.load = patched_load
# ----------------------------------------

# Import after patching
try:
    from model import ECAPA_gender
except ImportError:
    print("Error: model.py not found. Please ensure ECAPA_gender model is available.")
    sys.exit(1)


class VoiceGenderClassifierCLI:
    def __init__(self):
        self.model = None
        self.device = None

    def load_model(
        self,
        model_path: str,
        device: str = "cpu"
    ):
        """Load the ECAPA gender classifier model from local filesystem"""
        try:
            print(f"Loading model from: {model_path}")
            print(f"Device: {device}")

            # Set device
            if device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            # Load model from local path
            # Initialize model first
            self.model = ECAPA_gender(C=1024)

            # Check if model_path is a directory or a file
            if os.path.isdir(model_path):
                # If it's a directory, look for model files
                safetensors_path = os.path.join(model_path, "model.safetensors")
                pytorch_path = os.path.join(model_path, "pytorch_model.bin")

                if os.path.exists(safetensors_path):
                    # Load safetensors file
                    state_dict = load_file(safetensors_path)
                    self.model.load_state_dict(state_dict)
                elif os.path.exists(pytorch_path):
                    # Load pytorch bin file
                    state_dict = torch.load(pytorch_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)
                else:
                    raise FileNotFoundError(f"No model file found in {model_path}")
            else:
                # If it's a file path, load directly based on extension
                if model_path.endswith('.safetensors'):
                    state_dict = load_file(model_path)
                    self.model.load_state_dict(state_dict)
                else:
                    state_dict = torch.load(model_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    def detect_gender(
        self,
        input_file: str,
        model_path: str,
        device: str = "cpu"
    ) -> str:
        """
        Detect gender from an audio file
        
        Args:
            input_file: Path to input audio file
            model_path: Path to model directory
            device: Device to use for inference (cpu or cuda)
            
        Returns:
            Detected gender as string
        """
        # Validate input file
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            sys.exit(1)

        # Load model if not already loaded
        if self.model is None:
            self.load_model(model_path=model_path, device=device)

        try:
            print(f"🎧 Analyzing: {input_file}")

            with torch.no_grad():
                # This will now use our patched loader!
                prediction = self.model.predict(input_file, device=self.device)

            print(f"Gender: {prediction}")
            return prediction

        except Exception as e:
            print(f"Error during gender detection: {str(e)}")
            sys.exit(1)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="ECAPA Voice Gender Classifier CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ecapa_voice_gender_classifier.py --function detect_gender --input audio.wav --model_path ./model --device cpu
  python run_ecapa_voice_gender_classifier.py --function detect_gender --input voice.mp3 --model_path /path/to/model --device cuda
        """
    )

    # Required arguments
    parser.add_argument(
        "--function",
        required=True,
        choices=["detect_gender"],
        help="Function to execute"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input audio file"
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model directory containing the pretrained model files"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize CLI
    cli = VoiceGenderClassifierCLI()

    # Execute requested function
    if args.function == "detect_gender":
        result = cli.detect_gender(
            input_file=args.input,
            model_path=args.model_path,
            device=args.device
        )
        # Print result to stdout for programmatic consumption
        print(result)
    else:
        print(f"Error: Unknown function '{args.function}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
