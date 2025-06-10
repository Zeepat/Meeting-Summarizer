import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Dict, Optional
import os
import librosa
import soundfile as sf
import numpy as np

class AudioTranscriber:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: Optional[str] = None):
        """Initialize the transcriber with Whisper model.
        
        Args:
            model_id: The Hugging Face model ID for Whisper
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Create pipeline first with basic settings
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def _trim_audio(self, audio_path: str, max_duration: float) -> str:
        """Trim audio file to specified duration.
        
        Args:
            audio_path: Path to the audio file
            max_duration: Maximum duration in seconds
            
        Returns:
            Path to the trimmed audio file
        """
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None, duration=max_duration)
        
        # Create a temporary file for the trimmed audio
        base_dir = os.path.dirname(audio_path)
        temp_dir = os.path.join(base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, "trimmed_audio.wav")
        sf.write(temp_path, y, sr)
        
        return temp_path

    def transcribe(self, audio_path: str, max_duration: Optional[float] = None) -> Dict:
        """Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            max_duration: Optional maximum duration in seconds to transcribe
            
        Returns:
            Dict containing the transcription and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # If max_duration is specified, trim the audio
        if max_duration is not None:
            temp_audio_path = self._trim_audio(audio_path, max_duration)
            audio_to_transcribe = temp_audio_path
        else:
            audio_to_transcribe = audio_path
            
        try:
            # Run transcription with simplified parameters
            result = self.pipe(
                audio_to_transcribe,
                chunk_length_s=30,  # Process audio in 30-second chunks
                return_timestamps=True
            )
            
            return {
                "text": result["text"],
                "chunks": result.get("chunks", []),
                "language": result.get("language", "unknown")
            }
        finally:
            # Clean up temporary file if it was created
            if max_duration is not None:
                try:
                    os.remove(temp_audio_path)
                    os.rmdir(os.path.dirname(temp_audio_path))
                except OSError:
                    pass 