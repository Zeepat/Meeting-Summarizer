import os
from typing import List, Optional
import librosa

def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    return librosa.get_duration(path=file_path)

def chunk_audio(file_path: str, chunk_duration: int = 30, overlap: int = 2) -> List[str]:
    """Split a long audio file into chunks with overlap.
    
    Args:
        file_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of paths to the chunked audio files
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Calculate chunk parameters
    samples_per_chunk = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    
    # Create temporary directory for chunks
    base_dir = os.path.dirname(file_path)
    temp_dir = os.path.join(base_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    chunk_files = []
    start_idx = 0
    chunk_num = 0
    
    while start_idx < len(y):
        # Calculate end index for this chunk
        end_idx = min(start_idx + samples_per_chunk, len(y))
        
        # Extract chunk
        chunk = y[start_idx:end_idx]
        
        # Save chunk to file
        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num}.wav")
        librosa.output.write_wav(chunk_path, chunk, sr)
        chunk_files.append(chunk_path)
        
        # Move start index for next chunk
        start_idx = end_idx - overlap_samples
        chunk_num += 1
    
    return chunk_files

def cleanup_chunks(chunk_files: List[str]) -> None:
    """Clean up temporary chunk files.
    
    Args:
        chunk_files: List of paths to chunk files to delete
    """
    for file in chunk_files:
        try:
            os.remove(file)
        except OSError:
            pass
    
    # Try to remove the temp directory
    try:
        temp_dir = os.path.dirname(chunk_files[0])
        os.rmdir(temp_dir)
    except OSError:
        pass 