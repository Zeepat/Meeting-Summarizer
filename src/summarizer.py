import json
import requests
from typing import Dict, Optional, List
import sys
import subprocess
import time
import re

class MeetingSummarizer:
    def __init__(self, model: str = "gemma3:4b", api_base: str = "http://localhost:11434"):
        """Initialize the meeting summarizer with Gemma model through Ollama.
        
        Args:
            model: The model name in Ollama
            api_base: The base URL for Ollama API
        """
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.chunk_size = 2000  # Number of characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks to maintain context
        
        # Check if Ollama is running and properly set up
        self._check_ollama_setup()
        
    def _split_transcript(self, transcript: str) -> List[str]:
        """Split a long transcript into overlapping chunks.
        
        Args:
            transcript: The full transcript text
            
        Returns:
            List of transcript chunks
        """
        # Split into sentences (roughly)
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_text = ' '.join(current_chunk[-3:])  # Keep last 3 sentences
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length if overlap_text else sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _merge_summaries(self, chunk_results: List[Dict]) -> Dict:
        """Merge multiple chunk summaries into a single summary.
        
        Args:
            chunk_results: List of summary dictionaries from each chunk
            
        Returns:
            Merged summary dictionary
        """
        all_summaries = []
        all_key_points = []
        all_action_items = []
        
        for result in chunk_results:
            if result["summary"]:
                all_summaries.append(result["summary"])
            all_key_points.extend(result["key_points"])
            all_action_items.extend(result["action_items"])
        
        # Remove duplicates while preserving order
        def deduplicate(items):
            seen = set()
            return [x for x in items if not (x.lower() in seen or seen.add(x.lower()))]
        
        final_summary = " ".join(all_summaries)
        final_key_points = deduplicate(all_key_points)
        final_action_items = deduplicate(all_action_items)
        
        return {
            "summary": final_summary,
            "key_points": final_key_points,
            "action_items": final_action_items
        }
        
    def _check_ollama_setup(self) -> None:
        """Check if Ollama is properly installed and running."""
        try:
            # Try to connect to Ollama
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_name = self.model.split(":")[0]
            if not any(model_name in model.get("name", "") for model in models):
                print(f"\nModel {self.model} not found in Ollama. Please run:")
                print(f"ollama pull {self.model}")
                print("\nIf Ollama is not installed:")
                print("1. Download Ollama from https://ollama.ai/download")
                print("2. Install and start Ollama")
                print("3. Run the pull command above")
                sys.exit(1)
                
        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to Ollama. Please make sure:")
            print("1. Ollama is installed (https://ollama.ai/download)")
            print("2. Ollama service is running")
            print("3. Try running 'ollama serve' in a new terminal")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"\nError connecting to Ollama: {str(e)}")
            print("Please make sure Ollama is properly installed and running.")
            sys.exit(1)
            
    def _generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 2) -> str:
        """Generate text using Ollama API with retries.
        
        Args:
            prompt: The input prompt for the model
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated text response
        """
        for attempt in range(max_retries):
            try:
                # First, try a simple completion to warm up the model
                if attempt == 0:
                    warmup_response = requests.post(
                        f"{self.api_base}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": "Hello",
                            "stream": False,
                        },
                        timeout=10
                    )
                    warmup_response.raise_for_status()
                
                # Now try the actual generation
                response = requests.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 1024,  # Limit response length
                        }
                    },
                    timeout=60  # Increased timeout
                )
                response.raise_for_status()
                return response.json()["response"]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"\nError generating summary: {str(e)}")
                    print("Please make sure Ollama is running and the model is properly installed.")
                    sys.exit(1)
                else:
                    print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
    def summarize(self, transcript: str, max_length: Optional[int] = None) -> Dict:
        """Summarize a meeting transcript.
        
        Args:
            transcript: The meeting transcript to summarize
            max_length: Optional maximum length for the summary
            
        Returns:
            Dict containing the summary and key points
        """
        if not transcript.strip():
            return {
                "summary": "No transcript content provided.",
                "key_points": [],
                "action_items": []
            }

        # For short transcripts, process directly
        if len(transcript) <= self.chunk_size:
            return self._summarize_chunk(transcript, max_length)
            
        # For long transcripts, split into chunks and process each
        print("\nTranscript is long, processing in chunks...")
        chunks = self._split_transcript(transcript)
        chunk_results = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i} of {len(chunks)}...")
            result = self._summarize_chunk(chunk, max_length)
            chunk_results.append(result)
            
        # Merge results from all chunks
        print("Merging summaries...")
        return self._merge_summaries(chunk_results)

    def _summarize_chunk(self, text: str, max_length: Optional[int] = None) -> Dict:
        """Summarize a single chunk of text.
        
        Args:
            text: The text to summarize
            max_length: Optional maximum length for the summary
            
        Returns:
            Dict containing the summary and key points
        """
        prompt = f"""Summarize this meeting transcript concisely:

{text}

Format your response as:
SUMMARY:
[Write a brief summary]

KEY POINTS:
- [Point 1]
- [Point 2]

ACTION ITEMS:
- [Item 1]
- [Item 2]"""
        
        if max_length:
            prompt += f"\n\nKeep response under {max_length} characters."
            
        # Generate summary using Gemma
        response = self._generate(prompt)
        
        if not response.strip():
            return {
                "summary": "The model was unable to generate a summary.",
                "key_points": [],
                "action_items": []
            }
        
        # Parse the response into structured format
        summary = ""
        key_points = []
        action_items = []
        
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('SUMMARY:'):
                current_section = 'summary'
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('KEY POINTS:'):
                current_section = 'key_points'
            elif line.startswith('ACTION ITEMS:'):
                current_section = 'action_items'
            elif line.startswith('- ') and current_section in ['key_points', 'action_items']:
                item = line.replace('- ', '').strip()
                if item:  # Only add non-empty items
                    if current_section == 'key_points':
                        key_points.append(item)
                    else:
                        action_items.append(item)
        
        # If we failed to parse any content but have a response, use it as the summary
        if not summary and not key_points and not action_items and response.strip():
            summary = response.strip()
                    
        return {
            "summary": summary or "No clear summary could be generated from the transcript.",
            "key_points": key_points,
            "action_items": action_items
        } 