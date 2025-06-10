import json
import requests
from typing import Dict, Optional
import sys
import subprocess
import time

class MeetingSummarizer:
    def __init__(self, model: str = "gemma3:4b", api_base: str = "http://localhost:11434"):
        """Initialize the meeting summarizer with Gemma model through Ollama.
        
        Args:
            model: The model name in Ollama
            api_base: The base URL for Ollama API
        """
        self.model = model
        self.api_base = api_base.rstrip('/')
        
        # Check if Ollama is running and properly set up
        self._check_ollama_setup()
        
    def _check_ollama_setup(self) -> None:
        """Check if Ollama is properly installed and running."""
        try:
            # Try to connect to Ollama
            response = requests.get(f"{self.api_base}/api/tags")
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
                response = requests.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    },
                    timeout=30  # Add timeout to prevent hanging
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
            
        # Create a detailed prompt for the model
        prompt = f"""You are a professional meeting summarizer. Please analyze this meeting transcript carefully and provide:

1. A concise but detailed summary of what happened in the meeting
2. Key points that were discussed or actions that took place
3. Any action items or next steps mentioned

Here is the meeting transcript to analyze:
---
{transcript}
---

Please be specific and detailed in your analysis. Even if the transcript is short or contains partial information, extract as much meaningful content as possible.

Format your response exactly as follows:
Summary: [Write a paragraph summarizing the key events and discussions]

Key Points:
- [List each key point or event]
- [Continue with more points]

Action Items:
- [List any action items or next steps]
- [Continue with more items]

If there are no clear action items, you can omit that section. Focus on providing accurate, meaningful information based on what's in the transcript."""
        
        if max_length:
            prompt += f"\n\nPlease keep your entire response under {max_length} characters."
            
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
                
            if line.startswith('Summary:'):
                current_section = 'summary'
                summary = line.replace('Summary:', '').strip()
            elif line.startswith('Key Points:'):
                current_section = 'key_points'
            elif line.startswith('Action Items:'):
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