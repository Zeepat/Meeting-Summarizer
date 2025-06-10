# Meeting Summarizer

A powerful tool that transcribes meeting audio using OpenAI's Whisper Large V3 model and generates summaries using Gemma 4B through Ollama.

## Features

- Audio transcription using state-of-the-art Whisper Large V3
- Meeting summarization using Gemma 4B
- Support for multiple audio formats
- Easy-to-use interface

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster transcription)
- [Ollama](https://ollama.ai/) installed and running locally

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Pull the Gemma model using Ollama:
```bash
ollama pull gemma:4b
```

## Usage

1. Place your meeting audio file in the `input` directory
2. Run the summarizer:
```bash
python main.py --input input/your_meeting.mp3
```

## Project Structure

```
Meeting-Summarizer/
├── input/                  # Directory for input audio files
├── output/                 # Directory for transcripts and summaries
├── src/
│   ├── __init__.py
│   ├── transcriber.py     # Whisper transcription logic
│   ├── summarizer.py      # Gemma summarization logic
│   └── utils.py           # Utility functions
├── main.py                # Main application entry point
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## License

MIT License