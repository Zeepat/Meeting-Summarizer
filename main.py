import argparse
import json
import os
from datetime import datetime
from src.transcriber import AudioTranscriber
from src.summarizer import MeetingSummarizer

def main():
    parser = argparse.ArgumentParser(description="Meeting Summarizer - Transcribe and summarize meeting recordings")
    parser.add_argument("--input", required=True, help="Path to the input audio file")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--max-summary-length", type=int, help="Maximum length of the summary in characters")
    parser.add_argument("--duration", type=float, default=None, help="Maximum duration in seconds to process (for testing)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    transcript_file = os.path.join(args.output_dir, f"{base_name}_{timestamp}_transcript.txt")
    summary_file = os.path.join(args.output_dir, f"{base_name}_{timestamp}_summary.json")
    
    print("Initializing models...")
    transcriber = AudioTranscriber()
    summarizer = MeetingSummarizer()
    
    print(f"Transcribing audio file: {args.input}")
    if args.duration:
        print(f"Processing only the first {args.duration} seconds for testing")
    transcript_result = transcriber.transcribe(args.input, max_duration=args.duration)
    
    # Save transcript
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_result["text"])
    print(f"Transcript saved to: {transcript_file}")
    
    print("Generating summary...")
    summary_result = summarizer.summarize(
        transcript_result["text"],
        max_length=args.max_summary_length
    )
    
    # Save summary
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}")
    
    # Print summary to console
    print("\nMeeting Summary:")
    print("===============")
    print(f"Summary: {summary_result['summary']}\n")
    print("Key Points:")
    for point in summary_result["key_points"]:
        print(f"- {point}")
    print("\nAction Items:")
    for item in summary_result["action_items"]:
        print(f"- {item}")

if __name__ == "__main__":
    main() 