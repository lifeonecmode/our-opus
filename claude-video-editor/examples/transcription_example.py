#!/usr/bin/env python3
"""
Audio Transcription Example - Using Whisper with Claude Video Editor

This example shows how to transcribe audio from videos and use the transcription
in your Claude-powered video edits.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper import load_model, transcribe
from video_ai_editor.core.video_state_editor import VideoStateEditor
from core.claude_orchestrator import ClaudeDecisionEngine


def transcribe_video_audio(video_path, model_name="base"):
    """
    Transcribe audio from a video file
    
    Args:
        video_path: Path to video file
        model_name: Whisper model to use (tiny, base, small, medium, large)
    
    Returns:
        Transcription result with text and timestamps
    """
    
    print(f"🎤 Loading Whisper model: {model_name}")
    model = load_model(model_name)
    
    print(f"📹 Transcribing audio from: {video_path}")
    result = transcribe(model, video_path)
    
    return result


def create_subtitled_video(video_path, transcription, output_path):
    """
    Create a video with subtitles based on transcription
    
    Args:
        video_path: Original video file
        transcription: Whisper transcription result
        output_path: Output video with subtitles
    """
    
    editor = VideoStateEditor()
    
    # Add main video clip
    clip_id = editor.addClip(video_path, start=0)
    
    # Add subtitle text overlays based on transcription segments
    for i, segment in enumerate(transcription["segments"]):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        
        # Add subtitle overlay
        editor.addText(
            text,
            style={
                "fontSize": 48,
                "color": "#FFFFFF",
                "backgroundColor": "#000000",
                "padding": 10,
                "position": "bottom_center",
                "fontFamily": "Arial",
                "stroke": "#000000",
                "strokeWidth": 2
            },
            start=start_time,
            duration=end_time - start_time
        )
    
    print(f"🎬 Rendering video with subtitles to: {output_path}")
    editor.render(output_path)


def claude_powered_transcription_edit(project_dir):
    """
    Use Claude to create an edit that incorporates transcription data
    
    Args:
        project_dir: Project directory with videos
    """
    
    print("🤖 Claude-Powered Transcription Edit")
    print("=" * 50)
    
    # Initialize Claude
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        print("❌ CLAUDE_API_KEY not set")
        return
    
    engine = ClaudeDecisionEngine(
        project_dir=project_dir,
        claude_api_key=claude_api_key
    )
    
    # Find videos in project
    video_files = list(Path(project_dir).glob("**/*.mp4"))
    if not video_files:
        print("❌ No video files found in project")
        return
    
    # Transcribe all videos
    transcriptions = {}
    for video_file in video_files:
        print(f"📝 Transcribing: {video_file.name}")
        transcription = transcribe_video_audio(str(video_file))
        transcriptions[str(video_file)] = transcription
        
        # Save transcription
        transcript_file = video_file.with_suffix(".txt")
        with open(transcript_file, "w") as f:
            f.write(transcription["text"])
        print(f"💾 Saved transcript: {transcript_file}")
    
    # Claude analyzes project with transcription data
    print("🧠 Claude analyzing project with transcription data...")
    
    # Add transcription context to Claude's analysis
    engine.add_context("transcriptions", transcriptions)
    
    # Create edit decisions that incorporate speech content
    decisions = engine.create_edit_decisions(
        style="viral",
        use_transcription=True  # Tell Claude to use speech content
    )
    
    print("✅ Claude created edit decisions using transcription data")
    return decisions


def main():
    """
    Main function demonstrating transcription capabilities
    """
    
    print("🎤 Claude Video Editor - Transcription Examples")
    print("=" * 60)
    
    # Check if we have example video
    demo_project = Path("demo_projects/viral_video_example")
    if not demo_project.exists():
        print("❌ Demo project not found")
        print("Please run from examples/ directory")
        return
    
    print("Choose an example:")
    print("1. Basic transcription")
    print("2. Create subtitled video")
    print("3. Claude-powered edit with transcription")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Basic transcription
        video_path = demo_project / "output" / "viral_edit.mp4"
        if video_path.exists():
            result = transcribe_video_audio(str(video_path), "base")
            print(f"\n📝 Transcription Result:")
            print(f"Language: {result['language']}")
            print(f"Text: {result['text']}")
            print(f"Segments: {len(result['segments'])}")
        else:
            print("❌ Demo video not found")
    
    elif choice == "2":
        # Create subtitled video
        video_path = demo_project / "output" / "viral_edit.mp4"
        if video_path.exists():
            result = transcribe_video_audio(str(video_path), "base")
            output_path = demo_project / "output" / "viral_edit_subtitled.mp4"
            create_subtitled_video(str(video_path), result, str(output_path))
            print(f"✅ Created subtitled video: {output_path}")
        else:
            print("❌ Demo video not found")
    
    elif choice == "3":
        # Claude-powered edit with transcription
        decisions = claude_powered_transcription_edit(str(demo_project))
        if decisions:
            print("✅ Claude successfully incorporated transcription data")
            print("Edit decisions created with speech-aware timing")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()