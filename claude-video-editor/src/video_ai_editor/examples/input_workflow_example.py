#!/usr/bin/env python3
"""
Input Workflow Example - Demonstrates automatic video processing

This example shows how to:
1. Set up an input directory for automatic processing
2. Drop videos and have them analyzed automatically
3. Create automatic edits based on video content
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from analyze_all_footage import ProjectAnalyzer
from claude_decision_engine import ClaudeDecisionEngine
from input_processor import InputProcessor


async def example_workflow():
    """Example workflow for automatic input processing"""
    
    # Create example project structure
    project_dir = Path("./example_project")
    project_dir.mkdir(exist_ok=True)
    
    # Create directories
    (project_dir / "input").mkdir(exist_ok=True)
    (project_dir / "footage").mkdir(exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "graphics").mkdir(exist_ok=True)
    (project_dir / "analysis").mkdir(exist_ok=True)
    (project_dir / "outputs").mkdir(exist_ok=True)
    
    print("🎬 Input Directory Workflow Example")
    print("=" * 50)
    print(f"Project directory: {project_dir.absolute()}")
    print("\n📁 Directory structure created:")
    print("  - /input/     <- Drop videos here")
    print("  - /footage/   <- Traditional footage organization")
    print("  - /audio/     <- Audio files")
    print("  - /graphics/  <- Visual assets")
    print("  - /analysis/  <- Analysis outputs")
    print("  - /outputs/   <- Rendered videos")
    
    print("\n🔄 Workflow Options:")
    print("\n1. Manual Processing:")
    print("   python analyze_all_footage.py ./example_project --input")
    print("   -> Analyzes all videos in /input/ directory")
    
    print("\n2. Watch Mode (Auto-process new videos):")
    print("   python input_processor.py ./example_project --watch")
    print("   -> Watches /input/ and processes new videos automatically")
    
    print("\n3. Auto-Edit Mode:")
    print("   python input_processor.py ./example_project --watch --auto-edit")
    print("   -> Watches, analyzes, AND creates automatic edits")
    
    print("\n📝 Instructions:")
    print("1. Drop any video file into the /input/ directory")
    print("2. The system will automatically:")
    print("   - Analyze the video content")
    print("   - Categorize it (speaker/product/lifestyle)")
    print("   - Generate analysis reports")
    print("   - Move processed videos to /processed/")
    print("   - (Optional) Create automatic edits")
    
    print("\n🤖 Claude Integration:")
    print("Videos from /input/ are automatically available to Claude for editing.")
    print("Claude will intelligently use them based on their content type.")
    
    # Example: Process a sample video if one exists
    input_dir = project_dir / "input"
    sample_videos = list(input_dir.glob("*.mp4"))
    
    if sample_videos:
        print(f"\n📹 Found {len(sample_videos)} videos in input directory")
        print("Processing them now...")
        
        # Create processor
        processor = InputProcessor(str(project_dir), auto_edit=True)
        
        # Process existing videos
        await processor.process_existing_videos()
        
        print("\n✅ Processing complete! Check the /analysis/ directory for reports.")
    else:
        print("\n💡 No videos found in input directory.")
        print(f"   Drop video files into: {input_dir.absolute()}")
        print("   Then run one of the commands above.")
    
    # Show how Claude would use input videos
    print("\n🎨 Claude Edit Example:")
    print("After analyzing input videos, Claude can create edits like:")
    print("```python")
    print("# Claude automatically uses categorized videos")
    print("python claude_decision_engine.py ./example_project --style viral")
    print("```")
    print("\nClaude will:")
    print("- Use 'speaker' videos as main narrative")
    print("- Use 'product' videos for hero shots")
    print("- Use 'lifestyle' videos as B-roll")
    print("- Create platform-optimized exports")


if __name__ == "__main__":
    print("Running input workflow example...")
    asyncio.run(example_workflow())