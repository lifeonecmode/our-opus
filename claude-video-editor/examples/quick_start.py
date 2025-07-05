#!/usr/bin/env python3
"""
Claude Video Editor - Quick Start Example

This example demonstrates how to use Claude Video Editor to automatically
edit videos with AI-powered decision making.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ai_editor.core.video_state_editor import VideoStateEditor
from core.claude_orchestrator import ClaudeDecisionEngine


async def quick_start_example():
    """
    Quick start example showing basic video editing workflow
    """
    
    print("🎬 Claude Video Editor - Quick Start")
    print("=" * 50)
    
    # Set up project directory
    project_dir = Path("./demo_projects/viral_video_example")
    
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_dir}")
        print("Please run this script from the examples/ directory")
        return
    
    # Initialize Claude Decision Engine
    print("🤖 Initializing Claude Decision Engine...")
    
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        print("❌ CLAUDE_API_KEY environment variable not set")
        print("Please set your Claude API key:")
        print("export CLAUDE_API_KEY='your_key_here'")
        return
    
    engine = ClaudeDecisionEngine(
        project_dir=str(project_dir),
        claude_api_key=claude_api_key
    )
    
    # Analyze project
    print("📊 Analyzing project structure...")
    analysis = await engine.analyze_project()
    print(f"✅ Found {len(analysis.get('videos', []))} videos")
    
    # Create edit decisions
    print("🎯 Creating edit decisions...")
    decisions = await engine.create_edit_decisions(style="viral")
    print(f"✅ Created {len(decisions.get('clips', []))} edit decisions")
    
    # Execute edit
    print("🎬 Executing edit...")
    output_path = project_dir / "output" / "quick_start_edit.mp4"
    await engine.execute_edit(decisions, str(output_path))
    print(f"✅ Edit complete: {output_path}")
    
    # Create platform variants
    print("📱 Creating platform variants...")
    
    platforms = ["tiktok", "instagram", "youtube"]
    for platform in platforms:
        platform_output = project_dir / "output" / f"quick_start_edit_{platform}.mp4"
        await engine.create_platform_variant(
            str(output_path), 
            str(platform_output), 
            platform
        )
        print(f"✅ {platform.title()} variant: {platform_output}")
    
    print("\n🎉 Quick start complete!")
    print("Check the output/ directory for your edited videos.")


def code_based_example():
    """
    Example showing direct code-based video editing
    """
    
    print("\n🔧 Code-Based Editing Example")
    print("=" * 50)
    
    # Initialize video editor
    editor = VideoStateEditor()
    
    # Add a clip
    clip_id = editor.addClip(
        "demo_projects/viral_video_example/input/sample.mp4",
        start=0,
        duration=10
    )
    
    # Add text overlay
    editor.addText(
        "WAIT FOR IT...",
        style={
            "fontSize": 72,
            "color": "#FFD700",
            "fontWeight": "bold",
            "stroke": "#000000",
            "strokeWidth": 2
        },
        start=0.5,
        duration=3,
        animate={"scale": [0, 1.2, 1]}
    )
    
    # Add zoom effect
    editor.addEffect(clip_id, "zoom", {
        "from": 1.0,
        "to": 1.15,
        "duration": 0.5,
        "easing": "ease_out"
    })
    
    # Add fade transition
    editor.addEffect(clip_id, "fade", {
        "type": "in",
        "duration": 0.5
    })
    
    print("✅ Created edit state with:")
    print(f"  - 1 video clip")
    print(f"  - 1 text overlay") 
    print(f"  - 2 effects (zoom, fade)")
    print("\nTo render: editor.render('output.mp4')")


if __name__ == "__main__":
    print("🎬 Claude Video Editor Examples")
    print("Choose an example:")
    print("1. Quick Start (AI-powered editing)")
    print("2. Code-Based Editing")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        asyncio.run(quick_start_example())
    elif choice == "2":
        code_based_example()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")