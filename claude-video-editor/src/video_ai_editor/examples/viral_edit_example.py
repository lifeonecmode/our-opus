#!/usr/bin/env python3
"""
Example: Create a viral video edit using state-based editing
No After Effects required - pure code-based video editing
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from video_ai_editor.core.video_state_editor import VideoStateEditor, Video
from instagram_frame_analyzer import InstagramFrameAnalyzer


async def create_viral_video(video_path: str, analysis_path: str = None):
    """
    Create a viral video using React-style state management
    """
    print("🎬 Creating viral video edit...")
    
    # Initialize editor
    editor = VideoStateEditor()
    
    # Load or generate analysis
    if analysis_path and Path(analysis_path).exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
    else:
        print("📊 Analyzing video...")
        analyzer = InstagramFrameAnalyzer(video_path, "temp_analysis")
        analyzer.analyze_video(interval=0.5)  # Analyze every 0.5 seconds
        analyzer.save_outputs()
        
        # Load generated analysis
        analysis_files = list(Path("temp_analysis").glob("analysis_*.json"))
        if analysis_files:
            with open(analysis_files[0], 'r') as f:
                analysis = json.load(f)
        else:
            raise RuntimeError("Analysis failed")
    
    print("✨ Building viral edit...")
    
    # Add main video clip
    main_clip_id = editor.addClip(video_path, start=0)
    
    # Find key moments from analysis
    frames = analysis.get('frame_data', [])
    metadata = analysis.get('metadata', {})
    duration = min(metadata.get('duration', 60), 60)  # Max 60 seconds
    
    # 1. HOOK SECTION (0-3s)
    # Black screen fade in
    editor.addEffect(main_clip_id, "fade", {
        "direction": "in",
        "duration": 0.5,
        "start": 0
    })
    
    # Hook text
    editor.addText(
        "WAIT FOR IT...",
        style={
            "fontSize": 72,
            "color": "#FFD700",  # Gold
            "position": {"x": 540, "y": 960},
            "font": "Arial-Bold"
        },
        start=0.5,
        duration=2.5,
        animate={
            "scale": [0, 1.2, 1],
            "opacity": [0, 1],
            "duration": 0.5
        }
    )
    
    # 2. EMOTIONAL PEAKS - Add zoom effects
    emotional_peaks = [f for f in frames if f.get('emotional_tone', {}).get('confidence', 0) > 0.8]
    
    for i, peak in enumerate(emotional_peaks[:5]):  # Top 5 peaks
        editor.addEffect(main_clip_id, "zoom", {
            "from": 1.0,
            "to": 1.1,
            "duration": 0.5,
            "at": peak['timestamp']
        }, timestamp=peak['timestamp'])
        
        # Add shake on high intensity
        if peak.get('emotional_tone', {}).get('confidence', 0) > 0.9:
            editor.addEffect(main_clip_id, "shake", {
                "amplitude": 3,
                "frequency": 15,
                "duration": 0.2
            }, timestamp=peak['timestamp'])
    
    # 3. LOW ENERGY SECTIONS - Speed up
    low_energy_sections = []
    for i, frame in enumerate(frames):
        if frame.get('cinematography_score', 0) < 0.3:
            if not low_energy_sections or frame['timestamp'] - low_energy_sections[-1][-1] > 1:
                low_energy_sections.append([frame['timestamp']])
            else:
                low_energy_sections[-1].append(frame['timestamp'])
    
    for section in low_energy_sections:
        if len(section) > 3:  # More than 3 seconds
            editor.addEffect(main_clip_id, "speed", {
                "speed": 1.5,
                "start": section[0],
                "end": section[-1]
            })
    
    # 4. SOCIAL PROOF (if applicable)
    if duration > 10:
        editor.addText(
            "Join 47,000+ Happy Users",
            style={
                "fontSize": 48,
                "color": "#FFFFFF",
                "position": {"x": 540, "y": 1600},
                "font": "Arial"
            },
            start=8,
            duration=3,
            animate={
                "opacity": [0, 1, 1, 0],
                "duration": 3
            }
        )
    
    # 5. BENEFITS DISPLAY (35-45s)
    if duration > 35:
        benefits = [
            "✓ Save 3 hours daily",
            "✓ Feel energized",
            "✓ Join the community"
        ]
        
        for i, benefit in enumerate(benefits):
            editor.addText(
                benefit,
                style={
                    "fontSize": 56,
                    "color": "#4CAF50",  # Green
                    "position": {"x": 540, "y": 800 + (i * 120)},
                    "font": "Arial"
                },
                start=35 + (i * 3),
                duration=3,
                animate={
                    "scale": [0, 1.1, 1],
                    "opacity": [0, 1],
                    "duration": 0.3
                }
            )
    
    # 6. URGENCY (50-57s)
    if duration > 50:
        editor.addText(
            "⏰ LIMITED TIME OFFER",
            style={
                "fontSize": 64,
                "color": "#FF6347",  # Tomato red
                "position": {"x": 540, "y": 500},
                "font": "Arial-Bold",
                "box": True
            },
            start=50,
            duration=7,
            animate={
                "scale": [1, 1.05, 1],  # Pulse
                "duration": 0.5,
                "repeat": 14  # Pulse for 7 seconds
            }
        )
    
    # 7. CALL TO ACTION (57-60s)
    editor.addText(
        "👆 TAP THE LINK NOW",
        style={
            "fontSize": 72,
            "color": "#FFD700",  # Gold
            "position": {"x": 540, "y": 960},
            "font": "Arial-Bold",
            "box": True
        },
        start=duration - 3,
        duration=3,
        animate={
            "scale": [0, 1.3, 1],
            "shake": {"x": 5, "y": 5},
            "duration": 0.3
        }
    )
    
    # Add arrow pointing up
    editor.addText(
        "⬆️",
        style={
            "fontSize": 120,
            "color": "#FFD700",
            "position": {"x": 540, "y": 1200},
            "font": "Arial"
        },
        start=duration - 3,
        duration=3,
        animate={
            "position": {"y": [1200, 1150, 1200]},  # Bounce
            "duration": 1,
            "repeat": 3
        }
    )
    
    # Color correction for viral feel
    editor.addEffect(main_clip_id, "color", {
        "brightness": 0.05,
        "contrast": 1.1,
        "saturation": 1.1  # Slightly more vibrant
    })
    
    # Set final duration
    editor.setState({"duration": duration})
    
    # Render the video
    print("🎥 Rendering video...")
    output_path = "viral_edit_output.mp4"
    success = await editor.render(output_path)
    
    if success:
        print(f"✅ Video created successfully: {output_path}")
        print("📱 Platform versions:")
        print("  - viral_edit_output_tiktok.mp4")
        print("  - viral_edit_output_instagram.mp4")
        print("  - viral_edit_output_youtube.mp4")
    else:
        print("❌ Render failed")
    
    return success


async def create_with_declarative_api():
    """
    Alternative: Create viral video using declarative API
    """
    print("🎬 Creating viral video with declarative API...")
    
    # Create video sequence declaratively (like JSX)
    editor = Video.sequence(
        duration=60,
        fps=30,
        children=[
            # Main video clip
            Video.clip("input.mp4", start=0),
            
            # Hook text
            Video.text(
                "WAIT FOR IT...",
                style={
                    "fontSize": 72,
                    "color": "#FFD700",
                    "position": {"x": 540, "y": 960},
                    "font": "Arial-Bold"
                },
                start=0.5,
                duration=2.5,
                animate={
                    "scale": [0, 1.2, 1],
                    "opacity": [0, 1]
                }
            ),
            
            # Benefits
            Video.text(
                "✓ Save Time",
                style={
                    "fontSize": 56,
                    "color": "#4CAF50",
                    "position": {"x": 540, "y": 800}
                },
                start=35,
                duration=3,
                animate={"scale": [0, 1.1, 1]}
            ),
            
            # CTA
            Video.text(
                "TAP THE LINK NOW",
                style={
                    "fontSize": 72,
                    "color": "#FFD700",
                    "position": {"x": 540, "y": 960},
                    "box": True
                },
                start=57,
                duration=3,
                animate={
                    "scale": [0, 1.3, 1],
                    "shake": {"x": 5, "y": 5}
                }
            )
        ]
    )
    
    # Render
    await editor.render("declarative_output.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create viral video edit")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--analysis", help="Pre-computed analysis JSON")
    parser.add_argument("--declarative", action="store_true", 
                       help="Use declarative API")
    
    args = parser.parse_args()
    
    if args.declarative:
        asyncio.run(create_with_declarative_api())
    else:
        asyncio.run(create_viral_video(args.video, args.analysis))