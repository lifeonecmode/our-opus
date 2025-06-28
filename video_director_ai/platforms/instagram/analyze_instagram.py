#!/usr/bin/env python3
"""
Analyze Instagram reel with full cinematographic analysis
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from director_brain_lite import DirectorBrainLite
from settings import Config

async def main():
    """Run full analysis on Instagram reel."""
    # Use the already downloaded video
    video_path = "test_downloads/Video by christianpolk.mp4"
    
    print("="*60)
    print("FULL CINEMATOGRAPHIC ANALYSIS")
    print("="*60)
    print(f"\nAnalyzing: {video_path}")
    
    # Initialize Director Brain Lite
    director = DirectorBrainLite()
    
    # Run analysis
    results = await director.analyze_video(video_path, output_format='all')
    
    # Display summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    metadata = results['metadata']
    print(f"\nVideo Information:")
    print(f"- File: {metadata['filename']}")
    print(f"- Duration: {metadata['duration']:.2f} seconds")
    print(f"- Resolution: {metadata['width']}x{metadata['height']}")
    print(f"- FPS: {metadata['fps']}")
    
    synthesis = results['synthesis']
    print(f"\nAnalysis Summary:")
    print(f"- Frames analyzed: {synthesis['total_frames_analyzed']}")
    print(f"- Scenes detected: {synthesis['total_scenes']}")
    print(f"- Average scene duration: {synthesis['average_scene_duration']:.2f}s")
    
    style = synthesis['cinematographic_style']
    print(f"\nCinematographic Style:")
    print(f"- Dominant shot type: {style['dominant_shot_type']}")
    print("- Shot distribution:")
    for shot, count in style['shot_distribution'].items():
        print(f"  - {shot}: {count}")
    
    key_moments = synthesis['key_moments']
    if key_moments:
        print(f"\nTop Key Moments:")
        for i, moment in enumerate(key_moments[:3], 1):
            print(f"{i}. {moment['timestamp']:.1f}s (score: {moment['score']:.2f})")
            print(f"   {moment['description']}")
    
    # Sample frame analysis
    if results['cinematography']:
        print(f"\nSample Frame Analysis (10s mark):")
        sample_frame = None
        for frame in results['cinematography']:
            if frame['timestamp'] >= 10.0:
                sample_frame = frame
                break
                
        if sample_frame:
            print(f"- Timestamp: {sample_frame['timestamp']:.1f}s")
            print(f"- Shot type: {sample_frame['shot_type']['shot_type']}")
            print(f"- Camera angle: {sample_frame['shot_type']['camera_angle']['angle']}")
            print(f"- Camera movement: {sample_frame['motion']['camera_movement']['type']}")
            print(f"- Motion intensity: {sample_frame['motion']['motion_intensity']['intensity']}")
            print(f"- Color grading: {sample_frame['color_grading']['color_grading']['style']}")
            print(f"- Color temperature: {sample_frame['color_grading']['temperature']['temperature']}")
            print(f"- Dominant emotion: {sample_frame['emotions']['dominant_emotion']}")
            print(f"- Overall score: {sample_frame['cinematography_score']['overall']:.2f}")
            print(f"\nDirector's Notes:")
            print(f"{sample_frame['directors_notes']}")
    
    print(f"\nFull results saved to: {Config.ANALYSIS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())