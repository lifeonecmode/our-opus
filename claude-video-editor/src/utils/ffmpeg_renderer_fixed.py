#!/usr/bin/env python3
"""
Fixed FFmpeg Renderer - Corrected filter syntax and error handling
"""

import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tempfile
import logging

logger = logging.getLogger(__name__)

class FFmpegRenderer:
    """
    Fixed FFmpeg renderer with proper filter syntax
    """
    
    def __init__(self, hardware_accel: bool = False):
        self.hardware_accel = hardware_accel
        self.temp_dir = Path(tempfile.mkdtemp())
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Verify FFmpeg is installed"""
        try:
            subprocess.run(["ffmpeg", "-version"], 
                         capture_output=True, check=True)
        except:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def render_simple(self, input_path: str, output_path: str, 
                     duration: float = 15, 
                     effects: List[Dict] = None,
                     texts: List[Dict] = None) -> bool:
        """
        Simplified render function with better error handling
        """
        cmd = ["ffmpeg", "-y"]  # Overwrite output
        
        # Input file
        cmd.extend(["-i", input_path])
        
        # Duration limit
        if duration:
            cmd.extend(["-t", str(duration)])
        
        # Build filter complex if needed
        filters = []
        
        # Scale to 9:16 vertical (1080x1920)
        filters.append("scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
        
        # Add effects
        if effects:
            for effect in effects:
                if effect["type"] == "zoom":
                    # Simple zoom using scale
                    zoom_from = effect.get("from", 1.0)
                    zoom_to = effect.get("to", 1.1)
                    duration = effect.get("duration", 1.0)
                    filters.append(f"zoompan=z='min(zoom+0.0015,{zoom_to})':d=125:s=1080x1920")
                    
                elif effect["type"] == "fade":
                    direction = effect.get("direction", "in")
                    fade_duration = effect.get("duration", 1.0)
                    filters.append(f"fade={direction}:d={fade_duration}")
                    
                elif effect["type"] == "shake":
                    intensity = effect.get("intensity", 5)
                    filters.append(f"crop=in_w-{intensity}:in_h-{intensity}:{intensity}*random(1):{intensity}*random(2)")
        
        # Apply video filters
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        # Add text overlays using drawtext
        if texts:
            text_filters = []
            for text in texts:
                content = text.get("content", "").replace("'", "\\'").replace(":", "\\:")
                size = text.get("size", 48)
                color = text.get("color", "white")
                x = text.get("x", "(w-text_w)/2")
                y = text.get("y", "(h-text_h)/2")
                
                drawtext = f"drawtext=text='{content}':fontsize={size}:fontcolor={color}:x={x}:y={y}"
                
                # Add timing if specified
                if "start" in text and "duration" in text:
                    start = text["start"]
                    end = start + text["duration"]
                    drawtext += f":enable='between(t,{start},{end})'"
                
                text_filters.append(drawtext)
            
            # Combine with existing filters
            if filters and text_filters:
                all_filters = ",".join(filters + text_filters)
                cmd[cmd.index("-vf") + 1] = all_filters
            elif text_filters:
                cmd.extend(["-vf", ",".join(text_filters)])
        
        # Output settings
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            output_path
        ])
        
        # Log the command
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        # Execute
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                logger.info(f"✅ Render successful: {output_path}")
                return True
            else:
                logger.error(f"❌ Render failed with code {process.returncode}")
                logger.error(f"Error output: {process.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during render: {e}")
            return False


def create_simple_instagram_edit():
    """
    Create a simple Instagram edit of C0018.MP4
    """
    renderer = FFmpegRenderer()
    
    input_path = "/Users/lifeonecmode/PycharmProjects/whisper_editor/video_ai_editor/input/C0018.MP4"
    output_path = "C0018_instagram_simple.mp4"
    
    # Simple effects
    effects = [
        {"type": "fade", "direction": "in", "duration": 0.5},
        {"type": "zoom", "from": 1.0, "to": 1.05, "duration": 15}
    ]
    
    # Simple text overlays
    texts = [
        {
            "content": "AMAZING VIDEO",
            "size": 72,
            "color": "yellow",
            "x": "(w-text_w)/2",
            "y": "100",
            "start": 1,
            "duration": 3
        },
        {
            "content": "TAP FOR MORE",
            "size": 60,
            "color": "white",
            "x": "(w-text_w)/2",
            "y": "h-150",
            "start": 12,
            "duration": 3
        }
    ]
    
    success = renderer.render_simple(
        input_path,
        output_path,
        duration=15,
        effects=effects,
        texts=texts
    )
    
    if success:
        print(f"\n✅ Successfully created: {output_path}")
        print("📱 Ready for Instagram (15s, 9:16)")
    else:
        print("\n❌ Failed to create video")
    
    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_simple_instagram_edit()