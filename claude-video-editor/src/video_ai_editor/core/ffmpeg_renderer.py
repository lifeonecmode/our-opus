#!/usr/bin/env python3
"""
FFmpeg Renderer - Converts component-based edits to FFmpeg commands
Handles all video processing without After Effects
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

@dataclass
class FFmpegFilter:
    """Single FFmpeg filter"""
    name: str
    params: Dict[str, Any]
    input_labels: List[str]
    output_label: str
    
    def to_string(self) -> str:
        """Convert to FFmpeg filter string"""
        param_str = ":".join(f"{k}={v}" for k, v in self.params.items())
        inputs = "".join(f"[{label}]" for label in self.input_labels)
        return f"{inputs}{self.name}={param_str}[{self.output_label}]"


class FFmpegRenderer:
    """
    Renders video edits using FFmpeg without After Effects
    Supports complex filtering, transitions, and effects
    """
    
    def __init__(self, hardware_accel: bool = True):
        self.hardware_accel = hardware_accel
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Check FFmpeg availability
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Verify FFmpeg is installed"""
        try:
            subprocess.run(["ffmpeg", "-version"], 
                         capture_output=True, check=True)
        except:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def render_sequence(self, sequence: Dict[str, Any], output_path: str) -> bool:
        """
        Render a video sequence from component description
        
        Args:
            sequence: Video sequence description
            output_path: Output file path
            
        Returns:
            Success status
        """
        # Parse sequence into FFmpeg command
        command = self._build_command(sequence, output_path)
        
        # Execute render
        logger.info(f"Rendering to: {output_path}")
        logger.debug(f"Command: {' '.join(command)}")
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor progress
            for line in process.stderr:
                if "frame=" in line:
                    # Extract progress
                    parts = line.split()
                    for part in parts:
                        if part.startswith("frame="):
                            frame = int(part.split("=")[1])
                            logger.info(f"Progress: Frame {frame}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("Render complete!")
                return True
            else:
                logger.error(f"Render failed: {process.stderr.read()}")
                return False
                
        except Exception as e:
            logger.error(f"Render error: {e}")
            return False
    
    def _build_command(self, sequence: Dict[str, Any], output_path: str) -> List[str]:
        """Build FFmpeg command from sequence"""
        cmd = ["ffmpeg", "-y"]  # -y for overwrite
        
        # Hardware acceleration
        if self.hardware_accel and self._detect_hardware():
            cmd.extend(["-hwaccel", "videotoolbox"])
        
        # Input files
        inputs = sequence.get("inputs", [])
        for inp in inputs:
            if "trim" in inp:
                # Trim input
                cmd.extend([
                    "-ss", str(inp["trim"]["start"]),
                    "-t", str(inp["trim"]["duration"])
                ])
            cmd.extend(["-i", inp["path"]])
        
        # Build filter complex
        filter_complex = self._build_filter_complex(sequence)
        if filter_complex:
            cmd.extend(["-filter_complex", filter_complex])
        
        # Output settings
        cmd.extend([
            "-map", "[out]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"
        ])
        
        # Audio settings
        if sequence.get("audio"):
            cmd.extend([
                "-map", "0:a?",
                "-c:a", "aac",
                "-b:a", "192k"
            ])
        
        # Frame rate
        fps = sequence.get("fps", 30)
        cmd.extend(["-r", str(fps)])
        
        # Resolution
        if "resolution" in sequence:
            w, h = sequence["resolution"]
            cmd.extend(["-s", f"{w}x{h}"])
        
        # Output
        cmd.append(output_path)
        
        return cmd
    
    def _build_filter_complex(self, sequence: Dict[str, Any]) -> str:
        """Build FFmpeg filter complex string"""
        filters = []
        
        # Process clips
        clips = sequence.get("clips", [])
        for i, clip in enumerate(clips):
            clip_filters = []
            
            # Scale/crop if needed
            if "scale" in clip:
                scale = clip["scale"]
                clip_filters.append(
                    f"scale={scale['x']}:{scale['y']}"
                )
            
            # Effects
            for effect in clip.get("effects", []):
                effect_filter = self._effect_to_filter(effect, i)
                if effect_filter:
                    clip_filters.append(effect_filter)
            
            # Apply filters to input
            if clip_filters:
                filter_str = f"[{i}:v]" + ",".join(clip_filters) + f"[v{i}]"
                filters.append(filter_str)
            else:
                filters.append(f"[{i}:v]copy[v{i}]")
        
        # Text overlays
        for i, text in enumerate(sequence.get("texts", [])):
            text_filter = self._create_text_filter(text, i)
            filters.append(text_filter)
        
        # Concatenate clips
        if len(clips) > 1:
            inputs = "".join(f"[v{i}]" for i in range(len(clips)))
            concat = f"{inputs}concat=n={len(clips)}:v=1:a=0[base]"
            filters.append(concat)
            final_output = "base"
        else:
            final_output = "v0"
        
        # Apply text overlays
        for i, text in enumerate(sequence.get("texts", [])):
            overlay = f"[{final_output}][text{i}]overlay={text.get('x', 0)}:{text.get('y', 0)}"
            if "enable" in text:
                overlay += f":enable='{text['enable']}'"
            overlay += f"[out{i}]"
            filters.append(overlay)
            final_output = f"out{i}"
        
        # Final output
        if final_output != "base":
            filters.append(f"[{final_output}]copy[out]")
        else:
            filters.append(f"[{final_output}]copy[out]")
        
        return ";".join(filters)
    
    def _effect_to_filter(self, effect: Dict[str, Any], input_idx: int) -> str:
        """Convert effect description to FFmpeg filter"""
        effect_type = effect["type"]
        params = effect.get("params", {})
        
        if effect_type == "zoom":
            # Zoom effect using scale
            scale_from = params.get("from", 1.0)
            scale_to = params.get("to", 1.1)
            duration = params.get("duration", 1.0)
            
            return (f"scale=w=iw*{scale_from}+iw*({scale_to}-{scale_from})*t/"
                   f"{duration}:h=ih*{scale_from}+ih*({scale_to}-{scale_from})*t/"
                   f"{duration}")
        
        elif effect_type == "fade":
            # Fade in/out
            fade_type = params.get("direction", "in")
            duration = params.get("duration", 1.0)
            start = params.get("start", 0)
            
            return f"fade={fade_type}:d={duration}:st={start}"
        
        elif effect_type == "blur":
            # Gaussian blur
            amount = params.get("amount", 5)
            return f"gblur=sigma={amount}"
        
        elif effect_type == "shake":
            # Camera shake
            amplitude = params.get("amplitude", 5)
            frequency = params.get("frequency", 10)
            
            return (f"crop=in_w:in_h:("
                   f"{amplitude}*sin({frequency}*2*PI*t)):("
                   f"{amplitude}*cos({frequency}*2*PI*t))")
        
        elif effect_type == "speed":
            # Speed change
            speed = params.get("speed", 1.0)
            return f"setpts={1/speed}*PTS"
        
        elif effect_type == "color":
            # Color correction
            brightness = params.get("brightness", 0)
            contrast = params.get("contrast", 1)
            saturation = params.get("saturation", 1)
            
            return f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"
        
        return ""
    
    def _create_text_filter(self, text: Dict[str, Any], index: int) -> str:
        """Create text overlay filter"""
        content = text["content"]
        font = text.get("font", "Arial")
        size = text.get("size", 48)
        color = text.get("color", "white")
        
        # Escape special characters
        content = content.replace(":", "\\:")
        
        # Build drawtext filter
        drawtext = (f"drawtext=text='{content}':"
                   f"fontfile={font}:fontsize={size}:"
                   f"fontcolor={color}")
        
        # Position
        x = text.get("x", "(w-text_w)/2")  # Center by default
        y = text.get("y", "(h-text_h)/2")
        drawtext += f":x={x}:y={y}"
        
        # Box background
        if text.get("box", False):
            drawtext += ":box=1:boxcolor=black@0.5:boxborderw=5"
        
        # Animation
        if "animate" in text:
            anim = text["animate"]
            if anim["type"] == "fade":
                drawtext += f":alpha='if(lt(t,{anim['start']}),0,if(lt(t,{anim['start']}+{anim['duration']}),(t-{anim['start']})/{anim['duration']},1))'"
        
        return f"color=c=black:s=1080x1920:d={text.get('duration', 1)}[text{index}];[text{index}]{drawtext}[text{index}]"
    
    def _detect_hardware(self) -> bool:
        """Detect hardware acceleration availability"""
        try:
            # Check for VideoToolbox on macOS
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True
            )
            return "videotoolbox" in result.stdout
        except:
            return False
    
    def create_viral_edit(self, video_path: str, analysis_data: Dict[str, Any], 
                         output_path: str, style: str = "viral") -> bool:
        """
        Create a viral edit based on analysis data
        
        Args:
            video_path: Source video
            analysis_data: Frame-by-frame analysis
            output_path: Output path
            style: Edit style
            
        Returns:
            Success status
        """
        # Build sequence from analysis
        sequence = self._build_viral_sequence(video_path, analysis_data, style)
        
        # Render
        return self.render_sequence(sequence, output_path)
    
    def _build_viral_sequence(self, video_path: str, 
                            analysis_data: Dict[str, Any], 
                            style: str) -> Dict[str, Any]:
        """Build video sequence for viral edit"""
        frames = analysis_data.get("frame_data", [])
        duration = analysis_data.get("metadata", {}).get("duration", 60)
        
        sequence = {
            "duration": min(duration, 60),  # Max 60 seconds
            "fps": 30,
            "resolution": [1080, 1920],
            "inputs": [],
            "clips": [],
            "texts": [],
            "audio": True
        }
        
        # Find key moments
        emotional_peaks = []
        low_energy_sections = []
        
        for i, frame in enumerate(frames):
            emotion_score = frame.get("emotional_tone", {}).get("confidence", 0)
            if emotion_score > 0.8:
                emotional_peaks.append(frame)
            
            if i > 0 and frame.get("cinematography_score", 0) < 0.3:
                if not low_energy_sections or frame["timestamp"] - low_energy_sections[-1][-1]["timestamp"] > 1:
                    low_energy_sections.append([frame])
                else:
                    low_energy_sections[-1].append(frame)
        
        # Build main clip with effects
        main_clip = {
            "path": video_path,
            "effects": []
        }
        
        # Add zoom effects on emotional peaks
        for peak in emotional_peaks[:5]:  # Top 5 peaks
            main_clip["effects"].append({
                "type": "zoom",
                "params": {
                    "from": 1.0,
                    "to": 1.1,
                    "duration": 0.5,
                    "at": peak["timestamp"]
                }
            })
        
        # Speed up low energy sections
        for section in low_energy_sections:
            if len(section) > 5:  # More than 5 frames
                start = section[0]["timestamp"]
                end = section[-1]["timestamp"]
                main_clip["effects"].append({
                    "type": "speed",
                    "params": {
                        "speed": 1.5,
                        "start": start,
                        "end": end
                    }
                })
        
        sequence["inputs"].append({"path": video_path})
        sequence["clips"].append(main_clip)
        
        # Add text overlays
        if style == "viral":
            # Hook text
            sequence["texts"].append({
                "content": "WAIT FOR IT...",
                "x": "(w-text_w)/2",
                "y": "h/2",
                "size": 72,
                "color": "yellow",
                "font": "/System/Library/Fonts/Helvetica.ttc",
                "duration": 3,
                "enable": "between(t,0.5,3)",
                "animate": {
                    "type": "fade",
                    "start": 0.5,
                    "duration": 0.5
                }
            })
            
            # CTA text
            sequence["texts"].append({
                "content": "TAP THE LINK NOW",
                "x": "(w-text_w)/2",
                "y": "h*0.7",
                "size": 60,
                "color": "white",
                "font": "/System/Library/Fonts/Helvetica.ttc",
                "duration": 3,
                "enable": f"between(t,{duration-3},{duration})",
                "box": True
            })
        
        return sequence
    
    def export_for_platform(self, input_path: str, platform: str) -> str:
        """Export video optimized for specific platform"""
        output_path = input_path.replace(".mp4", f"_{platform}.mp4")
        
        platform_settings = {
            "tiktok": {
                "resolution": "1080x1920",
                "bitrate": "8M",
                "fps": 30,
                "filters": "eq=saturation=1.1:contrast=1.05"
            },
            "instagram": {
                "resolution": "1080x1920",
                "bitrate": "8M",
                "fps": 30,
                "filters": "eq=saturation=0.95:contrast=0.98"
            },
            "youtube": {
                "resolution": "1080x1920",
                "bitrate": "10M",
                "fps": 30,
                "filters": None
            }
        }
        
        settings = platform_settings.get(platform, platform_settings["tiktok"])
        
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", f"scale={settings['resolution']}",
            "-b:v", settings["bitrate"],
            "-r", str(settings["fps"]),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k"
        ]
        
        if settings["filters"]:
            cmd[4] += f",{settings['filters']}"
        
        cmd.extend(["-movflags", "+faststart", "-y", output_path])
        
        subprocess.run(cmd, check=True)
        return output_path


# Example usage
if __name__ == "__main__":
    renderer = FFmpegRenderer()
    
    # Example sequence
    sequence = {
        "duration": 10,
        "fps": 30,
        "resolution": [1080, 1920],
        "inputs": [{"path": "video.mp4"}],
        "clips": [{
            "effects": [
                {"type": "zoom", "params": {"from": 1.0, "to": 1.2, "duration": 2}}
            ]
        }],
        "texts": [{
            "content": "Hello World",
            "x": "(w-text_w)/2",
            "y": "h/2",
            "size": 48,
            "duration": 5
        }]
    }
    
    renderer.render_sequence(sequence, "output.mp4")