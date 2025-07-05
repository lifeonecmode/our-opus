#!/usr/bin/env python3
"""
Video State Editor - React-style state management for video editing
Edit videos programmatically using state and components
"""

import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import timedelta
import asyncio
import logging

from .ffmpeg_renderer import FFmpegRenderer

logger = logging.getLogger(__name__)


@dataclass
class VideoState:
    """Global video editor state"""
    clips: List[Dict[str, Any]] = field(default_factory=list)
    effects: List[Dict[str, Any]] = field(default_factory=list)
    texts: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0
    fps: int = 30
    resolution: tuple = (1080, 1920)
    current_time: float = 0


class VideoStateEditor:
    """
    State-based video editor that works like React
    Manages video editing through state changes
    """
    
    def __init__(self):
        self.state = VideoState()
        self.renderer = FFmpegRenderer()
        self.subscribers: List[Callable] = []
        
    def setState(self, updates: Dict[str, Any]):
        """Update state and trigger re-render"""
        for key, value in updates.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        # Notify subscribers
        self._notify_subscribers()
    
    def useState(self, initial_value: Any) -> tuple:
        """Hook for managing component state"""
        value = initial_value
        
        def set_value(new_value):
            nonlocal value
            value = new_value
            self._notify_subscribers()
        
        return value, set_value
    
    def useEffect(self, effect: Callable, dependencies: List[Any]):
        """Hook for side effects"""
        # Track dependencies and run effect when they change
        effect()
    
    def addClip(self, src: str, start: float = 0, duration: Optional[float] = None, 
                trim: Optional[Dict[str, float]] = None):
        """Add video clip to timeline"""
        clip = {
            "src": src,
            "start": start,
            "duration": duration,
            "trim": trim,
            "effects": [],
            "id": f"clip_{len(self.state.clips)}"
        }
        
        self.setState({
            "clips": self.state.clips + [clip]
        })
        
        return clip["id"]
    
    def addEffect(self, clip_id: str, effect_type: str, params: Dict[str, Any], 
                  timestamp: Optional[float] = None):
        """Add effect to clip"""
        effect = {
            "type": effect_type,
            "params": params,
            "timestamp": timestamp or self.state.current_time,
            "id": f"effect_{len(self.state.effects)}"
        }
        
        # Find clip and add effect
        for clip in self.state.clips:
            if clip["id"] == clip_id:
                clip["effects"].append(effect)
                break
        
        self.setState({
            "effects": self.state.effects + [effect]
        })
        
        return effect["id"]
    
    def addText(self, text: str, style: Dict[str, Any], 
                start: float, duration: float,
                animate: Optional[Dict[str, Any]] = None):
        """Add text overlay"""
        text_obj = {
            "content": text,
            "style": style,
            "start": start,
            "duration": duration,
            "animate": animate,
            "id": f"text_{len(self.state.texts)}"
        }
        
        self.setState({
            "texts": self.state.texts + [text_obj]
        })
        
        return text_obj["id"]
    
    def addTransition(self, from_clip: str, to_clip: str, 
                     transition_type: str, duration: float = 1.0):
        """Add transition between clips"""
        transition = {
            "from": from_clip,
            "to": to_clip,
            "type": transition_type,
            "duration": duration,
            "id": f"transition_{len(self.state.transitions)}"
        }
        
        self.setState({
            "transitions": self.state.transitions + [transition]
        })
        
        return transition["id"]
    
    def applyAnalysis(self, analysis_data: Dict[str, Any], style: str = "viral"):
        """Apply frame analysis to create automatic edits"""
        frames = analysis_data.get("frame_data", [])
        
        if style == "viral":
            self._apply_viral_edits(frames, analysis_data)
        elif style == "cinematic":
            self._apply_cinematic_edits(frames, analysis_data)
        elif style == "fast":
            self._apply_fast_edits(frames, analysis_data)
    
    def _apply_viral_edits(self, frames: List[Dict], analysis_data: Dict):
        """Apply viral-style edits based on analysis"""
        # Find emotional peaks
        emotional_peaks = [f for f in frames if f.get("emotional_tone", {}).get("confidence", 0) > 0.8]
        
        # Add zoom effects on peaks
        for peak in emotional_peaks[:5]:
            # Find which clip contains this timestamp
            for clip in self.state.clips:
                if self._timestamp_in_clip(peak["timestamp"], clip):
                    self.addEffect(
                        clip["id"],
                        "zoom",
                        {
                            "scale": {"from": 100, "to": 110},
                            "duration": 0.5,
                            "ease": "easeInOut"
                        },
                        peak["timestamp"]
                    )
        
        # Add viral text overlays
        self.addText(
            "WAIT FOR IT...",
            {
                "fontSize": 72,
                "color": "#FFD700",
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
        
        # Add CTA at end
        video_duration = self.state.duration or 60
        self.addText(
            "TAP THE LINK NOW",
            {
                "fontSize": 60,
                "color": "#FFFFFF",
                "position": {"x": 540, "y": 1400},
                "font": "Arial-Bold",
                "box": True
            },
            start=video_duration - 3,
            duration=3,
            animate={
                "scale": [0, 1.3, 1],
                "shake": {"x": 5, "y": 5},
                "duration": 0.3
            }
        )
    
    def _timestamp_in_clip(self, timestamp: float, clip: Dict) -> bool:
        """Check if timestamp falls within clip"""
        clip_start = clip["start"]
        clip_duration = clip.get("duration", float('inf'))
        return clip_start <= timestamp < clip_start + clip_duration
    
    async def render(self, output_path: str) -> bool:
        """Render the current state to video file"""
        # Convert state to FFmpeg sequence
        sequence = self._state_to_sequence()
        
        # Render using FFmpeg
        logger.info(f"Rendering video to: {output_path}")
        success = self.renderer.render_sequence(sequence, output_path)
        
        if success:
            logger.info("Render complete!")
            
            # Platform-specific exports
            if output_path.endswith(".mp4"):
                base_path = output_path[:-4]
                await self._export_platforms(base_path)
        
        return success
    
    def _state_to_sequence(self) -> Dict[str, Any]:
        """Convert current state to FFmpeg sequence"""
        sequence = {
            "duration": self.state.duration,
            "fps": self.state.fps,
            "resolution": list(self.state.resolution),
            "inputs": [],
            "clips": [],
            "texts": [],
            "audio": True
        }
        
        # Process clips
        for clip in self.state.clips:
            input_obj = {"path": clip["src"]}
            if clip.get("trim"):
                input_obj["trim"] = clip["trim"]
            
            sequence["inputs"].append(input_obj)
            
            # Convert clip with effects
            clip_obj = {
                "effects": []
            }
            
            for effect in clip.get("effects", []):
                clip_obj["effects"].append({
                    "type": effect["type"],
                    "params": effect["params"]
                })
            
            sequence["clips"].append(clip_obj)
        
        # Process texts
        for text in self.state.texts:
            text_obj = {
                "content": text["content"],
                "x": text["style"].get("position", {}).get("x", "(w-text_w)/2"),
                "y": text["style"].get("position", {}).get("y", "(h-text_h)/2"),
                "size": text["style"].get("fontSize", 48),
                "color": text["style"].get("color", "white"),
                "font": self._get_font_path(text["style"].get("font", "Arial")),
                "duration": text["duration"],
                "enable": f"between(t,{text['start']},{text['start'] + text['duration']})"
            }
            
            if text["style"].get("box"):
                text_obj["box"] = True
            
            if text.get("animate"):
                text_obj["animate"] = text["animate"]
            
            sequence["texts"].append(text_obj)
        
        return sequence
    
    def _get_font_path(self, font_name: str) -> str:
        """Get system font path"""
        # macOS font paths
        font_paths = {
            "Arial": "/System/Library/Fonts/Helvetica.ttc",
            "Arial-Bold": "/System/Library/Fonts/Helvetica.ttc",
            "Helvetica": "/System/Library/Fonts/Helvetica.ttc",
            "Times": "/System/Library/Fonts/Times.ttc"
        }
        
        return font_paths.get(font_name, "/System/Library/Fonts/Helvetica.ttc")
    
    async def _export_platforms(self, base_path: str):
        """Export platform-specific versions"""
        platforms = ["tiktok", "instagram", "youtube"]
        
        for platform in platforms:
            output = f"{base_path}_{platform}.mp4"
            logger.info(f"Exporting for {platform}: {output}")
            self.renderer.export_for_platform(f"{base_path}.mp4", platform)
    
    def _notify_subscribers(self):
        """Notify all subscribers of state change"""
        for subscriber in self.subscribers:
            subscriber(self.state)
    
    def subscribe(self, callback: Callable):
        """Subscribe to state changes"""
        self.subscribers.append(callback)
        
    def createViralVideo(self, video_path: str, analysis_data: Dict[str, Any]) -> 'VideoStateEditor':
        """Create a viral video from analysis data"""
        # Reset state
        self.state = VideoState()
        
        # Add main video clip
        self.addClip(video_path, start=0)
        
        # Apply viral edits based on analysis
        self.applyAnalysis(analysis_data, style="viral")
        
        # Set duration
        self.setState({
            "duration": min(analysis_data.get("metadata", {}).get("duration", 60), 60)
        })
        
        return self


# Declarative API wrapper
class Video:
    """Declarative video editing API"""
    
    @staticmethod
    def sequence(duration: float, fps: int = 30, children: List[Any] = None):
        """Create video sequence"""
        editor = VideoStateEditor()
        editor.setState({
            "duration": duration,
            "fps": fps
        })
        
        if children:
            for child in children:
                child(editor)
        
        return editor
    
    @staticmethod
    def clip(src: str, start: float = 0, duration: float = None, 
             trim: Dict[str, float] = None):
        """Create video clip"""
        def add_clip(editor: VideoStateEditor):
            return editor.addClip(src, start, duration, trim)
        return add_clip
    
    @staticmethod
    def text(content: str, style: Dict[str, Any], start: float, 
             duration: float, animate: Dict[str, Any] = None):
        """Create text overlay"""
        def add_text(editor: VideoStateEditor):
            return editor.addText(content, style, start, duration, animate)
        return add_text
    
    @staticmethod
    def effect(clip_id: str, effect_type: str, params: Dict[str, Any]):
        """Create effect"""
        def add_effect(editor: VideoStateEditor):
            return editor.addEffect(clip_id, effect_type, params)
        return add_effect


# Example: Create viral video with declarative API
async def create_viral_edit_example():
    """Example of creating a viral edit"""
    
    # Load analysis data
    with open("analysis.json", "r") as f:
        analysis = json.load(f)
    
    # Create editor with declarative API
    editor = Video.sequence(
        duration=60,
        fps=30,
        children=[
            Video.clip("input.mp4", start=0),
            Video.text(
                "WAIT FOR IT...",
                style={
                    "fontSize": 72,
                    "color": "#FFD700",
                    "position": {"x": 540, "y": 960}
                },
                start=0.5,
                duration=2.5,
                animate={
                    "scale": [0, 1.2, 1],
                    "opacity": [0, 1]
                }
            ),
            Video.text(
                "TAP THE LINK NOW",
                style={
                    "fontSize": 60,
                    "color": "#FFFFFF",
                    "position": {"x": 540, "y": 1400},
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
    
    # Apply analysis-based edits
    editor.applyAnalysis(analysis, style="viral")
    
    # Render
    await editor.render("output_viral.mp4")


if __name__ == "__main__":
    # Run example
    asyncio.run(create_viral_edit_example())