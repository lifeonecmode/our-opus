"""
Claude Video Editor - AI-powered video editing with Claude Code integration
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .video_ai_editor import VideoEditor
from .core.claude_orchestrator import ClaudeDecisionEngine
from .core.nuclear_frame_extractor import NuclearFrameExtractor
from .core.cinematic_scene_analyzer import CinematicSceneAnalyzer
from .whisper import transcribe, load_model

__all__ = [
    "VideoEditor",
    "ClaudeDecisionEngine", 
    "NuclearFrameExtractor",
    "CinematicSceneAnalyzer",
    "transcribe",
    "load_model",
]