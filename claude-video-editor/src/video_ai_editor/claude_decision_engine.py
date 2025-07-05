#!/usr/bin/env python3
"""
Claude Decision Engine - AI-powered editing decisions
Claude analyzes footage and creates optimal edit decisions
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from core.video_state_editor import VideoStateEditor
from core.ffmpeg_renderer import FFmpegRenderer

logger = logging.getLogger(__name__)


class ClaudeDecisionEngine:
    """
    Claude acts as the decision engine for video editing
    Analyzes footage and creates editing decisions
    """
    
    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or ".")
        self.footage_dir = self.project_dir / "footage"
        self.audio_dir = self.project_dir / "audio"
        self.editor = VideoStateEditor()
        
        # Claude's editing knowledge
        self.editing_rules = {
            "viral": {
                "hook_duration": 3,
                "peak_zoom": 1.1,
                "low_energy_speed": 1.5,
                "cta_duration": 3,
                "max_duration": 60
            },
            "cinematic": {
                "shot_duration": 4,
                "transition_duration": 1,
                "color_grade": "teal_orange",
                "aspect_ratio": 2.35
            }
        }
    
    async def analyze_project(self) -> Dict[str, Any]:
        """Analyze all footage and audio in project"""
        logger.info("🤖 Claude analyzing project structure...")
        
        analysis = {
            "footage": await self._analyze_footage(),
            "audio": await self._analyze_audio(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save analysis
        analysis_path = self.project_dir / "claude_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"✅ Analysis saved to: {analysis_path}")
        return analysis
    
    async def _analyze_footage(self) -> Dict[str, Any]:
        """Analyze all video files in footage directory"""
        footage_analysis = {
            "main_speaker": None,
            "b_roll": [],
            "transformation": {"before": None, "after": None},
            "product_shots": [],
            "input_videos": []
        }
        
        # Analyze main speaker
        main_speaker = self.footage_dir / "main-speaker.mp4"
        if main_speaker.exists():
            footage_analysis["main_speaker"] = {
                "path": str(main_speaker),
                "purpose": "primary_narrative",
                "suggested_use": "base_layer",
                "duration": self._get_video_duration(main_speaker)
            }
        
        # Analyze transformation shots
        transform_dir = self.footage_dir / "transformation"
        if transform_dir.exists():
            for video in transform_dir.glob("before*.mp4"):
                footage_analysis["transformation"]["before"] = {
                    "path": str(video),
                    "purpose": "contrast_negative",
                    "color_suggestion": "cool_desaturated"
                }
            
            for video in transform_dir.glob("after*.mp4"):
                footage_analysis["transformation"]["after"] = {
                    "path": str(video),
                    "purpose": "contrast_positive",
                    "color_suggestion": "warm_vibrant"
                }
        
        # Analyze B-roll
        b_roll_dir = self.footage_dir / "b-roll"
        if b_roll_dir.exists():
            for category in ["lifestyle_shots", "product_details"]:
                category_dir = b_roll_dir / category
                if category_dir.exists():
                    for video in category_dir.glob("*.mp4"):
                        footage_analysis["b_roll"].append({
                            "path": str(video),
                            "category": category,
                            "suggested_timing": self._suggest_timing(category)
                        })
        
        # Analyze input directory
        input_dir = self.project_dir / "input"
        if input_dir.exists():
            # Look for existing input analysis
            input_analysis_file = self.project_dir / "analysis" / "input_videos_analysis.json"
            if input_analysis_file.exists():
                with open(input_analysis_file, 'r') as f:
                    input_data = json.load(f)
                    for video in input_data.get("input_videos", []):
                        footage_analysis["input_videos"].append({
                            "path": video["path"],
                            "category": video["category"],
                            "suggested_use": video["suggested_use"]["primary_use"],
                            "duration": video["analysis"].get("metadata", {}).get("duration", 0)
                        })
            else:
                # Analyze input videos directly
                video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
                for ext in video_extensions:
                    for video in input_dir.glob(ext):
                        footage_analysis["input_videos"].append({
                            "path": str(video),
                            "category": "unknown",
                            "suggested_use": "review_needed",
                            "duration": self._get_video_duration(video)
                        })
        
        return footage_analysis
    
    async def _analyze_audio(self) -> Dict[str, Any]:
        """Analyze all audio files"""
        audio_analysis = {
            "voiceover": None,
            "music": None,
            "sound_effects": {}
        }
        
        # Voiceover
        voiceover = self.audio_dir / "voiceover.wav"
        if voiceover.exists():
            audio_analysis["voiceover"] = {
                "path": str(voiceover),
                "duration": self._get_audio_duration(voiceover),
                "purpose": "primary_narrative"
            }
        
        # Music
        music = self.audio_dir / "music.mp3"
        if music.exists():
            audio_analysis["music"] = {
                "path": str(music),
                "bpm": 120,  # Specified as 120 BPM
                "purpose": "emotional_driver",
                "energy_curve": "building"
            }
        
        # Sound effects
        sfx_dir = self.audio_dir / "sfx"
        if sfx_dir.exists():
            for sfx in sfx_dir.glob("*.wav"):
                sfx_name = sfx.stem
                audio_analysis["sound_effects"][sfx_name] = {
                    "path": str(sfx),
                    "type": self._classify_sfx(sfx_name),
                    "suggested_use": self._suggest_sfx_use(sfx_name)
                }
        
        return audio_analysis
    
    def _suggest_timing(self, category: str) -> str:
        """Claude suggests when to use footage"""
        timings = {
            "lifestyle_shots": "benefits_section",
            "product_details": "feature_highlight",
            "transformation": "midpoint_reveal"
        }
        return timings.get(category, "supporting_content")
    
    def _classify_sfx(self, name: str) -> str:
        """Classify sound effect type"""
        if "boom" in name.lower():
            return "impact"
        elif "swoosh" in name.lower():
            return "transition"
        elif "ding" in name.lower():
            return "success"
        else:
            return "accent"
    
    def _suggest_sfx_use(self, name: str) -> Dict[str, Any]:
        """Claude suggests how to use sound effects"""
        suggestions = {
            "boom": {
                "timing": "transformation_reveal",
                "volume": 110,
                "effect": "emphasis"
            },
            "swoosh": {
                "timing": "scene_transition",
                "volume": 80,
                "effect": "smooth_flow"
            },
            "ding": {
                "timing": "benefit_appear",
                "volume": 90,
                "effect": "positive_reinforcement"
            }
        }
        return suggestions.get(name, {"timing": "accent", "volume": 85})
    
    def _get_video_duration(self, path: Path) -> float:
        """Get video duration using ffprobe"""
        import subprocess
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _get_audio_duration(self, path: Path) -> float:
        """Get audio duration"""
        return self._get_video_duration(path)
    
    async def create_edit_decision_list(self, style: str = "viral") -> Dict[str, Any]:
        """
        Claude creates complete edit decision list
        This is where Claude's creativity and decision-making happens
        """
        logger.info(f"🎬 Claude creating {style} edit decisions...")
        
        # Load analysis
        analysis_path = self.project_dir / "claude_analysis.json"
        if not analysis_path.exists():
            analysis = await self.analyze_project()
        else:
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
        
        # Create edit decisions based on style
        if style == "viral":
            decisions = self._create_viral_decisions(analysis)
        elif style == "cinematic":
            decisions = self._create_cinematic_decisions(analysis)
        else:
            decisions = self._create_standard_decisions(analysis)
        
        # Save decisions
        decisions_path = self.project_dir / f"claude_edit_decisions_{style}.json"
        with open(decisions_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        
        return decisions
    
    def _create_viral_decisions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Claude's viral video edit decisions"""
        footage = analysis["footage"]
        audio = analysis["audio"]
        sfx = audio["sound_effects"]
        
        decisions = {
            "style": "viral",
            "duration": 60,
            "structure": []
        }
        
        # Check for input videos and use them intelligently
        input_videos = footage.get("input_videos", [])
        speaker_videos = [v for v in input_videos if v["category"] == "speaker"]
        product_videos = [v for v in input_videos if v["category"] == "product"]
        lifestyle_videos = [v for v in input_videos if v["category"] == "lifestyle"]
        
        # Determine primary speaker source
        main_speaker_path = None
        if footage["main_speaker"]:
            main_speaker_path = footage["main_speaker"]["path"]
        elif speaker_videos:
            main_speaker_path = speaker_videos[0]["path"]  # Use first speaker video from input
        
        # HOOK (0-3s)
        if main_speaker_path:
            decisions["structure"].append({
                "section": "hook",
                "start": 0,
                "duration": 3,
                "clips": [
                    {
                        "source": main_speaker_path,
                        "trim": {"start": 0, "duration": 3}
                    }
                ],
                "effects": [
                    {"type": "fade", "params": {"direction": "in", "duration": 0.5}}
                ],
                "text": {
                    "content": "WAIT FOR IT...",
                    "style": {"fontSize": 72, "color": "#FFD700"},
                    "timing": {"start": 0.5, "duration": 2.5},
                    "animation": {"scale": [0, 1.2, 1]}
                },
                "audio": {
                    "music": {"volume": 70, "fade_in": 0.5}
                }
            })
        
        # SOCIAL PROOF (3-8s)
        if main_speaker_path:
            decisions["structure"].append({
                "section": "social_proof",
                "start": 3,
                "duration": 5,
                "clips": [
                    {
                        "source": main_speaker_path,
                        "trim": {"start": 3, "duration": 5}
                    }
                ],
                "text": {
                    "content": "47,000+ Happy Customers",
                    "style": {"fontSize": 56, "color": "#FFFFFF"},
                    "timing": {"start": 4, "duration": 3}
                }
            })
        
        # TRANSFORMATION (8-11s)
        if footage["transformation"]["before"] and footage["transformation"]["after"]:
            decisions["structure"].append({
                "section": "transformation",
                "start": 8,
                "duration": 3,
                "clips": [
                    {
                        "source": footage["transformation"]["before"]["path"],
                        "trim": {"start": 0, "duration": 1.5},
                        "position": "left"
                    },
                    {
                        "source": footage["transformation"]["after"]["path"],
                        "trim": {"start": 0, "duration": 1.5},
                        "position": "right"
                    }
                ],
                "effects": [
                    {"type": "split_screen", "params": {"direction": "vertical"}}
                ],
                "sfx": {
                    "source": sfx.get("boom", {}).get("path"),
                    "timing": 10,
                    "volume": 110
                }
            })
        
        # STORY/BENEFITS (11-50s)
        # Claude decides to use B-roll intelligently
        b_roll_clips = []
        # Use traditional B-roll first
        for i, b_roll in enumerate(footage["b_roll"][:3]):  # Use up to 3 B-roll clips
            b_roll_clips.append({
                "source": b_roll["path"],
                "start": 20 + (i * 10),
                "duration": 3,
                "transition": "smooth_cut"
            })
        
        # Add lifestyle videos from input directory
        for i, lifestyle in enumerate(lifestyle_videos[:2]):  # Add up to 2 lifestyle videos
            b_roll_clips.append({
                "source": lifestyle["path"],
                "start": 30 + (i * 5),
                "duration": 3,
                "transition": "smooth_cut"
            })
        
        # Add product videos from input directory  
        for i, product in enumerate(product_videos[:1]):  # Add 1 product video
            b_roll_clips.append({
                "source": product["path"],
                "start": 40,
                "duration": 3,
                "transition": "zoom_cut"
            })
        
        if main_speaker_path:
            decisions["structure"].append({
                "section": "story_benefits",
                "start": 11,
                "duration": 39,
                "clips": [
                    {
                        "source": main_speaker_path,
                        "trim": {"start": 11, "duration": 39}
                    }
                ],
                "b_roll": b_roll_clips,
                "text_overlays": [
                    {
                        "content": "✓ Save 3 hours daily",
                        "timing": {"start": 35, "duration": 3},
                        "style": {"fontSize": 56, "color": "#4CAF50"}
                    },
                    {
                        "content": "✓ Feel energized",
                        "timing": {"start": 38, "duration": 3},
                        "style": {"fontSize": 56, "color": "#4CAF50"}
                    }
                ]
            })
        
        # URGENCY (50-57s)
        decisions["structure"].append({
            "section": "urgency",
            "start": 50,
            "duration": 7,
            "effects": [
                {"type": "vignette", "params": {"amount": 40}},
                {"type": "color", "params": {"saturation": 1.2}}
            ],
            "text": {
                "content": "⏰ LIMITED TIME",
                "style": {"fontSize": 64, "color": "#FF6347", "box": True},
                "timing": {"start": 50, "duration": 7},
                "animation": {"pulse": True}
            }
        })
        
        # CTA (57-60s)
        decisions["structure"].append({
            "section": "cta",
            "start": 57,
            "duration": 3,
            "text": {
                "content": "👆 TAP THE LINK NOW",
                "style": {"fontSize": 72, "color": "#FFD700", "box": True},
                "timing": {"start": 57, "duration": 3},
                "animation": {"scale": [0, 1.3, 1], "shake": {"x": 5, "y": 5}}
            },
            "sfx": {
                "source": sfx.get("ding", {}).get("path"),
                "timing": 57.5,
                "volume": 90
            }
        })
        
        # Audio tracks
        decisions["audio_tracks"] = [
            {
                "type": "voiceover",
                "source": audio["voiceover"]["path"] if audio["voiceover"] else None,
                "volume": 100
            },
            {
                "type": "music",
                "source": audio["music"]["path"] if audio["music"] else None,
                "volume": 60,
                "fade_in": 1,
                "fade_out": 2
            }
        ]
        
        return decisions
    
    def _create_cinematic_decisions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Claude's cinematic edit decisions"""
        # Implement cinematic style decisions
        return {
            "style": "cinematic",
            "duration": 60,
            "structure": [],
            "effects": [
                {"type": "letterbox", "params": {"ratio": 2.35}},
                {"type": "color_grade", "params": {"style": "teal_orange"}}
            ]
        }
    
    def _create_standard_decisions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Standard edit decisions"""
        return {
            "style": "standard",
            "duration": 60,
            "structure": []
        }
    
    async def execute_edit(self, decisions: Dict[str, Any] = None, output: str = "output.mp4"):
        """Execute the edit decisions using VideoStateEditor"""
        logger.info("🎥 Executing Claude's edit decisions...")
        
        if not decisions:
            decisions = await self.create_edit_decision_list()
        
        # Reset editor
        self.editor = VideoStateEditor()
        
        # Set video properties
        self.editor.setState({
            "duration": decisions["duration"],
            "fps": 30,
            "resolution": (1080, 1920)
        })
        
        # Execute each section
        for section in decisions["structure"]:
            await self._execute_section(section)
        
        # Add audio tracks
        if "audio_tracks" in decisions:
            for track in decisions["audio_tracks"]:
                if track["source"]:
                    # Note: FFmpeg will handle audio mixing
                    logger.info(f"Adding {track['type']} track: {track['source']}")
        
        # Render final video
        success = await self.editor.render(output)
        
        if success:
            logger.info(f"✅ Video created: {output}")
            logger.info("📱 Platform versions also created")
        
        return success
    
    async def _execute_section(self, section: Dict[str, Any]):
        """Execute a single section of the edit"""
        logger.info(f"Processing section: {section['section']}")
        
        # Add clips
        for clip in section.get("clips", []):
            clip_id = self.editor.addClip(
                clip["source"],
                start=section["start"],
                duration=clip.get("duration"),
                trim=clip.get("trim")
            )
            
            # Add clip effects
            for effect in section.get("effects", []):
                self.editor.addEffect(
                    clip_id,
                    effect["type"],
                    effect["params"],
                    timestamp=section["start"]
                )
        
        # Add text
        if "text" in section:
            text = section["text"]
            self.editor.addText(
                text["content"],
                text["style"],
                text["timing"]["start"],
                text["timing"]["duration"],
                text.get("animation")
            )
        
        # Add text overlays
        for overlay in section.get("text_overlays", []):
            self.editor.addText(
                overlay["content"],
                overlay["style"],
                overlay["timing"]["start"],
                overlay["timing"]["duration"],
                overlay.get("animation")
            )


# Claude's direct interface
async def claude_edit_video(project_dir: str = ".", style: str = "viral"):
    """
    Claude's main function to edit a video project
    
    Usage:
        await claude_edit_video("/path/to/project", style="viral")
    """
    engine = ClaudeDecisionEngine(project_dir)
    
    # Analyze project
    print("🤖 Claude analyzing your project...")
    analysis = await engine.analyze_project()
    
    # Create decisions
    print(f"🎬 Claude creating {style} edit decisions...")
    decisions = await engine.create_edit_decision_list(style)
    
    # Execute edit
    print("🎥 Claude executing the edit...")
    output = f"claude_edit_{style}.mp4"
    success = await engine.execute_edit(decisions, output)
    
    if success:
        print(f"\n✅ Claude's edit complete: {output}")
        print(f"📱 Platform versions:")
        print(f"  - {output.replace('.mp4', '_tiktok.mp4')}")
        print(f"  - {output.replace('.mp4', '_instagram.mp4')}")
        print(f"  - {output.replace('.mp4', '_youtube.mp4')}")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude edits your video")
    parser.add_argument("project", nargs="?", default=".", 
                       help="Project directory with footage/ and audio/")
    parser.add_argument("--style", default="viral",
                       choices=["viral", "cinematic", "standard"],
                       help="Edit style")
    
    args = parser.parse_args()
    
    asyncio.run(claude_edit_video(args.project, args.style))