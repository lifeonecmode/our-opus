#!/usr/bin/env python3
"""
Claude Orchestrator - Hybrid approach for maximum reliability
This runs locally but can be controlled by Claude Code
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List, Any, Optional

# Add imports
sys.path.append(str(Path(__file__).parent))
from nuclear_frame_extractor import NuclearFrameExtractor
from video_ai_editor.analyze_all_footage import ProjectAnalyzer
from video_ai_editor.claude_decision_engine import ClaudeDecisionEngine
from cinematic_scene_analyzer import CinematicSceneAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClaudeOrchestrator:
    """
    Local orchestrator that Claude Code can control
    This gives us the best of both worlds:
    - Claude's intelligence for decisions
    - Local persistence for long-running tasks
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"
        self.processing_dir = self.base_dir / "processing"
        self.output_dir = self.base_dir / "output"
        self.state_file = self.base_dir / ".orchestrator_state.json"
        
        # Create directories
        for dir in [self.input_dir, self.processing_dir, self.output_dir]:
            dir.mkdir(exist_ok=True)
            
        # Load state
        self.state = self._load_state()
        
        # Initialize cinematic analyzer (WARNING: This loads 2GB+ of models)
        try:
            self.cinematic_analyzer = CinematicSceneAnalyzer()
            logger.info("✅ Cinematic analyzer loaded - Full descriptive capability enabled")
        except Exception as e:
            logger.warning(f"⚠️ Cinematic analyzer failed to load: {e}")
            logger.warning("Falling back to basic descriptions")
            self.cinematic_analyzer = None
        
    def _load_state(self) -> Dict[str, Any]:
        """Load persistent state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "processed_videos": [],
            "failed_videos": [],
            "current_processing": None,
            "statistics": {
                "total_processed": 0,
                "total_frames": 0,
                "success_rate": 100.0
            }
        }
    
    def _save_state(self):
        """Save state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    async def process_video_nuclear(self, video_path: Path) -> Dict[str, Any]:
        """
        Process a single video with NUCLEAR precision
        Returns detailed results
        """
        logger.info(f"🚀 NUCLEAR PROCESSING: {video_path.name}")
        
        result = {
            "video": str(video_path),
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
            "nuclear_extraction": {},
            "analysis": {},
            "validation": {},
            "outputs": {}
        }
        
        try:
            # Update state
            self.state["current_processing"] = str(video_path)
            self._save_state()
            
            # STEP 1: Nuclear Frame Extraction
            logger.info("📸 Extracting exactly 1,800 frames...")
            output_dir = self.processing_dir / f"{video_path.stem}_nuclear"
            output_dir.mkdir(exist_ok=True)
            
            extractor = NuclearFrameExtractor(
                video_path=str(video_path),
                output_dir=str(output_dir),
                target_fps=30,
                duration=60,
                force_frame_count=True
            )
            
            extraction_result = extractor.extract()
            result["nuclear_extraction"] = {
                "success": extraction_result["success"],
                "frame_count": extraction_result["frame_count"],
                "actual_fps": extraction_result["fps"],
                "duration": extraction_result["duration"]
            }
            
            if not extraction_result["success"]:
                raise Exception(f"Nuclear extraction failed: {extraction_result.get('error')}")
            
            # VERIFY: Exactly 1,800 frames
            if extraction_result["frame_count"] != 1800:
                raise Exception(f"FRAME COUNT VIOLATION: {extraction_result['frame_count']} != 1,800")
            
            # STEP 2: Validate frames
            logger.info("🔍 Validating frame integrity...")
            from frame_validation_system import FrameValidationSystem
            
            validator = FrameValidationSystem(str(output_dir))
            validation_result = validator.validate_all()
            result["validation"] = validation_result["summary"]
            
            if not validation_result["valid"]:
                raise Exception(f"Frame validation failed: {validation_result['errors']}")
            
            # STEP 3: Generate outputs
            logger.info("📝 Generating JSONL, JSON, and screenplay...")
            
            if self.cinematic_analyzer:
                # CINEMATIC MODE: Generate detailed, beautiful descriptions
                logger.info("🎬 Generating CINEMATIC descriptions...")
                
                cinematic_analysis = self.cinematic_analyzer.analyze_video_cinematic(
                    str(video_path),
                    str(output_dir)
                )
                
                # Generate cinema-grade JSONL (exactly like your examples)
                jsonl_path = output_dir / "cinematic_scenes.jsonl"
                with open(jsonl_path, 'w') as f:
                    f.write(self.cinematic_analyzer.generate_jsonl_cinematic(cinematic_analysis) + '\n')
                
                # Generate structured scene JSON
                json_path = output_dir / "structured_scenes.json"
                scene_data = self._prepare_scene_data(cinematic_analysis)
                with open(json_path, 'w') as f:
                    json.dump(self.cinematic_analyzer.generate_structured_scene_json(scene_data), f, indent=2)
                
                # Generate screenplay format
                screenplay_path = output_dir / "screenplay.txt"
                with open(screenplay_path, 'w') as f:
                    f.write(self.cinematic_analyzer.generate_screenplay_format(scene_data))
                
                result["outputs"] = {
                    "jsonl": str(jsonl_path),
                    "json": str(json_path),
                    "screenplay": str(screenplay_path),
                    "quality": "CINEMATIC"
                }
                
                # Store cinematic analysis
                result["cinematic_analysis"] = cinematic_analysis
                
            else:
                # FALLBACK: Basic outputs
                outputs = extractor.generate_outputs()
                result["outputs"] = {
                    "jsonl": outputs.get("jsonl_path"),
                    "json": outputs.get("json_path"),
                    "screenplay": outputs.get("screenplay_path"),
                    "quality": "BASIC"
                }
            
            # STEP 4: Comprehensive analysis
            logger.info("🧠 Running comprehensive analysis...")
            analyzer = ProjectAnalyzer(str(self.base_dir))
            analysis = await analyzer._analyze_video(
                video_path,
                f"nuclear_{video_path.stem}"
            )
            result["analysis"] = analysis
            
            # STEP 5: Create edit decisions
            logger.info("🎬 Creating AI edit decisions...")
            engine = ClaudeDecisionEngine(str(self.base_dir))
            decisions = await engine.create_edit_decision_list("viral")
            result["edit_decisions"] = decisions
            
            # Success!
            result["status"] = "completed"
            self.state["processed_videos"].append(result)
            self.state["statistics"]["total_processed"] += 1
            self.state["statistics"]["total_frames"] += 1800
            
            logger.info("✅ NUCLEAR PROCESSING COMPLETE!")
            
        except Exception as e:
            logger.error(f"❌ NUCLEAR PROCESSING FAILED: {str(e)}")
            result["status"] = "failed"
            result["error"] = str(e)
            self.state["failed_videos"].append(result)
            
            # Update success rate
            total = len(self.state["processed_videos"]) + len(self.state["failed_videos"])
            if total > 0:
                self.state["statistics"]["success_rate"] = (
                    len(self.state["processed_videos"]) / total * 100
                )
        
        finally:
            self.state["current_processing"] = None
            self._save_state()
            
        return result
    
    async def watch_input_directory(self):
        """
        Continuously watch for new videos
        This runs as a daemon that Claude can monitor
        """
        logger.info(f"👁️ Watching {self.input_dir} for new videos...")
        
        processed = set(self.state["processed_videos"])
        
        while True:
            try:
                # Find new videos
                video_files = list(self.input_dir.glob("*.mp4")) + \
                             list(self.input_dir.glob("*.mov")) + \
                             list(self.input_dir.glob("*.avi"))
                
                for video in video_files:
                    if str(video) not in processed:
                        logger.info(f"🆕 New video detected: {video.name}")
                        
                        # Process it
                        result = await self.process_video_nuclear(video)
                        
                        if result["status"] == "completed":
                            # Move to processed
                            processed_path = self.output_dir / video.name
                            video.rename(processed_path)
                            logger.info(f"➡️ Moved to output: {processed_path}")
                        
                        processed.add(str(video))
                
                # Check every 5 seconds
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping watch mode...")
                break
            except Exception as e:
                logger.error(f"Watch error: {e}")
                await asyncio.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current orchestrator status
        Claude can call this to understand progress
        """
        return {
            "state": self.state,
            "input_videos": len(list(self.input_dir.glob("*.mp4"))),
            "currently_processing": self.state["current_processing"],
            "success_rate": self.state["statistics"]["success_rate"],
            "total_frames_processed": self.state["statistics"]["total_frames"]
        }
    
    def get_commands(self) -> List[str]:
        """
        Return available commands for Claude to execute
        """
        return [
            "python claude_orchestrator.py watch",
            "python claude_orchestrator.py process <video>",
            "python claude_orchestrator.py status",
            "python claude_orchestrator.py analyze",
            "python claude_orchestrator.py export"
        ]
    
    def _prepare_scene_data(self, cinematic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare scene data for structured output generation
        """
        # Extract key elements from cinematic analysis
        scene_data = {
            "video_id": cinematic_analysis.get("video_id", "unknown"),
            "scene_number": 1,
            "location": self._infer_location(cinematic_analysis),
            "time_of_day": self._infer_time_of_day(cinematic_analysis),
            "lighting": cinematic_analysis.get("cinematic_elements", {}).get("lighting", {}),
            "objects": self._extract_objects(cinematic_analysis),
            "narrative": cinematic_analysis.get("cinematic_elements", {}).get("narrative", ""),
            "mood": self._extract_mood(cinematic_analysis),
            "camera_suggestions": self._generate_camera_suggestions(cinematic_analysis),
            "opening_description": cinematic_analysis.get("detailed_description", ""),
            "camera_movement": "SLOWLY TRACKS",
            "visual_elements": self._extract_visual_elements(cinematic_analysis),
            "atmosphere_description": self._generate_atmosphere(cinematic_analysis),
            "technical_notes": self._generate_technical_notes(cinematic_analysis)
        }
        
        return scene_data
    
    def _infer_location(self, analysis: Dict[str, Any]) -> str:
        """Infer location from scene analysis"""
        description = analysis.get("detailed_description", "").lower()
        if "kitchen" in description:
            return "modern kitchen"
        elif "forest" in description:
            return "enchanted forest"
        elif "beach" in description or "ocean" in description:
            return "ocean beach"
        elif "city" in description:
            return "urban cityscape"
        else:
            return "interior space"
    
    def _infer_time_of_day(self, analysis: Dict[str, Any]) -> str:
        """Infer time of day from lighting analysis"""
        lighting = analysis.get("cinematic_elements", {}).get("lighting", {})
        color_temp = lighting.get("color_temperature", 5600)
        
        if color_temp < 3500:
            return "twilight"
        elif color_temp < 4500:
            return "golden hour"
        elif color_temp > 6000:
            return "midday"
        else:
            return "morning"
    
    def _extract_objects(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract objects from frame analyses"""
        # This would be populated by object detection in cinematic analyzer
        return [
            {
                "name": "primary subject",
                "position": [0.5, 0.5, 0.7],
                "state": "active",
                "material": "organic"
            }
        ]
    
    def _extract_mood(self, analysis: Dict[str, Any]) -> str:
        """Extract overall mood"""
        moods = []
        for timestamp in analysis.get("timestamps", []):
            if "mood" in timestamp:
                moods.append(timestamp["mood"])
        
        if moods:
            return ", ".join(set(moods[:3]))  # Top 3 unique moods
        return "contemplative"
    
    def _generate_camera_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate camera movement suggestions"""
        complexity = analysis.get("scene_complexity", 0.5)
        
        suggestions = []
        if complexity > 0.7:
            suggestions.append("slow dolly in to emphasize detail")
        if "motion_vectors" in analysis:
            suggestions.append("track subject movement")
        suggestions.append("subtle handheld for organic feel")
        
        return suggestions
    
    def _extract_visual_elements(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key visual elements"""
        elements = []
        description = analysis.get("detailed_description", "")
        
        # Parse description for visual elements
        if "light" in description.lower():
            elements.append("Dynamic lighting creating atmosphere")
        if "motion" in description.lower():
            elements.append("Fluid motion throughout the frame")
        if "color" in description.lower():
            elements.append("Rich color palette defining mood")
        
        return elements if elements else ["Primary subject in frame", "Environmental context"]
    
    def _generate_atmosphere(self, analysis: Dict[str, Any]) -> str:
        """Generate atmospheric description"""
        mood = self._extract_mood(analysis)
        return f"The scene evokes a sense of {mood}. Every element contributes to the overall narrative."
    
    def _generate_technical_notes(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate technical specifications"""
        notes = []
        
        # Frame rate
        fps = analysis.get("fps", 30)
        notes.append(f"Frame rate: {fps}fps")
        
        # Resolution
        resolution = analysis.get("resolution", [1920, 1080])
        notes.append(f"Resolution: {resolution[0]}x{resolution[1]}")
        
        # Scene complexity
        complexity = analysis.get("scene_complexity", 0.5)
        notes.append(f"Scene complexity: {complexity:.1%}")
        
        # Color grading suggestion
        notes.append("Color grading: Match reference mood")
        
        return notes


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Claude Orchestrator - AI-controlled video processing"
    )
    parser.add_argument("command", choices=["watch", "process", "status", "analyze"],
                       help="Command to execute")
    parser.add_argument("video", nargs="?", help="Video file to process")
    parser.add_argument("--dir", default=".", help="Base directory")
    
    args = parser.parse_args()
    
    orchestrator = ClaudeOrchestrator(args.dir)
    
    if args.command == "watch":
        print("🚀 NUCLEAR VIDEO PROCESSOR ACTIVATED")
        print("Drop videos into 'input/' directory")
        print("Press Ctrl+C to stop")
        await orchestrator.watch_input_directory()
        
    elif args.command == "process":
        if not args.video:
            print("ERROR: Specify video file to process")
            sys.exit(1)
        
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: Video not found: {video_path}")
            sys.exit(1)
            
        result = await orchestrator.process_video_nuclear(video_path)
        print(json.dumps(result, indent=2))
        
    elif args.command == "status":
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        
    elif args.command == "analyze":
        # Show detailed analysis of all processed videos
        for video in orchestrator.state["processed_videos"]:
            print(f"\n📹 {video['video']}")
            print(f"   Frames: {video['nuclear_extraction']['frame_count']}")
            print(f"   Status: {video['status']}")
            print(f"   Outputs: {video['outputs']}")


if __name__ == "__main__":
    asyncio.run(main())