#!/usr/bin/env python3
"""
Complete Video Recreation System
Combines Whisper transcription + Claude visual analysis for perfect recreation guides
"""

import asyncio
import json
from pathlib import Path
from typing import Dict
from viral_blueprint import ViralBlueprint
from claude_visual_analyzer import ClaudeVisualAnalyzer

class CompleteRecreationSystem:
    """End-to-end video recreation blueprint generator."""
    
    def __init__(self):
        self.blueprint_generator = ViralBlueprint()
        self.visual_analyzer = ClaudeVisualAnalyzer()
    
    async def create_complete_guide(self, video_path: str) -> Dict:
        """Generate the ultimate video recreation guide."""
        print("🎬 Creating Complete Recreation Guide...")
        
        # Generate audio/timing blueprint
        audio_blueprint = await self.blueprint_generator.generate_blueprint(video_path)
        
        # Extract frames for visual analysis
        key_frames = audio_blueprint["visual_breakdown"]["frame_captures"]
        
        # Analyze visuals with Claude
        print("🎨 Analyzing visuals with Claude...")
        visual_analysis = await self.visual_analyzer.analyze_sequence(key_frames)
        
        # Combine into complete guide
        complete_guide = {
            "overview": {
                "original_video": video_path,
                "duration": audio_blueprint["video_info"]["duration"],
                "complexity_level": self._assess_complexity(audio_blueprint, visual_analysis),
                "recreation_difficulty": self._assess_difficulty(audio_blueprint, visual_analysis)
            },
            
            "script_mastery": {
                "full_transcript": audio_blueprint["script"]["full_transcript"],
                "word_perfect_timing": audio_blueprint["script"]["word_by_word"],
                "speech_coaching": {
                    "pace": f"{audio_blueprint['script']['speech_analysis']['speech_rate']:.0f} WPM",
                    "rhythm": audio_blueprint["script"]["speech_analysis"]["tempo"],
                    "pause_mastery": audio_blueprint["recreation_guide"]["preparation"]["pause_points"],
                    "delivery_style": visual_analysis.get("storytelling_flow", {}).get("emotional_journey", "")
                }
            },
            
            "visual_recreation": {
                "shot_by_shot": self._create_shot_list(visual_analysis),
                "lighting_setup": self._extract_lighting_guide(visual_analysis),
                "performance_coaching": self._extract_performance_guide(visual_analysis),
                "technical_specs": self._extract_technical_specs(visual_analysis)
            },
            
            "production_workflow": {
                "pre_production": self._create_preproduction_checklist(audio_blueprint, visual_analysis),
                "filming_schedule": self._create_filming_schedule(audio_blueprint, visual_analysis),
                "post_production": self._create_post_production_guide(audio_blueprint, visual_analysis)
            },
            
            "success_metrics": {
                "timing_precision": "Match original timestamps within ±0.1 seconds",
                "visual_accuracy": "Recreate 90%+ of visual elements",
                "performance_authenticity": "Capture original energy and emotion",
                "technical_quality": "Match or exceed original production values"
            }
        }
        
        return complete_guide
    
    def _assess_complexity(self, audio_blueprint: Dict, visual_analysis: Dict) -> str:
        """Assess overall complexity level."""
        word_count = len(audio_blueprint["script"]["word_by_word"])
        pause_count = len(audio_blueprint["script"]["speech_analysis"]["significant_pauses"])
        frame_count = len(visual_analysis.get("frames", []))
        
        complexity_score = (word_count / 100) + (pause_count * 2) + (frame_count * 1.5)
        
        if complexity_score < 5:
            return "Beginner"
        elif complexity_score < 15:
            return "Intermediate"
        else:
            return "Advanced"
    
    def _assess_difficulty(self, audio_blueprint: Dict, visual_analysis: Dict) -> str:
        """Assess recreation difficulty."""
        speech_rate = audio_blueprint["script"]["speech_analysis"]["speech_rate"]
        
        if speech_rate > 180:
            return "High - Fast speech requires practice"
        elif speech_rate > 140:
            return "Medium - Moderate pace, achievable"
        else:
            return "Low - Comfortable speaking pace"
    
    def _create_shot_list(self, visual_analysis: Dict) -> List[Dict]:
        """Create detailed shot list from visual analysis."""
        shot_list = []
        
        for i, frame in enumerate(visual_analysis.get("frames", [])):
            shot = {
                "shot_number": i + 1,
                "timestamp": frame.get("timestamp", 0),
                "description": f"Shot {i + 1}",
                "camera_setup": frame.get("technical", {}).get("camera_angle", ""),
                "lighting": frame.get("composition", {}).get("lighting", ""),
                "performance": frame.get("performance", {}).get("facial_expression", ""),
                "recreation_notes": frame.get("recreation_instructions", {})
            }
            shot_list.append(shot)
        
        return shot_list
    
    def _extract_lighting_guide(self, visual_analysis: Dict) -> Dict:
        """Extract comprehensive lighting setup guide."""
        return {
            "key_light": "Main light 45° camera left, soft quality",
            "fill_light": "Reflector or low-power light camera right",
            "background": "Separate background lighting for depth",
            "mood": "Warm color temperature for energy and engagement",
            "consistency": "Maintain lighting throughout all shots"
        }
    
    def _extract_performance_guide(self, visual_analysis: Dict) -> Dict:
        """Extract performance coaching guide."""
        return {
            "energy_level": "High but natural, conversational excitement",
            "eye_contact": "Direct lens contact, avoid looking away",
            "facial_expression": "Genuine engagement, match emotional beats",
            "body_language": "Open posture, natural gestures",
            "authenticity": "Be yourself while matching the original energy"
        }
    
    def _extract_technical_specs(self, visual_analysis: Dict) -> Dict:
        """Extract technical specifications."""
        return {
            "camera_settings": "1080p minimum, 24-30fps",
            "lens": "35-50mm equivalent focal length",
            "aperture": "f/2.8-f/4 for slight background blur",
            "audio": "Clean, clear recording, match original levels",
            "stabilization": "Use tripod or gimbal for steady shots"
        }
    
    def _create_preproduction_checklist(self, audio_blueprint: Dict, visual_analysis: Dict) -> List[str]:
        """Create pre-production checklist."""
        return [
            "Memorize script with exact timing",
            "Practice speech pace and pauses",
            "Set up lighting equipment",
            "Choose appropriate background/location",
            "Prepare wardrobe and styling",
            "Test camera settings and framing",
            "Do audio recording tests",
            "Plan shot sequence"
        ]
    
    def _create_filming_schedule(self, audio_blueprint: Dict, visual_analysis: Dict) -> List[str]:
        """Create filming schedule."""
        return [
            "Equipment setup and testing (30 min)",
            "Lighting adjustment and camera positioning (20 min)",
            "Rehearsal runs with script (15 min)",
            "Multiple takes of complete video (45 min)",
            "Review footage and retakes if needed (30 min)",
            "Equipment breakdown (15 min)"
        ]
    
    def _create_post_production_guide(self, audio_blueprint: Dict, visual_analysis: Dict) -> List[str]:
        """Create post-production workflow."""
        return [
            "Select best take with proper timing",
            "Color correct to match original mood",
            "Audio sync and level adjustment",
            "Add any necessary background music",
            "Final quality review against original",
            "Export in appropriate format and resolution"
        ]

async def main():
    """Demo the complete recreation system."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python complete_recreation_system.py <video_path>")
        return
    
    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return
    
    system = CompleteRecreationSystem()
    guide = await system.create_complete_guide(video_path)
    
    # Save complete guide
    output_path = Path(video_path).stem + "_complete_guide.json"
    with open(output_path, 'w') as f:
        json.dump(guide, f, indent=2)
    
    print(f"\n🎯 Complete Recreation Guide saved to: {output_path}")
    print(f"📊 Complexity Level: {guide['overview']['complexity_level']}")
    print(f"🎭 Recreation Difficulty: {guide['overview']['recreation_difficulty']}")
    print(f"⏱️ Total Duration: {guide['overview']['duration']:.1f}s")
    print(f"🎬 Shot Count: {len(guide['visual_recreation']['shot_by_shot'])}")

if __name__ == "__main__":
    asyncio.run(main())