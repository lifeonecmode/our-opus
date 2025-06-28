#!/usr/bin/env python3
"""
Viral Video Blueprint Generator
Creates step-by-step recreation guides using Whisper + Claude analysis
"""

import asyncio
import whisper
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import subprocess
import tempfile

class ViralBlueprint:
    """Generate detailed video recreation blueprints combining audio and visual analysis."""
    
    def __init__(self, whisper_model="medium"):
        self.whisper_model = whisper.load_model(whisper_model)
        
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video for Whisper processing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            cmd = [
                "ffmpeg", "-i", video_path, 
                "-ac", "1", "-ar", "16000", 
                "-y", temp_audio.name
            ]
            subprocess.run(cmd, capture_output=True)
            return temp_audio.name
    
    def get_word_timestamps(self, audio_path: str) -> List[Dict]:
        """Get word-level timestamps using Whisper."""
        result = self.whisper_model.transcribe(
            audio_path, 
            word_timestamps=True,
            verbose=False
        )
        
        word_segments = []
        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    word_segments.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })
        
        return word_segments
    
    def extract_key_frames(self, video_path: str, timestamps: List[float]) -> List[str]:
        """Extract frames at specific timestamps."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frame_path = f"temp_frame_{i}_{timestamp:.2f}.jpg"
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
        
        cap.release()
        return frames
    
    def analyze_timing_rhythm(self, word_segments: List[Dict]) -> Dict:
        """Analyze speech timing and rhythm patterns."""
        if not word_segments:
            return {}
            
        # Calculate speech rate
        total_duration = word_segments[-1]["end"] - word_segments[0]["start"]
        total_words = len(word_segments)
        words_per_minute = (total_words / total_duration) * 60
        
        # Find pauses and emphasis
        pauses = []
        for i in range(len(word_segments) - 1):
            gap = word_segments[i + 1]["start"] - word_segments[i]["end"]
            if gap > 0.3:  # Significant pause
                pauses.append({
                    "after_word": word_segments[i]["word"],
                    "duration": gap,
                    "timestamp": word_segments[i]["end"]
                })
        
        return {
            "speech_rate": words_per_minute,
            "total_duration": total_duration,
            "word_count": total_words,
            "significant_pauses": pauses,
            "tempo": "fast" if words_per_minute > 160 else "medium" if words_per_minute > 120 else "slow"
        }
    
    async def generate_blueprint(self, video_path: str) -> Dict:
        """Generate complete recreation blueprint."""
        print("🎯 Generating Viral Recreation Blueprint...")
        
        # Extract audio and get transcription with timestamps
        print("📝 Transcribing with word-level timestamps...")
        audio_path = self.extract_audio(video_path)
        word_segments = self.get_word_timestamps(audio_path)
        
        # Analyze timing patterns
        timing_analysis = self.analyze_timing_rhythm(word_segments)
        
        # Extract key frames for visual analysis
        print("🎬 Extracting key visual moments...")
        key_timestamps = [0]  # Always include start
        
        # Add timestamps for pauses (scene changes)
        for pause in timing_analysis.get("significant_pauses", []):
            key_timestamps.append(pause["timestamp"])
        
        # Add middle and end
        if timing_analysis.get("total_duration"):
            key_timestamps.append(timing_analysis["total_duration"] / 2)
            key_timestamps.append(timing_analysis["total_duration"])
        
        key_frames = self.extract_key_frames(video_path, key_timestamps)
        
        # Build comprehensive blueprint
        blueprint = {
            "video_info": {
                "path": video_path,
                "duration": timing_analysis.get("total_duration", 0)
            },
            "script": {
                "full_transcript": " ".join([w["word"] for w in word_segments]),
                "word_by_word": word_segments,
                "speech_analysis": timing_analysis
            },
            "visual_breakdown": {
                "key_timestamps": key_timestamps,
                "frame_captures": key_frames
            },
            "recreation_guide": self._generate_step_by_step_guide(word_segments, timing_analysis)
        }
        
        # Cleanup temp files
        Path(audio_path).unlink()
        for frame in key_frames:
            Path(frame).unlink()
        
        return blueprint
    
    def _generate_step_by_step_guide(self, word_segments: List[Dict], timing: Dict) -> Dict:
        """Generate actionable recreation steps."""
        return {
            "preparation": {
                "script_memorization": "Practice the exact transcript with precise timing",
                "pace_target": f"Maintain {timing.get('speech_rate', 0):.0f} words per minute",
                "pause_points": [f"Pause {p['duration']:.1f}s after '{p['after_word']}'" 
                               for p in timing.get('significant_pauses', [])]
            },
            "delivery_instructions": {
                "tempo": timing.get('tempo', 'medium'),
                "timing_precision": "Match word-level timestamps within 0.1 seconds",
                "emphasis_points": "Follow pause patterns for dramatic effect"
            },
            "technical_setup": {
                "recording_duration": f"{timing.get('total_duration', 0):.1f} seconds",
                "audio_sync": "Use word timestamps to ensure perfect lip-sync",
                "scene_timing": "Change visuals at pause points for maximum impact"
            }
        }

async def main():
    """Demo the blueprint generator."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python viral_blueprint.py <video_path>")
        return
    
    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return
    
    generator = ViralBlueprint()
    blueprint = await generator.generate_blueprint(video_path)
    
    # Save blueprint
    output_path = Path(video_path).stem + "_blueprint.json"
    with open(output_path, 'w') as f:
        json.dump(blueprint, f, indent=2)
    
    print(f"\n🎯 Blueprint saved to: {output_path}")
    print(f"📊 Total words: {len(blueprint['script']['word_by_word'])}")
    print(f"⏱️ Duration: {blueprint['video_info']['duration']:.1f}s")
    print(f"🗣️ Speech rate: {blueprint['script']['speech_analysis']['speech_rate']:.0f} WPM")
    print(f"⏸️ Key pauses: {len(blueprint['script']['speech_analysis']['significant_pauses'])}")

if __name__ == "__main__":
    asyncio.run(main())