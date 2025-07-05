#!/usr/bin/env python3
"""
Cinematic Scene Analyzer - Generate cinema-grade scene descriptions
This is what the nuclear extractor SHOULD have been doing
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CinematicSceneAnalyzer:
    """
    Generate detailed, cinematic scene descriptions that match your examples
    """
    
    def __init__(self):
        # Load advanced models
        self.image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        
        # Load BLIP for detailed descriptions
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Scene understanding components
        self.lighting_analyzer = LightingAnalyzer()
        self.composition_analyzer = CompositionAnalyzer()
        self.motion_tracker = MotionTracker()
        self.narrative_generator = NarrativeGenerator()
        
    def analyze_video_cinematic(self, video_path: str, frames_dir: str) -> Dict[str, Any]:
        """
        Generate cinema-grade analysis of entire video
        """
        logger.info(f"🎬 Generating cinematic analysis for {video_path}")
        
        # Load video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        cap.release()
        
        # Analyze frames
        frame_analyses = []
        scene_segments = []
        current_scene = None
        
        frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
        
        for i, frame_path in enumerate(frame_files):
            if i % 30 == 0:  # Analyze every second
                frame_analysis = self._analyze_frame_detailed(frame_path, i/fps)
                frame_analyses.append(frame_analysis)
                
                # Detect scene changes
                if self._is_scene_change(current_scene, frame_analysis):
                    if current_scene:
                        scene_segments.append(current_scene)
                    current_scene = {
                        "start": i/fps,
                        "end": i/fps,
                        "description": frame_analysis["description"],
                        "mood": frame_analysis["mood"],
                        "technical": frame_analysis["technical"]
                    }
                else:
                    current_scene["end"] = i/fps
        
        if current_scene:
            scene_segments.append(current_scene)
        
        # Generate comprehensive description
        video_description = self._generate_video_description(frame_analyses, scene_segments)
        
        return {
            "video_id": Path(video_path).stem,
            "duration": duration,
            "fps": fps,
            "resolution": [width, height],
            "caption": video_description["caption"],
            "detailed_description": video_description["detailed"],
            "timestamps": scene_segments,
            "scene_complexity": self._calculate_complexity(frame_analyses),
            "motion_vectors": self._extract_motion_vectors(frames_dir),
            "cinematic_elements": {
                "lighting": self._analyze_overall_lighting(frame_analyses),
                "composition": self._analyze_overall_composition(frame_analyses),
                "narrative": self._generate_narrative(scene_segments)
            }
        }
    
    def _analyze_frame_detailed(self, frame_path: Path, timestamp: float) -> Dict[str, Any]:
        """
        Generate detailed analysis of a single frame
        """
        # Load frame
        image = Image.open(frame_path)
        frame = cv2.imread(str(frame_path))
        
        # Generate detailed caption
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=100)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Analyze lighting
        lighting = self.lighting_analyzer.analyze(frame)
        
        # Analyze composition
        composition = self.composition_analyzer.analyze(frame)
        
        # Estimate depth
        depth = self.depth_estimator(image)
        
        # Detect objects and their positions
        objects = self._detect_objects_with_positions(frame)
        
        # Generate mood
        mood = self._analyze_mood(frame, caption)
        
        return {
            "timestamp": timestamp,
            "description": caption,
            "lighting": lighting,
            "composition": composition,
            "depth_map": depth,
            "objects": objects,
            "mood": mood,
            "technical": {
                "exposure": self._analyze_exposure(frame),
                "color_temperature": self._estimate_color_temperature(frame),
                "contrast": self._calculate_contrast(frame)
            }
        }
    
    def generate_jsonl_cinematic(self, analysis: Dict[str, Any]) -> str:
        """
        Generate JSONL in the exact format requested
        """
        jsonl_entry = {
            "video_id": analysis["video_id"],
            "duration": round(analysis["duration"], 1),
            "fps": int(analysis["fps"]),
            "resolution": analysis["resolution"],
            "caption": analysis["caption"],
            "detailed_description": analysis["detailed_description"],
            "timestamps": [
                {
                    "start": round(ts["start"], 1),
                    "end": round(ts["end"], 1),
                    "description": ts["description"]
                }
                for ts in analysis["timestamps"]
            ],
            "scene_complexity": round(analysis["scene_complexity"], 2),
            "motion_vectors": analysis["motion_vectors"][:3]  # Top 3 motion vectors
        }
        
        # Add cinematic elements
        if "face_landmarks" in analysis:
            jsonl_entry["face_landmarks"] = True
        if "emotion_tags" in analysis:
            jsonl_entry["emotion_tags"] = analysis["emotion_tags"]
        if "camera_movement" in analysis:
            jsonl_entry["camera_movement"] = analysis["camera_movement"]
        
        return json.dumps(jsonl_entry)
    
    def generate_structured_scene_json(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured scene description in the exact format requested
        """
        return {
            "scene_id": f"{scene_data['video_id']}_{scene_data['scene_number']:03d}",
            "environment": {
                "location": scene_data["location"],
                "time_of_day": scene_data["time_of_day"],
                "lighting": {
                    "type": scene_data["lighting"]["type"],
                    "direction": scene_data["lighting"]["direction"],
                    "intensity": round(scene_data["lighting"]["intensity"], 1),
                    "color_temperature": scene_data["lighting"]["color_temperature"]
                }
            },
            "objects": [
                {
                    "name": obj["name"],
                    "position": [round(p, 1) for p in obj["position"]],
                    "state": obj.get("state", "static"),
                    "material": obj.get("material", "unknown")
                }
                for obj in scene_data["objects"]
            ],
            "narrative": scene_data["narrative"],
            "mood": scene_data["mood"],
            "camera_suggestions": scene_data["camera_suggestions"]
        }
    
    def generate_screenplay_format(self, scene_data: Dict[str, Any]) -> str:
        """
        Generate screenplay-style description in the exact format requested
        """
        screenplay = f"""SCENE {scene_data['scene_number']:03d} - {scene_data['location'].upper()} - {scene_data['time_of_day'].upper()}

{scene_data['opening_description']}

CAMERA {scene_data['camera_movement']}, revealing:

"""
        
        # Add visual elements
        for element in scene_data['visual_elements']:
            screenplay += f"- {element}\n"
        
        screenplay += f"\n{scene_data['atmosphere_description']}\n\n"
        
        # Add technical notes
        screenplay += "TECHNICAL NOTES:\n"
        for note in scene_data['technical_notes']:
            screenplay += f"- {note}\n"
        
        return screenplay


class LightingAnalyzer:
    """Analyze lighting in frames"""
    
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Analyze lighting direction
        gradient_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
        
        # Determine primary light direction
        avg_gradient_x = np.mean(gradient_x)
        avg_gradient_y = np.mean(gradient_y)
        
        if abs(avg_gradient_x) > abs(avg_gradient_y):
            direction = "side" if avg_gradient_x > 0 else "opposite side"
        else:
            direction = "top" if avg_gradient_y > 0 else "bottom"
        
        # Analyze light quality
        intensity = np.mean(l_channel) / 255.0
        contrast = np.std(l_channel) / 255.0
        
        # Determine lighting type
        if contrast < 0.2:
            light_type = "soft"
        elif contrast > 0.4:
            light_type = "hard"
        else:
            light_type = "natural"
        
        return {
            "type": light_type,
            "direction": direction,
            "intensity": intensity,
            "contrast": contrast,
            "color_temperature": self._estimate_color_temp(frame)
        }
    
    def _estimate_color_temp(self, frame: np.ndarray) -> int:
        # Simple color temperature estimation
        b, g, r = cv2.split(frame)
        avg_b = np.mean(b)
        avg_r = np.mean(r)
        
        if avg_b > avg_r * 1.2:
            return 6500  # Cool/daylight
        elif avg_r > avg_b * 1.2:
            return 3200  # Warm/tungsten
        else:
            return 5600  # Neutral


class CompositionAnalyzer:
    """Analyze frame composition"""
    
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        
        # Rule of thirds analysis
        thirds_points = [
            (w//3, h//3), (2*w//3, h//3),
            (w//3, 2*h//3), (2*w//3, 2*h//3)
        ]
        
        # Find points of interest
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        
        composition_score = 0
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                # Check proximity to thirds points
                for tp in thirds_points:
                    dist = np.sqrt((x - tp[0])**2 + (y - tp[1])**2)
                    if dist < 50:
                        composition_score += 1
        
        return {
            "rule_of_thirds_score": min(composition_score / 10, 1.0),
            "balance": self._analyze_balance(frame),
            "leading_lines": self._detect_leading_lines(frame)
        }
    
    def _analyze_balance(self, frame: np.ndarray) -> str:
        # Simple balance analysis
        h, w = frame.shape[:2]
        left_half = frame[:, :w//2]
        right_half = frame[:, w//2:]
        
        left_weight = np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
        right_weight = np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))
        
        if abs(left_weight - right_weight) < 10:
            return "balanced"
        elif left_weight > right_weight:
            return "left-heavy"
        else:
            return "right-heavy"
    
    def _detect_leading_lines(self, frame: np.ndarray) -> bool:
        # Detect lines using Hough transform
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        return lines is not None and len(lines) > 3


class MotionTracker:
    """Track motion between frames"""
    
    def calculate_motion_vectors(self, frame1: np.ndarray, frame2: np.ndarray) -> List[List[float]]:
        # Optical flow calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Extract main motion vectors
        h, w = flow.shape[:2]
        regions = [(0, 0, w//2, h//2), (w//2, 0, w, h//2), 
                   (0, h//2, w//2, h), (w//2, h//2, w, h)]
        
        motion_vectors = []
        for x1, y1, x2, y2 in regions:
            region_flow = flow[y1:y2, x1:x2]
            avg_x = np.mean(region_flow[:, :, 0])
            avg_y = np.mean(region_flow[:, :, 1])
            motion_vectors.append([round(avg_x, 2), round(avg_y, 2)])
        
        return motion_vectors


class NarrativeGenerator:
    """Generate narrative descriptions"""
    
    def generate(self, scene_data: Dict[str, Any]) -> str:
        # This would use a language model to generate narrative
        # For now, using templates
        templates = {
            "morning": "A peaceful morning scene in {location}. {lighting_desc} creates {mood} atmosphere.",
            "sunset": "Golden hour transforms {location} into a canvas of warm hues. {action} as {lighting_desc}.",
            "night": "Under the cover of darkness, {location} takes on a {mood} quality. {lighting_desc} {action}."
        }
        
        time_of_day = scene_data.get("time_of_day", "day")
        template = templates.get(time_of_day, "A scene unfolds in {location}. {lighting_desc} {action}.")
        
        return template.format(
            location=scene_data.get("location", "an unknown location"),
            lighting_desc=self._describe_lighting(scene_data.get("lighting", {})),
            mood=scene_data.get("mood", "mysterious"),
            action=scene_data.get("action", "Time passes quietly")
        )
    
    def _describe_lighting(self, lighting: Dict[str, Any]) -> str:
        intensity = lighting.get("intensity", 0.5)
        light_type = lighting.get("type", "natural")
        
        if intensity > 0.7:
            return f"Bright {light_type} light"
        elif intensity < 0.3:
            return f"Dim {light_type} light"
        else:
            return f"Soft {light_type} light"


# Integration with the orchestrator
def upgrade_orchestrator_descriptions():
    """
    Upgrade the orchestrator to use cinematic descriptions
    """
    return """
    # In claude_orchestrator.py, replace the basic analysis with:
    
    from cinematic_scene_analyzer import CinematicSceneAnalyzer
    
    class ClaudeOrchestrator:
        def __init__(self):
            # ... existing code ...
            self.cinematic_analyzer = CinematicSceneAnalyzer()
        
        async def process_video_nuclear(self, video_path: Path):
            # ... existing extraction code ...
            
            # UPGRADE: Use cinematic analysis
            cinematic_analysis = self.cinematic_analyzer.analyze_video_cinematic(
                str(video_path),
                str(output_dir)
            )
            
            # Generate cinema-grade outputs
            with open(output_dir / "cinematic.jsonl", 'w') as f:
                for scene in cinematic_analysis["scenes"]:
                    f.write(self.cinematic_analyzer.generate_jsonl_cinematic(scene) + '\\n')
            
            # ... rest of processing ...
    """