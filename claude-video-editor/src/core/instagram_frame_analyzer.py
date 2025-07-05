#!/usr/bin/env python3
"""
Instagram Video Frame-by-Frame Analyzer
Generates detailed analysis in multiple formats for Instagram videos
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstagramFrameAnalyzer:
    """Comprehensive frame-by-frame analyzer for Instagram videos"""
    
    def __init__(self, video_path: str, output_dir: str = "instagram_analysis"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Analysis storage
        self.frame_data = []
        self.scene_data = []
        
        logger.info(f"Initialized analyzer for: {self.video_path}")
        logger.info(f"Video specs: {self.width}x{self.height} @ {self.fps}fps, {self.duration:.2f}s")
    
    def analyze_frame(self, frame: np.ndarray, frame_num: int) -> Dict[str, Any]:
        """Analyze a single frame comprehensively"""
        timestamp = frame_num / self.fps
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic metrics
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        contrast = np.std(gray)
        
        # Edge detection for composition analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Motion blur detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_metric = np.var(laplacian)
        
        # Color analysis
        hist_b = cv2.calcHist([frame], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [32], [0, 256])
        
        # Dominant colors using k-means
        pixels = frame.reshape(-1, 3)
        # Sample pixels for speed
        sample_size = min(1000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Simple k-means for dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(sample_pixels)
        dominant_colors = kmeans.cluster_centers_.tolist()
        
        # Shot type detection (simplified)
        shot_type = self.detect_shot_type(frame)
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Composition analysis
        composition = self.analyze_composition(frame, edges)
        
        # Emotional tone (heuristic based on color/brightness)
        emotional_tone = self.detect_emotional_tone(brightness, saturation, dominant_colors)
        
        return {
            "frame_id": frame_num,
            "timestamp": round(timestamp, 3),
            "technical": {
                "brightness": round(brightness, 2),
                "saturation": round(saturation, 2),
                "contrast": round(contrast, 2),
                "edge_density": round(edge_density, 4),
                "blur_metric": round(blur_metric, 2),
                "is_sharp": blur_metric > 100
            },
            "color": {
                "dominant_colors": [[round(c, 2) for c in color] for color in dominant_colors],
                "avg_hue": round(np.mean(hsv[:, :, 0]), 2),
                "color_temperature": self.estimate_color_temperature(frame),
                "histogram": {
                    "blue": hist_b.flatten().tolist()[::4],  # Subsample for size
                    "green": hist_g.flatten().tolist()[::4],
                    "red": hist_r.flatten().tolist()[::4]
                }
            },
            "composition": composition,
            "shot_type": shot_type,
            "faces": {
                "count": len(faces),
                "positions": faces.tolist() if len(faces) > 0 else []
            },
            "emotional_tone": emotional_tone,
            "cinematography_score": self.calculate_cinematography_score(
                composition, edge_density, blur_metric, len(faces)
            )
        }
    
    def detect_shot_type(self, frame: np.ndarray) -> str:
        """Detect shot type based on frame content"""
        # Simplified shot detection based on face size and edge patterns
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Determine shot type by face size relative to frame
            max_face_area = max([w * h for x, y, w, h in faces])
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = max_face_area / frame_area
            
            if face_ratio > 0.3:
                return "extreme_close"
            elif face_ratio > 0.15:
                return "close"
            elif face_ratio > 0.05:
                return "medium"
            else:
                return "wide"
        else:
            # Use edge density for non-face shots
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < 0.05:
                return "extreme_wide"
            elif edge_density < 0.1:
                return "wide"
            else:
                return "medium"
    
    def analyze_composition(self, frame: np.ndarray, edges: np.ndarray) -> Dict[str, Any]:
        """Analyze frame composition"""
        h, w = frame.shape[:2]
        
        # Rule of thirds analysis
        thirds_v = [w // 3, 2 * w // 3]
        thirds_h = [h // 3, 2 * h // 3]
        
        # Check edge concentration near thirds
        thirds_score = 0
        for v in thirds_v:
            thirds_score += np.sum(edges[:, max(0, v-10):min(w, v+10)]) / edges.size
        for h_line in thirds_h:
            thirds_score += np.sum(edges[max(0, h_line-10):min(h, h_line+10), :]) / edges.size
        
        # Symmetry analysis
        left_half = edges[:, :w//2]
        right_half = cv2.flip(edges[:, w//2:], 1)
        symmetry_score = 1 - (np.sum(np.abs(left_half - right_half[:, :left_half.shape[1]])) / left_half.size)
        
        # Leading lines detection (simplified)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        has_leading_lines = lines is not None and len(lines) > 5
        
        return {
            "rule_of_thirds_score": round(thirds_score * 100, 2),
            "symmetry_score": round(symmetry_score, 2),
            "has_leading_lines": has_leading_lines,
            "is_centered": self.check_centered_composition(edges),
            "visual_weight_distribution": self.calculate_visual_weight(frame)
        }
    
    def check_centered_composition(self, edges: np.ndarray) -> bool:
        """Check if composition is centered"""
        h, w = edges.shape
        center_region = edges[h//4:3*h//4, w//4:3*w//4]
        center_density = np.sum(center_region > 0) / center_region.size
        overall_density = np.sum(edges > 0) / edges.size
        return center_density > overall_density * 1.5
    
    def calculate_visual_weight(self, frame: np.ndarray) -> str:
        """Calculate visual weight distribution"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        top_weight = np.mean(gray[:h//2, :])
        bottom_weight = np.mean(gray[h//2:, :])
        left_weight = np.mean(gray[:, :w//2])
        right_weight = np.mean(gray[:, w//2:])
        
        if abs(top_weight - bottom_weight) < 10 and abs(left_weight - right_weight) < 10:
            return "balanced"
        elif top_weight > bottom_weight:
            return "top_heavy"
        elif bottom_weight > top_weight:
            return "bottom_heavy"
        elif left_weight > right_weight:
            return "left_heavy"
        else:
            return "right_heavy"
    
    def estimate_color_temperature(self, frame: np.ndarray) -> str:
        """Estimate color temperature of frame"""
        avg_color = np.mean(frame, axis=(0, 1))
        blue, green, red = avg_color
        
        if red > blue * 1.2:
            return "warm"
        elif blue > red * 1.2:
            return "cool"
        else:
            return "neutral"
    
    def detect_emotional_tone(self, brightness: float, saturation: float, 
                            dominant_colors: List[List[float]]) -> Dict[str, Any]:
        """Detect emotional tone based on visual elements"""
        # Simplified emotional detection based on color psychology
        emotion_scores = {
            "energetic": 0,
            "calm": 0,
            "dramatic": 0,
            "cheerful": 0,
            "mysterious": 0,
            "neutral": 0
        }
        
        # Brightness influence
        if brightness > 180:
            emotion_scores["cheerful"] += 0.3
            emotion_scores["energetic"] += 0.2
        elif brightness < 80:
            emotion_scores["mysterious"] += 0.3
            emotion_scores["dramatic"] += 0.2
        else:
            emotion_scores["neutral"] += 0.2
        
        # Saturation influence
        if saturation > 150:
            emotion_scores["energetic"] += 0.3
            emotion_scores["dramatic"] += 0.1
        elif saturation < 50:
            emotion_scores["calm"] += 0.3
            emotion_scores["neutral"] += 0.2
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            "dominant": dominant_emotion[0],
            "confidence": round(dominant_emotion[1], 2),
            "all_emotions": {k: round(v, 2) for k, v in emotion_scores.items()}
        }
    
    def calculate_cinematography_score(self, composition: Dict, edge_density: float, 
                                     blur_metric: float, face_count: int) -> float:
        """Calculate overall cinematography quality score"""
        score = 0.0
        
        # Composition score (40%)
        comp_score = (composition["rule_of_thirds_score"] / 100 * 0.3 + 
                     composition["symmetry_score"] * 0.2 +
                     (0.2 if composition["has_leading_lines"] else 0))
        score += comp_score * 0.4
        
        # Technical quality (30%)
        sharpness_score = min(1.0, blur_metric / 500) if blur_metric > 0 else 0
        score += sharpness_score * 0.3
        
        # Subject presence (20%)
        subject_score = min(1.0, face_count * 0.5) if face_count > 0 else 0.3
        score += subject_score * 0.2
        
        # Visual interest (10%)
        interest_score = min(1.0, edge_density * 10)
        score += interest_score * 0.1
        
        return round(score, 2)
    
    def detect_scene_changes(self) -> List[int]:
        """Detect scene changes in the video"""
        scene_changes = [0]  # First frame is always a scene start
        prev_hist = None
        
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Calculate histogram difference
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if diff < 0.7:  # Threshold for scene change
                    scene_changes.append(frame_num)
            
            prev_hist = hist
            frame_num += 1
            
            # Skip frames for efficiency
            if frame_num % 10 != 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Reset capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return scene_changes
    
    def analyze_video(self, interval: float = 0.1, detect_scenes: bool = True):
        """Analyze video at specified interval"""
        logger.info(f"Starting analysis with {interval}s interval")
        
        # Detect scene changes if requested
        scene_changes = self.detect_scene_changes() if detect_scenes else [0]
        logger.info(f"Detected {len(scene_changes)} scenes")
        
        # Analyze frames at interval
        frame_interval = int(self.fps * interval)
        frame_num = 0
        current_scene = 0
        scene_frames = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num % frame_interval == 0:
                logger.info(f"Analyzing frame {frame_num}/{self.frame_count}")
                
                # Analyze frame
                frame_analysis = self.analyze_frame(frame, frame_num)
                self.frame_data.append(frame_analysis)
                scene_frames.append(frame_analysis)
                
                # Save key frame
                if frame_num % (frame_interval * 10) == 0:  # Every 10th analyzed frame
                    frame_path = self.output_dir / f"frame_{frame_num:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
            
            # Check for scene change
            if current_scene < len(scene_changes) - 1 and frame_num >= scene_changes[current_scene + 1]:
                # Save scene data
                self.scene_data.append({
                    "scene_id": f"scene_{current_scene:03d}",
                    "start_frame": scene_changes[current_scene],
                    "end_frame": frame_num,
                    "start_time": scene_changes[current_scene] / self.fps,
                    "end_time": frame_num / self.fps,
                    "duration": (frame_num - scene_changes[current_scene]) / self.fps,
                    "frames": scene_frames,
                    "summary": self.summarize_scene(scene_frames)
                })
                current_scene += 1
                scene_frames = []
            
            frame_num += 1
        
        # Save last scene
        if scene_frames:
            self.scene_data.append({
                "scene_id": f"scene_{current_scene:03d}",
                "start_frame": scene_changes[current_scene],
                "end_frame": frame_num - 1,
                "start_time": scene_changes[current_scene] / self.fps,
                "end_time": (frame_num - 1) / self.fps,
                "duration": (frame_num - 1 - scene_changes[current_scene]) / self.fps,
                "frames": scene_frames,
                "summary": self.summarize_scene(scene_frames)
            })
        
        self.cap.release()
        logger.info(f"Analysis complete: {len(self.frame_data)} frames analyzed")
    
    def summarize_scene(self, scene_frames: List[Dict]) -> Dict[str, Any]:
        """Summarize a scene from its frames"""
        if not scene_frames:
            return {}
        
        # Aggregate metrics
        avg_brightness = np.mean([f["technical"]["brightness"] for f in scene_frames])
        avg_saturation = np.mean([f["technical"]["saturation"] for f in scene_frames])
        dominant_shot = max(set([f["shot_type"] for f in scene_frames]), 
                          key=[f["shot_type"] for f in scene_frames].count)
        
        # Emotional arc
        emotions = defaultdict(list)
        for frame in scene_frames:
            for emotion, score in frame["emotional_tone"]["all_emotions"].items():
                emotions[emotion].append(score)
        
        emotional_arc = {emotion: {
            "mean": round(np.mean(scores), 2),
            "peak": round(max(scores), 2),
            "variation": round(np.std(scores), 2)
        } for emotion, scores in emotions.items()}
        
        # Cinematography quality
        cinematography_scores = [f["cinematography_score"] for f in scene_frames]
        
        return {
            "technical_summary": {
                "avg_brightness": round(avg_brightness, 2),
                "avg_saturation": round(avg_saturation, 2),
                "dominant_shot_type": dominant_shot,
                "shot_variety": len(set([f["shot_type"] for f in scene_frames]))
            },
            "emotional_arc": emotional_arc,
            "cinematography": {
                "avg_score": round(np.mean(cinematography_scores), 2),
                "peak_score": round(max(cinematography_scores), 2),
                "consistency": round(1 - np.std(cinematography_scores), 2)
            },
            "face_presence": sum([f["faces"]["count"] for f in scene_frames]) / len(scene_frames)
        }
    
    def save_outputs(self):
        """Save analysis results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save JSONL format (frame by frame)
        jsonl_path = self.output_dir / f"frames_{timestamp}.jsonl"
        with open(jsonl_path, 'w') as f:
            for frame in self.frame_data:
                f.write(json.dumps(frame) + '\n')
        logger.info(f"Saved JSONL to: {jsonl_path}")
        
        # 2. Save structured JSON
        json_data = {
            "video_id": self.video_path.stem,
            "analysis_timestamp": timestamp,
            "metadata": {
                "duration": self.duration,
                "fps": self.fps,
                "resolution": [self.width, self.height],
                "total_frames": self.frame_count,
                "analyzed_frames": len(self.frame_data)
            },
            "scenes": self.scene_data,
            "frame_data": self.frame_data,
            "summary": self.generate_summary()
        }
        
        json_path = self.output_dir / f"analysis_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved JSON to: {json_path}")
        
        # 3. Save screenplay-style report
        report_path = self.output_dir / f"screenplay_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_screenplay_report())
        logger.info(f"Saved screenplay to: {report_path}")
        
        # 4. Save visualizations
        self.generate_visualizations(timestamp)
        
        # 5. Save markdown report
        md_path = self.output_dir / f"report_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_markdown_report())
        logger.info(f"Saved markdown report to: {md_path}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall video summary"""
        if not self.frame_data:
            return {}
        
        # Technical summary
        brightness_values = [f["technical"]["brightness"] for f in self.frame_data]
        saturation_values = [f["technical"]["saturation"] for f in self.frame_data]
        cinematography_scores = [f["cinematography_score"] for f in self.frame_data]
        
        # Shot type distribution
        shot_types = [f["shot_type"] for f in self.frame_data]
        shot_distribution = {shot: shot_types.count(shot) / len(shot_types) 
                           for shot in set(shot_types)}
        
        # Emotional journey
        emotional_journey = defaultdict(list)
        for frame in self.frame_data:
            emotional_journey[frame["emotional_tone"]["dominant"]].append(frame["timestamp"])
        
        # Key moments (high cinematography scores)
        key_moments = sorted(self.frame_data, 
                           key=lambda x: x["cinematography_score"], 
                           reverse=True)[:5]
        
        return {
            "technical": {
                "brightness": {
                    "mean": round(np.mean(brightness_values), 2),
                    "std": round(np.std(brightness_values), 2),
                    "range": [round(min(brightness_values), 2), 
                            round(max(brightness_values), 2)]
                },
                "saturation": {
                    "mean": round(np.mean(saturation_values), 2),
                    "std": round(np.std(saturation_values), 2),
                    "range": [round(min(saturation_values), 2), 
                            round(max(saturation_values), 2)]
                }
            },
            "cinematography": {
                "avg_score": round(np.mean(cinematography_scores), 2),
                "peak_score": round(max(cinematography_scores), 2),
                "consistency": round(1 - np.std(cinematography_scores), 2)
            },
            "shot_distribution": shot_distribution,
            "dominant_emotions": {emotion: len(times) 
                               for emotion, times in emotional_journey.items()},
            "key_moments": [{"timestamp": m["timestamp"], 
                           "score": m["cinematography_score"],
                           "description": f"{m['shot_type']} shot with {m['emotional_tone']['dominant']} tone"}
                          for m in key_moments]
        }
    
    def generate_screenplay_report(self) -> str:
        """Generate screenplay-style report"""
        report = f"""INSTAGRAM VIDEO ANALYSIS - {self.video_path.name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO SPECIFICATIONS:
- Duration: {self.duration:.2f} seconds
- Resolution: {self.width}x{self.height}
- Frame Rate: {self.fps} fps
- Total Frames: {self.frame_count}
- Analyzed Frames: {len(self.frame_data)}

{'=' * 60}

"""
        
        for i, scene in enumerate(self.scene_data):
            report += f"""
SCENE {i+1:03d} - {scene['start_time']:.2f}s to {scene['end_time']:.2f}s ({scene['duration']:.2f}s)

Visual Summary:
- Dominant Shot Type: {scene['summary']['technical_summary']['dominant_shot_type']}
- Shot Variety: {scene['summary']['technical_summary']['shot_variety']} different shots
- Average Brightness: {scene['summary']['technical_summary']['avg_brightness']}/255
- Average Saturation: {scene['summary']['technical_summary']['avg_saturation']}/255

Emotional Arc:
"""
            # Add top 3 emotions
            emotions = scene['summary']['emotional_arc']
            top_emotions = sorted(emotions.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
            for emotion, stats in top_emotions:
                report += f"- {emotion.capitalize()}: {stats['mean']:.0%} average (peak: {stats['peak']:.0%})\n"
            
            report += f"""
Cinematography Quality:
- Average Score: {scene['summary']['cinematography']['avg_score']:.0%}
- Peak Score: {scene['summary']['cinematography']['peak_score']:.0%}
- Consistency: {scene['summary']['cinematography']['consistency']:.0%}

Technical Notes:
"""
            # Add key frames from scene
            key_frames = sorted(scene['frames'], 
                              key=lambda x: x['cinematography_score'], 
                              reverse=True)[:2]
            for frame in key_frames:
                report += f"- Frame {frame['frame_id']} ({frame['timestamp']:.2f}s): "
                report += f"{frame['shot_type']} shot, "
                report += f"{frame['emotional_tone']['dominant']} tone, "
                report += f"score: {frame['cinematography_score']:.0%}\n"
            
            report += "\n" + "-" * 60 + "\n"
        
        # Add overall summary
        summary = self.generate_summary()
        report += f"""
OVERALL ANALYSIS:

Cinematographic Excellence:
- Average Score: {summary['cinematography']['avg_score']:.0%}
- Peak Achievement: {summary['cinematography']['peak_score']:.0%}
- Visual Consistency: {summary['cinematography']['consistency']:.0%}

Shot Distribution:
"""
        for shot, percentage in summary['shot_distribution'].items():
            report += f"- {shot}: {percentage:.0%}\n"
        
        report += "\nKey Moments:\n"
        for moment in summary['key_moments']:
            report += f"- {moment['timestamp']:.2f}s: {moment['description']} (score: {moment['score']:.0%})\n"
        
        return report
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report"""
        summary = self.generate_summary()
        report = f"""# Instagram Video Analysis Report

**Video**: {self.video_path.name}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Video Specifications

| Metric | Value |
|--------|-------|
| Duration | {self.duration:.2f} seconds |
| Resolution | {self.width}x{self.height} |
| Frame Rate | {self.fps} fps |
| Total Frames | {self.frame_count} |
| Analyzed Frames | {len(self.frame_data)} |
| Number of Scenes | {len(self.scene_data)} |

## Cinematographic Analysis

### Overall Quality
- **Average Score**: {summary['cinematography']['avg_score']:.0%}
- **Peak Score**: {summary['cinematography']['peak_score']:.0%}
- **Consistency**: {summary['cinematography']['consistency']:.0%}

### Shot Composition
"""
        # Add shot distribution chart
        report += "| Shot Type | Percentage |\n|-----------|------------|\n"
        for shot, pct in sorted(summary['shot_distribution'].items(), key=lambda x: x[1], reverse=True):
            report += f"| {shot.replace('_', ' ').title()} | {pct:.1%} |\n"
        
        report += "\n### Emotional Journey\n"
        report += "| Emotion | Frequency |\n|---------|----------|\n"
        for emotion, count in sorted(summary['dominant_emotions'].items(), key=lambda x: x[1], reverse=True):
            report += f"| {emotion.capitalize()} | {count} frames |\n"
        
        report += "\n## Scene-by-Scene Breakdown\n"
        
        for i, scene in enumerate(self.scene_data):
            report += f"""
### Scene {i+1} ({scene['start_time']:.2f}s - {scene['end_time']:.2f}s)

**Duration**: {scene['duration']:.2f}s  
**Dominant Shot**: {scene['summary']['technical_summary']['dominant_shot_type'].replace('_', ' ').title()}  
**Cinematography Score**: {scene['summary']['cinematography']['avg_score']:.0%}

**Emotional Tone**:
"""
            top_emotions = sorted(scene['summary']['emotional_arc'].items(), 
                                key=lambda x: x[1]['mean'], reverse=True)[:3]
            for emotion, stats in top_emotions:
                report += f"- {emotion.capitalize()}: {stats['mean']:.0%}\n"
        
        report += "\n## Key Moments\n\n"
        for i, moment in enumerate(summary['key_moments'], 1):
            report += f"{i}. **{moment['timestamp']:.2f}s** - {moment['description']} (Score: {moment['score']:.0%})\n"
        
        report += "\n## Technical Analysis\n\n"
        report += f"""### Brightness Distribution
- Mean: {summary['technical']['brightness']['mean']:.1f}/255
- Standard Deviation: {summary['technical']['brightness']['std']:.1f}
- Range: {summary['technical']['brightness']['range'][0]:.1f} - {summary['technical']['brightness']['range'][1]:.1f}

### Color Saturation
- Mean: {summary['technical']['saturation']['mean']:.1f}/255
- Standard Deviation: {summary['technical']['saturation']['std']:.1f}
- Range: {summary['technical']['saturation']['range'][0]:.1f} - {summary['technical']['saturation']['range'][1]:.1f}
"""
        
        return report
    
    def generate_visualizations(self, timestamp: str):
        """Generate visualization plots"""
        if not self.frame_data:
            return
        
        # Prepare data
        timestamps = [f["timestamp"] for f in self.frame_data]
        brightness = [f["technical"]["brightness"] for f in self.frame_data]
        saturation = [f["technical"]["saturation"] for f in self.frame_data]
        cinematography_scores = [f["cinematography_score"] for f in self.frame_data]
        
        # Create emotion timeline
        emotions = defaultdict(list)
        for frame in self.frame_data:
            for emotion, score in frame["emotional_tone"]["all_emotions"].items():
                emotions[emotion].append(score)
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 1. Cinematography Score
        axes[0].plot(timestamps, cinematography_scores, 'b-', linewidth=2)
        axes[0].fill_between(timestamps, cinematography_scores, alpha=0.3)
        axes[0].set_ylabel('Cinematography Score')
        axes[0].set_title('Cinematography Quality Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Mark scene changes
        for scene in self.scene_data[1:]:
            axes[0].axvline(x=scene['start_time'], color='red', linestyle='--', alpha=0.5)
        
        # 2. Brightness and Saturation
        axes[1].plot(timestamps, brightness, 'orange', label='Brightness', linewidth=2)
        axes[1].plot(timestamps, saturation, 'purple', label='Saturation', linewidth=2)
        axes[1].set_ylabel('Value (0-255)')
        axes[1].set_title('Brightness and Saturation Levels')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Emotional Timeline
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (emotion, values) in enumerate(emotions.items()):
            if i < len(colors):
                axes[2].plot(timestamps, values, colors[i], label=emotion, alpha=0.7)
        axes[2].set_ylabel('Emotion Score')
        axes[2].set_title('Emotional Arc Throughout Video')
        axes[2].legend(loc='upper right', ncol=3)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        # 4. Shot Type Timeline
        shot_types = [f["shot_type"] for f in self.frame_data]
        shot_map = {"extreme_close": 5, "close": 4, "medium": 3, "wide": 2, "extreme_wide": 1}
        shot_values = [shot_map.get(s, 3) for s in shot_types]
        axes[3].plot(timestamps, shot_values, 'g-', linewidth=2, marker='o', markersize=4)
        axes[3].set_ylabel('Shot Type')
        axes[3].set_title('Shot Type Progression')
        axes[3].set_yticks(list(shot_map.values()))
        axes[3].set_yticklabels(list(shot_map.keys()))
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        viz_path = self.output_dir / f"timeline_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to: {viz_path}")
        
        # Create summary visualization
        self.create_summary_visualization(timestamp)
    
    def create_summary_visualization(self, timestamp: str):
        """Create summary visualization"""
        summary = self.generate_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Shot distribution pie chart
        shot_dist = summary['shot_distribution']
        axes[0, 0].pie(shot_dist.values(), labels=shot_dist.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Shot Type Distribution')
        
        # 2. Emotion distribution bar chart
        emotions = summary['dominant_emotions']
        axes[0, 1].bar(emotions.keys(), emotions.values())
        axes[0, 1].set_title('Emotional Tone Distribution')
        axes[0, 1].set_xlabel('Emotion')
        axes[0, 1].set_ylabel('Frame Count')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Technical metrics
        tech_data = summary['technical']
        metrics = ['Brightness', 'Saturation']
        means = [tech_data['brightness']['mean'], tech_data['saturation']['mean']]
        stds = [tech_data['brightness']['std'], tech_data['saturation']['std']]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 0].bar(x - width/2, means, width, label='Mean', yerr=stds)
        axes[1, 0].set_ylabel('Value (0-255)')
        axes[1, 0].set_title('Technical Metrics Summary')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        
        # 4. Key moments timeline
        key_moments = summary['key_moments']
        times = [m['timestamp'] for m in key_moments]
        scores = [m['score'] for m in key_moments]
        
        axes[1, 1].scatter(times, scores, s=100, c='red', alpha=0.6)
        for i, moment in enumerate(key_moments):
            axes[1, 1].annotate(f"{moment['timestamp']:.1f}s", 
                              (times[i], scores[i]), 
                              xytext=(5, 5), 
                              textcoords='offset points',
                              fontsize=8)
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Cinematography Score')
        axes[1, 1].set_title('Key Cinematographic Moments')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = self.output_dir / f"summary_{timestamp}.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary visualization to: {summary_path}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Instagram Video Frame-by-Frame Analyzer')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--interval', type=float, default=0.1, 
                       help='Frame analysis interval in seconds (default: 0.1)')
    parser.add_argument('--output-dir', default='instagram_analysis',
                       help='Output directory (default: instagram_analysis)')
    parser.add_argument('--no-scenes', action='store_true',
                       help='Skip scene detection')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = InstagramFrameAnalyzer(args.video_path, args.output_dir)
    
    # Run analysis
    analyzer.analyze_video(interval=args.interval, 
                          detect_scenes=not args.no_scenes)
    
    # Save outputs
    analyzer.save_outputs()
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("- frames_*.jsonl: Frame-by-frame data")
    print("- analysis_*.json: Complete structured analysis")
    print("- screenplay_*.txt: Screenplay-style report")
    print("- report_*.md: Markdown report")
    print("- timeline_*.png: Timeline visualizations")
    print("- summary_*.png: Summary visualizations")
    print("- frame_*.jpg: Key frame images")


if __name__ == "__main__":
    main()