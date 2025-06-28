import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import datetime
import numpy as np
import cv2

from utils.video_downloader import VideoDownloader
from analyzers.color_grading_analyzer import ColorGradingAnalyzer
from analyzers.shot_detector import ShotTypeDetector
from analyzers.motion_analyzer import MotionAnalyzer
from settings import Config
from visualizer import CinematographyVisualizer

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class SimplifiedEmotionAnalyzer:
    """Simplified emotion analyzer without deep learning dependencies."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def analyze_emotions(self, frame):
        """Basic emotion analysis based on face detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Simulate emotion detection based on image characteristics
        if len(faces) > 0:
            # Use color and brightness as proxy for emotion
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_brightness = np.mean(hsv[:, :, 2])
            avg_saturation = np.mean(hsv[:, :, 1])
            
            # Simple heuristic for emotion
            if avg_brightness > 180:
                dominant_emotion = "happy"
                intensity = 0.7
            elif avg_brightness < 80:
                dominant_emotion = "sad"
                intensity = 0.6
            elif avg_saturation > 150:
                dominant_emotion = "surprise"
                intensity = 0.8
            else:
                dominant_emotion = "neutral"
                intensity = 0.5
                
            return {
                'detected': True,
                'dominant_emotion': dominant_emotion,
                'emotions': {emotion: 0.1 for emotion in Config.EMOTIONS},
                'num_faces': len(faces),
                'emotional_intensity': intensity
            }
        
        return {
            'detected': False,
            'dominant_emotion': 'neutral',
            'emotions': {emotion: 0.0 for emotion in Config.EMOTIONS},
            'num_faces': 0,
            'emotional_intensity': 0.0
        }


class DirectorBrainLite:
    """Lite version of DirectorBrain without heavy ML dependencies."""
    
    def __init__(self):
        self.downloader = VideoDownloader()
        self.visualizer = CinematographyVisualizer()
        self.results = {}
        
    async def analyze_video(self, video_input: str, output_format: str = 'all') -> Dict[str, Any]:
        """Perform video analysis without Whisper/LLaVA."""
        logger.info(f"Starting DirectorBrain Lite analysis for: {video_input}")
        
        # Step 1: Get video file
        video_path = await self._get_video_file(video_input)
        
        # Step 2: Extract metadata
        metadata = await self._extract_metadata(video_path)
        
        # Step 3: Analyze cinematography
        cinematography = await self._analyze_cinematography(video_path)
        
        # Step 4: Detect scenes
        scenes = await self._detect_scenes(video_path)
        
        # Step 5: Synthesize results
        self.results = {
            'metadata': metadata,
            'cinematography': cinematography,
            'scenes': scenes,
            'synthesis': self._synthesize_analysis(cinematography, scenes),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Step 6: Generate outputs
        await self._generate_outputs(output_format)
        
        logger.info("DirectorBrain Lite analysis complete")
        
        return self.results
        
    async def _get_video_file(self, video_input: str) -> Path:
        """Get video file from URL or verify local path."""
        if video_input.startswith(('http://', 'https://', 'www.')):
            logger.info("Downloading video from URL")
            download_info = self.downloader.download(video_input)
            return Path(download_info['file_path'])
        else:
            path = Path(video_input)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            return path
            
    async def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata."""
        cap = cv2.VideoCapture(str(video_path))
        
        metadata = {
            'filename': video_path.name,
            'filepath': str(video_path),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'file_size': video_path.stat().st_size
        }
        
        cap.release()
        return metadata
        
    async def _analyze_cinematography(self, video_path: Path) -> List[Dict[str, Any]]:
        """Run cinematographic analysis."""
        logger.info("Running cinematographic analysis")
        
        # Initialize analyzers
        emotion_analyzer = SimplifiedEmotionAnalyzer()
        shot_detector = ShotTypeDetector()
        motion_analyzer = MotionAnalyzer()
        color_analyzer = ColorGradingAnalyzer()
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        frame_interval = int(fps * Config.FRAME_INTERVAL)
        
        for frame_num in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            timestamp = frame_num / fps
            
            # Run analyses
            emotions = emotion_analyzer.analyze_emotions(frame)
            shot_type = shot_detector.detect_shot_type(frame)
            motion = motion_analyzer.analyze_motion(frame, timestamp)
            color_grading = color_analyzer.analyze_frame(frame)
            
            # Calculate cinematography score
            cinematography_score = self._calculate_cinematography_score(
                emotions, shot_type, motion, color_grading
            )
            
            result = {
                'timestamp': timestamp,
                'frame_number': frame_num,
                'emotions': emotions,
                'shot_type': shot_type,
                'motion': motion,
                'color_grading': color_grading,
                'cinematography_score': cinematography_score,
                'directors_notes': self._generate_directors_notes({
                    'emotions': emotions,
                    'shot_type': shot_type,
                    'motion': motion,
                    'color_grading': color_grading
                })
            }
            
            results.append(result)
            logger.info(f"Analyzed frame {frame_num} at {timestamp:.2f}s")
            
        cap.release()
        return results
        
    async def _detect_scenes(self, video_path: Path) -> List[Dict[str, Any]]:
        """Basic scene detection using frame differences."""
        logger.info("Detecting scene boundaries")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        scenes = []
        prev_frame = None
        scene_start = 0
        scene_threshold = 30  # Adjust as needed
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, frame)
                diff_score = np.mean(diff)
                
                # Scene change detected
                if diff_score > scene_threshold:
                    scenes.append({
                        'scene_number': len(scenes) + 1,
                        'start_time': scene_start / fps,
                        'end_time': frame_count / fps,
                        'duration': (frame_count - scene_start) / fps,
                        'start_frame': scene_start,
                        'end_frame': frame_count
                    })
                    scene_start = frame_count
                    
            prev_frame = frame.copy()
            frame_count += 1
            
        # Add final scene
        if scene_start < frame_count:
            scenes.append({
                'scene_number': len(scenes) + 1,
                'start_time': scene_start / fps,
                'end_time': frame_count / fps,
                'duration': (frame_count - scene_start) / fps,
                'start_frame': scene_start,
                'end_frame': frame_count
            })
            
        cap.release()
        return scenes
        
    def _calculate_cinematography_score(self, emotions, shot_type, motion, color_grading):
        """Calculate cinematography scores."""
        scores = {}
        
        # Emotional impact score
        emotional_intensity = emotions.get('emotional_intensity', 0)
        shot_intimacy = {
            'extreme_close': 1.0, 'close': 0.8, 'medium': 0.5, 
            'wide': 0.3, 'extreme_wide': 0.1
        }.get(shot_type.get('shot_type', 'medium'), 0.5)
        
        scores['emotional_impact'] = emotional_intensity * shot_intimacy
        
        # Visual dynamics score
        motion_intensity = {
            'extreme': 1.0, 'high': 0.8, 'moderate': 0.5, 
            'minimal': 0.2, 'none': 0.0
        }.get(motion.get('motion_intensity', {}).get('intensity', 'none'), 0.0)
        
        camera_movement_score = 0.0 if motion.get('camera_movement', {}).get('type') == 'static' else 0.5
        
        scores['visual_dynamics'] = (motion_intensity + camera_movement_score) / 2
        
        # Aesthetic score
        composition_score = 0.5
        if shot_type.get('composition', {}).get('rule_of_thirds', {}).get('follows_rule'):
            composition_score += 0.2
        if shot_type.get('composition', {}).get('symmetry', {}).get('is_symmetric'):
            composition_score += 0.2
            
        color_score = 0.5
        if color_grading.get('color_grading', {}).get('style') != 'Standard':
            color_score += 0.3
            
        scores['aesthetic_quality'] = (composition_score + color_score) / 2
        
        # Overall score
        scores['overall'] = np.mean([
            scores['emotional_impact'],
            scores['visual_dynamics'],
            scores['aesthetic_quality']
        ])
        
        return scores
        
    def _generate_directors_notes(self, analysis):
        """Generate director-style interpretation."""
        notes = []
        
        shot_info = analysis['shot_type']
        shot_type = shot_info.get('shot_type', 'unknown')
        shot_desc = shot_info.get('description', '')
        
        motion_info = analysis['motion']
        movement_type = motion_info.get('camera_movement', {}).get('type', 'static')
        
        color_info = analysis['color_grading']
        color_style = color_info.get('color_grading', {}).get('style', 'Standard')
        
        # Build notes
        notes.append(shot_desc)
        
        if movement_type != 'static':
            notes.append(f"Camera {movement_type} adds dynamic energy.")
            
        if color_style != 'Standard':
            notes.append(f"{color_style} color grading enhances mood.")
            
        return " ".join(notes)
        
    def _synthesize_analysis(self, cinematography, scenes):
        """Synthesize analysis results."""
        return {
            'total_frames_analyzed': len(cinematography),
            'total_scenes': len(scenes),
            'average_scene_duration': np.mean([s['duration'] for s in scenes]) if scenes else 0,
            'cinematographic_style': self._identify_style(cinematography),
            'key_moments': self._identify_key_moments(cinematography)
        }
        
    def _identify_style(self, cinematography):
        """Identify cinematographic style."""
        shot_counts = {}
        for frame in cinematography:
            shot = frame['shot_type']['shot_type']
            shot_counts[shot] = shot_counts.get(shot, 0) + 1
            
        dominant_shot = max(shot_counts.items(), key=lambda x: x[1])[0] if shot_counts else 'unknown'
        
        return {
            'dominant_shot_type': dominant_shot,
            'shot_distribution': shot_counts
        }
        
    def _identify_key_moments(self, cinematography):
        """Identify key moments based on scores."""
        key_moments = []
        
        for frame in cinematography:
            if frame['cinematography_score']['overall'] > 0.7:
                key_moments.append({
                    'timestamp': frame['timestamp'],
                    'score': frame['cinematography_score']['overall'],
                    'description': frame['directors_notes']
                })
                
        return sorted(key_moments, key=lambda x: x['score'], reverse=True)[:5]
        
    async def _generate_outputs(self, output_format):
        """Generate output files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.ANALYSIS_DIR / f"analysis_lite_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        if output_format in ['json', 'all']:
            output_path = output_dir / "analysis.json"
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"JSON saved to: {output_path}")
            
        # Generate visualization
        try:
            self.visualizer.create_comprehensive_visualization(
                self.results, 
                output_dir / "visualization.png"
            )
            logger.info(f"Visualization saved to: {output_dir / 'visualization.png'}")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            
        logger.info(f"All outputs saved to: {output_dir}")
        
        return output_dir