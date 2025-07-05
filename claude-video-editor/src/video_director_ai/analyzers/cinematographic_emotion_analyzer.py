import os
import cv2
import asyncio
import aiofiles
import datetime
import subprocess
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from fer import FER
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .shot_detector import ShotTypeDetector
from .motion_analyzer import MotionAnalyzer
from .color_grading_analyzer import ColorGradingAnalyzer
from ..settings import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class CinematographicEmotionAnalyzer:
    """Enhanced emotion analyzer that includes cinematographic analysis."""
    
    def __init__(self, video_path: str) -> None:
        """
        Initializes the analyzer with enhanced capabilities.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Error: Could not open video file '{video_path}'")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Initialize analyzers
        self.emotion_detector = FER(mtcnn=True)  # Use MTCNN for better face detection
        self.shot_detector = ShotTypeDetector()
        self.motion_analyzer = MotionAnalyzer()
        self.color_analyzer = ColorGradingAnalyzer()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Results storage
        self.analysis_results = []
        
    async def analyze_video(self, interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        Perform comprehensive video analysis.
        
        Args:
            interval: Analysis interval in seconds
            
        Returns:
            List of analysis results for each timestamp
        """
        logger.info(f"Starting comprehensive analysis of {self.video_path}")
        logger.info(f"Video duration: {self.total_duration:.2f} seconds, FPS: {self.fps}")
        
        frame_interval = int(self.fps * interval)
        analysis_tasks = []
        
        # Process frames at intervals
        for frame_num in range(0, self.total_frames, frame_interval):
            timestamp = frame_num / self.fps
            analysis_tasks.append(self._analyze_frame_async(frame_num, timestamp))
            
        # Process all frames in parallel
        self.analysis_results = await asyncio.gather(*analysis_tasks)
        
        # Close video capture
        self.cap.release()
        
        # Post-process results
        self._post_process_results()
        
        # Generate clips for interesting segments
        await self._generate_clips()
        
        return self.analysis_results
        
    async def _analyze_frame_async(self, frame_num: int, timestamp: float) -> Dict[str, Any]:
        """Analyze a single frame asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Read frame in thread pool
        frame = await loop.run_in_executor(self.executor, self._read_frame_sync, frame_num)
        
        if frame is None:
            return None
            
        # Run all analyses in parallel
        emotion_task = loop.run_in_executor(self.executor, self._analyze_emotions, frame)
        shot_task = loop.run_in_executor(self.executor, self.shot_detector.detect_shot_type, frame)
        motion_task = loop.run_in_executor(self.executor, self.motion_analyzer.analyze_motion, frame, timestamp)
        color_task = loop.run_in_executor(self.executor, self.color_analyzer.analyze_frame, frame)
        
        # Wait for all analyses to complete
        emotions, shot_type, motion, color_grading = await asyncio.gather(
            emotion_task, shot_task, motion_task, color_task
        )
        
        # Combine results
        result = {
            'timestamp': timestamp,
            'frame_number': frame_num,
            'emotions': emotions,
            'shot_type': shot_type,
            'motion': motion,
            'color_grading': color_grading,
            'cinematography_score': self._calculate_cinematography_score(
                emotions, shot_type, motion, color_grading
            )
        }
        
        # Add director's interpretation
        result['directors_notes'] = self._generate_directors_notes(result)
        
        logger.debug(f"Analyzed frame {frame_num} at {timestamp:.2f}s")
        
        return result
        
    def _read_frame_sync(self, frame_num: int) -> Optional[np.ndarray]:
        """Synchronously read a frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def _analyze_emotions(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze emotions in frame with enhanced context."""
        # Detect emotions
        result = self.emotion_detector.detect_emotions(frame)
        
        if not result:
            return {
                'detected': False,
                'dominant_emotion': 'none',
                'emotions': self._init_emotion_dict(),
                'num_faces': 0
            }
            
        # Aggregate emotions from all faces
        aggregated_emotions = self._init_emotion_dict()
        
        for face_data in result:
            emotions = face_data['emotions']
            for emotion, score in emotions.items():
                aggregated_emotions[emotion] += score
                
        # Normalize by number of faces
        num_faces = len(result)
        for emotion in aggregated_emotions:
            aggregated_emotions[emotion] /= num_faces
            
        # Find dominant emotion
        dominant_emotion = max(aggregated_emotions.items(), key=lambda x: x[1])[0]
        
        return {
            'detected': True,
            'dominant_emotion': dominant_emotion,
            'emotions': aggregated_emotions,
            'num_faces': num_faces,
            'face_positions': [face['box'] for face in result],
            'emotional_intensity': max(aggregated_emotions.values())
        }
        
    def _calculate_cinematography_score(self, emotions: Dict, shot_type: Dict, 
                                      motion: Dict, color_grading: Dict) -> Dict[str, float]:
        """Calculate various cinematography scores."""
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
        composition_score = 0.5  # Base score
        if shot_type.get('composition', {}).get('rule_of_thirds', {}).get('follows_rule'):
            composition_score += 0.2
        if shot_type.get('composition', {}).get('symmetry', {}).get('is_symmetric'):
            composition_score += 0.2
        if shot_type.get('composition', {}).get('leading_lines'):
            composition_score += 0.1
            
        color_score = 0.5  # Base score
        if color_grading.get('color_grading', {}).get('style') != 'Standard':
            color_score += 0.3
        if color_grading.get('film_stock', {}).get('film_stock') != 'Digital':
            color_score += 0.2
            
        scores['aesthetic_quality'] = (composition_score + color_score) / 2
        
        # Overall cinematography score
        scores['overall'] = np.mean([
            scores['emotional_impact'],
            scores['visual_dynamics'],
            scores['aesthetic_quality']
        ])
        
        return scores
        
    def _generate_directors_notes(self, analysis: Dict) -> str:
        """Generate director-style interpretation of the frame."""
        notes = []
        
        # Shot type interpretation
        shot_info = analysis['shot_type']
        shot_type = shot_info.get('shot_type', 'unknown')
        shot_desc = shot_info.get('description', '')
        
        # Motion interpretation
        motion_info = analysis['motion']
        camera_movement = motion_info.get('camera_movement', {})
        movement_type = camera_movement.get('type', 'static')
        
        # Color grading interpretation
        color_info = analysis['color_grading']
        color_style = color_info.get('color_grading', {}).get('style', 'Standard')
        color_temp = color_info.get('temperature', {}).get('temperature', 'Neutral')
        
        # Emotion interpretation
        emotion_info = analysis['emotions']
        dominant_emotion = emotion_info.get('dominant_emotion', 'neutral')
        
        # Build comprehensive notes
        if shot_type == 'extreme_close' and dominant_emotion in ['fear', 'surprise', 'anger']:
            notes.append(f"{shot_desc}. The extreme proximity amplifies the {dominant_emotion}, creating visceral viewer connection.")
        elif shot_type == 'wide' and movement_type != 'static':
            notes.append(f"{shot_desc} with {movement_type} camera movement, establishing spatial context while maintaining visual energy.")
        else:
            notes.append(shot_desc)
            
        # Add movement notes
        if movement_type in ['zoom_in', 'dolly_forward']:
            notes.append("Camera pushing in creates psychological pressure and draws viewer into the scene.")
        elif movement_type in ['zoom_out', 'dolly_backward']:
            notes.append("Camera pulling back reveals context and creates emotional distance.")
        elif 'pan' in movement_type:
            notes.append(f"Horizontal {movement_type} guides viewer attention across the scene.")
        elif camera_movement.get('shake_detection', {}).get('has_shake'):
            notes.append("Handheld camera work adds documentary realism and urgency.")
            
        # Add color notes
        if color_style == 'Teal_Orange':
            notes.append("Teal and orange color grading creates Hollywood blockbuster aesthetic with vibrant color contrast.")
        elif color_style == 'Bleach_Bypass':
            notes.append("Bleach bypass processing creates gritty, desaturated look suggesting harsh reality.")
        elif color_temp == 'Warm':
            notes.append("Warm color temperature evokes comfort, nostalgia, or heat.")
        elif color_temp == 'Cool':
            notes.append("Cool color temperature suggests isolation, technology, or nighttime.")
            
        # Add composition notes
        composition = shot_info.get('composition', {})
        if composition.get('rule_of_thirds', {}).get('follows_rule'):
            notes.append("Composition follows rule of thirds for balanced, pleasing framing.")
        if composition.get('symmetry', {}).get('is_symmetric'):
            notes.append("Symmetrical composition creates formal, controlled aesthetic.")
            
        # Combine into single interpretation
        return " ".join(notes)
        
    def _post_process_results(self) -> None:
        """Post-process results to identify patterns and key moments."""
        # Remove None results
        self.analysis_results = [r for r in self.analysis_results if r is not None]
        
        if not self.analysis_results:
            return
            
        # Identify emotional peaks
        for i, result in enumerate(self.analysis_results):
            if i == 0 or i == len(self.analysis_results) - 1:
                continue
                
            # Check for emotional peaks
            current_intensity = result['emotions'].get('emotional_intensity', 0)
            prev_intensity = self.analysis_results[i-1]['emotions'].get('emotional_intensity', 0)
            next_intensity = self.analysis_results[i+1]['emotions'].get('emotional_intensity', 0)
            
            if current_intensity > prev_intensity and current_intensity > next_intensity:
                result['is_emotional_peak'] = True
                
            # Check for shot changes
            current_shot = result['shot_type'].get('shot_type')
            prev_shot = self.analysis_results[i-1]['shot_type'].get('shot_type')
            
            if current_shot != prev_shot:
                result['shot_change'] = True
                
        # Identify sequences
        self._identify_sequences()
        
    def _identify_sequences(self) -> None:
        """Identify cinematographic sequences and patterns."""
        sequence_start = 0
        current_pattern = None
        
        for i in range(1, len(self.analysis_results)):
            prev_result = self.analysis_results[i-1]
            curr_result = self.analysis_results[i]
            
            # Detect montage sequences (rapid shot changes)
            if i > 2:
                recent_changes = sum(
                    1 for j in range(i-3, i) 
                    if self.analysis_results[j].get('shot_change', False)
                )
                if recent_changes >= 2:
                    curr_result['sequence_type'] = 'montage'
                    
            # Detect dialogue sequences (medium shots with minimal movement)
            if (prev_result['shot_type'].get('shot_type') in ['medium', 'close'] and
                curr_result['shot_type'].get('shot_type') in ['medium', 'close'] and
                prev_result['motion'].get('camera_movement', {}).get('type') == 'static' and
                curr_result['motion'].get('camera_movement', {}).get('type') == 'static'):
                curr_result['sequence_type'] = 'dialogue'
                
            # Detect action sequences (high motion + rapid cuts)
            motion_intensity = curr_result['motion'].get('motion_intensity', {}).get('intensity', 'none')
            if motion_intensity in ['high', 'extreme'] and curr_result.get('shot_change', False):
                curr_result['sequence_type'] = 'action'
                
    async def _generate_clips(self) -> None:
        """Generate clips for interesting segments."""
        interesting_segments = []
        
        # Find segments with high emotional intensity or cinematographic interest
        for i, result in enumerate(self.analysis_results):
            score = result['cinematography_score']['overall']
            
            if (score > 0.7 or 
                result.get('is_emotional_peak', False) or
                result['emotions'].get('dominant_emotion') == 'surprise'):
                
                # Calculate segment boundaries
                start_time = max(0, result['timestamp'] - 12.5)
                end_time = min(self.total_duration, result['timestamp'] + 12.5)
                
                # Ensure minimum duration
                if end_time - start_time < 25:
                    if start_time == 0:
                        end_time = min(25, self.total_duration)
                    else:
                        start_time = max(0, end_time - 25)
                        
                interesting_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'reason': self._get_clip_reason(result),
                    'score': score
                })
                
        # Merge overlapping segments
        merged_segments = self._merge_overlapping_segments(interesting_segments)
        
        # Extract clips
        await self._extract_clips(merged_segments)
        
    def _get_clip_reason(self, result: Dict) -> str:
        """Generate reason for clip extraction."""
        reasons = []
        
        if result.get('is_emotional_peak'):
            reasons.append(f"emotional_peak_{result['emotions']['dominant_emotion']}")
            
        if result['cinematography_score']['overall'] > 0.8:
            reasons.append("high_cinematography_score")
            
        if result['emotions'].get('dominant_emotion') == 'surprise':
            reasons.append("surprise_moment")
            
        if result.get('sequence_type'):
            reasons.append(f"{result['sequence_type']}_sequence")
            
        return "_".join(reasons) if reasons else "interesting_moment"
        
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping time segments."""
        if not segments:
            return []
            
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last_merged = merged[-1]
            
            # Check for overlap
            if segment['start'] <= last_merged['end']:
                # Merge segments
                last_merged['end'] = max(last_merged['end'], segment['end'])
                last_merged['score'] = max(last_merged['score'], segment['score'])
                last_merged['reason'] += f"_{segment['reason']}"
            else:
                merged.append(segment)
                
        return merged
        
    async def _extract_clips(self, segments: List[Dict]) -> None:
        """Extract video clips for segments."""
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clips_dir = Config.CLIPS_DIR / f"cinematographic_{date_str}"
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        for i, segment in enumerate(segments):
            output_path = clips_dir / f"clip_{i:03d}_{segment['reason']}.mp4"
            
            command = [
                "ffmpeg",
                "-i", self.video_path,
                "-ss", str(segment['start']),
                "-to", str(segment['end']),
                "-c:v", "copy",
                "-c:a", "copy",
                "-y",  # Overwrite if exists
                str(output_path)
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"Extracted clip: {output_path}")
                else:
                    logger.error(f"Failed to extract clip {i}")
                    
            except Exception as e:
                logger.error(f"Error extracting clip: {e}")
                
    @staticmethod
    def _init_emotion_dict() -> Dict[str, float]:
        """Initialize emotion dictionary."""
        return {emotion: 0.0 for emotion in Config.EMOTIONS}
        
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'executor'):
            self.executor.shutdown()