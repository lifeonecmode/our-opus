import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import datetime
import whisper
import numpy as np
from scenedetect import detect, ContentDetector, AdaptiveDetector

from .analyzers.cinematographic_emotion_analyzer import CinematographicEmotionAnalyzer
from .utils.video_downloader import VideoDownloader
from .settings import Config
from .visualizer import CinematographyVisualizer

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DirectorBrain:
    """Master controller orchestrating all video analysis components."""
    
    def __init__(self):
        self.downloader = VideoDownloader()
        self.whisper_model = None
        self.visualizer = CinematographyVisualizer()
        self.results = {}
        
    async def analyze_video(self, video_input: str, output_format: str = 'all') -> Dict[str, Any]:
        """
        Perform complete video analysis.
        
        Args:
            video_input: URL or local path to video
            output_format: Output format (json, markdown, csv, all)
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting DirectorBrain analysis for: {video_input}")
        
        # Step 1: Get video file
        video_path = await self._get_video_file(video_input)
        
        # Step 2: Run all analyses in parallel
        analysis_tasks = [
            self._analyze_cinematography(video_path),
            self._transcribe_audio(video_path),
            self._detect_scenes(video_path),
            self._extract_metadata(video_path)
        ]
        
        cinematography, transcription, scenes, metadata = await asyncio.gather(*analysis_tasks)
        
        # Step 3: Synthesize results
        self.results = {
            'metadata': metadata,
            'cinematography': cinematography,
            'transcription': transcription,
            'scenes': scenes,
            'synthesis': self._synthesize_analysis(cinematography, transcription, scenes),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Step 4: Generate outputs
        await self._generate_outputs(output_format)
        
        logger.info("DirectorBrain analysis complete")
        
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
            
    async def _analyze_cinematography(self, video_path: Path) -> List[Dict[str, Any]]:
        """Run cinematographic analysis."""
        logger.info("Running cinematographic analysis")
        
        analyzer = CinematographicEmotionAnalyzer(str(video_path))
        results = await analyzer.analyze_video(interval=Config.FRAME_INTERVAL)
        
        return results
        
    async def _transcribe_audio(self, video_path: Path) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        logger.info("Transcribing audio with Whisper")
        
        # Load Whisper model (cached)
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
            
        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.whisper_model.transcribe,
            str(video_path),
            {'language': 'en', 'task': 'transcribe'}
        )
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('confidence', 1.0)
            })
            
        return {
            'text': result.get('text', ''),
            'segments': segments,
            'language': result.get('language', 'en')
        }
        
    async def _detect_scenes(self, video_path: Path) -> List[Dict[str, Any]]:
        """Detect scene boundaries."""
        logger.info("Detecting scene boundaries")
        
        # Use both content and adaptive detection
        scene_list = await asyncio.get_event_loop().run_in_executor(
            None,
            detect,
            str(video_path),
            ContentDetector(threshold=Config.SCENE_THRESHOLD),
            show_progress=True
        )
        
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            scenes.append({
                'scene_number': i + 1,
                'start_time': start.get_seconds(),
                'end_time': end.get_seconds(),
                'duration': (end - start).get_seconds(),
                'start_frame': start.get_frames(),
                'end_frame': end.get_frames()
            })
            
        return scenes
        
    async def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata."""
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        
        metadata = {
            'filename': video_path.name,
            'filepath': str(video_path),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': cap.get(cv2.CAP_PROP_FOURCC),
            'file_size': video_path.stat().st_size
        }
        
        cap.release()
        
        return metadata
        
    def _synthesize_analysis(self, cinematography: List[Dict], 
                           transcription: Dict, 
                           scenes: List[Dict]) -> Dict[str, Any]:
        """Synthesize all analyses into comprehensive insights."""
        synthesis = {
            'visual_narrative': self._analyze_visual_narrative(cinematography, scenes),
            'emotional_arc': self._analyze_emotional_arc(cinematography),
            'cinematographic_style': self._identify_cinematographic_style(cinematography),
            'pacing': self._analyze_pacing(cinematography, scenes),
            'key_moments': self._identify_key_moments(cinematography, transcription),
            'technical_analysis': self._technical_summary(cinematography),
            'directorial_patterns': self._identify_directorial_patterns(cinematography)
        }
        
        # Generate overall assessment
        synthesis['overall_assessment'] = self._generate_overall_assessment(synthesis)
        
        return synthesis
        
    def _analyze_visual_narrative(self, cinematography: List[Dict], 
                                scenes: List[Dict]) -> Dict[str, Any]:
        """Analyze the visual storytelling structure."""
        narrative = {
            'structure': 'unknown',
            'acts': [],
            'turning_points': [],
            'visual_motifs': []
        }
        
        # Analyze three-act structure based on emotional and visual intensity
        total_duration = cinematography[-1]['timestamp'] if cinematography else 0
        act_boundaries = [total_duration * 0.25, total_duration * 0.75]
        
        acts = [
            {'start': 0, 'end': act_boundaries[0], 'frames': []},
            {'start': act_boundaries[0], 'end': act_boundaries[1], 'frames': []},
            {'start': act_boundaries[1], 'end': total_duration, 'frames': []}
        ]
        
        # Assign frames to acts
        for frame in cinematography:
            timestamp = frame['timestamp']
            for i, act in enumerate(acts):
                if act['start'] <= timestamp < act['end']:
                    act['frames'].append(frame)
                    break
                    
        # Analyze each act
        for i, act in enumerate(acts):
            if not act['frames']:
                continue
                
            # Calculate average scores
            emotional_scores = [f['cinematography_score']['emotional_impact'] 
                              for f in act['frames']]
            dynamic_scores = [f['cinematography_score']['visual_dynamics'] 
                            for f in act['frames']]
            
            act_analysis = {
                'act_number': i + 1,
                'duration': act['end'] - act['start'],
                'avg_emotional_intensity': np.mean(emotional_scores),
                'avg_visual_dynamics': np.mean(dynamic_scores),
                'dominant_shots': self._get_dominant_shots(act['frames']),
                'dominant_emotions': self._get_dominant_emotions(act['frames'])
            }
            
            narrative['acts'].append(act_analysis)
            
        # Identify turning points
        for i in range(len(scenes) - 1):
            if scenes[i+1]['scene_number'] - scenes[i]['scene_number'] == 1:
                # Check for significant changes
                scene_start = scenes[i+1]['start_time']
                
                # Find corresponding cinematography data
                for frame in cinematography:
                    if abs(frame['timestamp'] - scene_start) < 1:
                        if frame.get('shot_change') or frame.get('is_emotional_peak'):
                            narrative['turning_points'].append({
                                'timestamp': scene_start,
                                'type': 'scene_transition',
                                'description': frame['directors_notes']
                            })
                            
        return narrative
        
    def _analyze_emotional_arc(self, cinematography: List[Dict]) -> Dict[str, Any]:
        """Analyze the emotional journey throughout the video."""
        timestamps = []
        emotional_intensities = []
        dominant_emotions = []
        
        for frame in cinematography:
            timestamps.append(frame['timestamp'])
            emotional_intensities.append(
                frame['emotions'].get('emotional_intensity', 0)
            )
            dominant_emotions.append(
                frame['emotions'].get('dominant_emotion', 'neutral')
            )
            
        # Identify emotional peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(emotional_intensities) - 1):
            if (emotional_intensities[i] > emotional_intensities[i-1] and 
                emotional_intensities[i] > emotional_intensities[i+1]):
                peaks.append({
                    'timestamp': timestamps[i],
                    'intensity': emotional_intensities[i],
                    'emotion': dominant_emotions[i]
                })
            elif (emotional_intensities[i] < emotional_intensities[i-1] and 
                  emotional_intensities[i] < emotional_intensities[i+1]):
                valleys.append({
                    'timestamp': timestamps[i],
                    'intensity': emotional_intensities[i],
                    'emotion': dominant_emotions[i]
                })
                
        # Determine overall arc pattern
        if len(peaks) == 1 and peaks[0]['timestamp'] > timestamps[-1] * 0.6:
            arc_type = "rising_climax"
        elif len(peaks) > 3:
            arc_type = "rollercoaster"
        elif np.std(emotional_intensities) < 0.1:
            arc_type = "flat"
        else:
            arc_type = "complex"
            
        return {
            'arc_type': arc_type,
            'peaks': peaks[:5],  # Top 5 peaks
            'valleys': valleys[:5],  # Bottom 5 valleys
            'average_intensity': np.mean(emotional_intensities),
            'intensity_variance': np.var(emotional_intensities),
            'emotion_distribution': self._calculate_emotion_distribution(cinematography)
        }
        
    def _identify_cinematographic_style(self, cinematography: List[Dict]) -> Dict[str, Any]:
        """Identify the overall cinematographic style."""
        style_indicators = {
            'shot_distribution': {},
            'movement_patterns': {},
            'color_styles': {},
            'composition_preferences': {}
        }
        
        # Analyze shot types
        shot_types = [f['shot_type']['shot_type'] for f in cinematography]
        for shot in shot_types:
            style_indicators['shot_distribution'][shot] = \
                style_indicators['shot_distribution'].get(shot, 0) + 1
                
        # Analyze camera movements
        movements = [f['motion']['camera_movement']['type'] for f in cinematography]
        for movement in movements:
            style_indicators['movement_patterns'][movement] = \
                style_indicators['movement_patterns'].get(movement, 0) + 1
                
        # Analyze color grading
        color_styles = [f['color_grading']['color_grading']['style'] for f in cinematography]
        for style in color_styles:
            style_indicators['color_styles'][style] = \
                style_indicators['color_styles'].get(style, 0) + 1
                
        # Determine predominant style
        if style_indicators['movement_patterns'].get('static', 0) > len(cinematography) * 0.7:
            movement_style = "static/observational"
        elif any(m in style_indicators['movement_patterns'] for m in ['pan_left', 'pan_right', 'tilt_up', 'tilt_down']):
            movement_style = "dynamic/exploratory"
        else:
            movement_style = "mixed"
            
        # Color style
        dominant_color = max(style_indicators['color_styles'].items(), 
                           key=lambda x: x[1])[0] if style_indicators['color_styles'] else 'Standard'
        
        # Shot style
        if style_indicators['shot_distribution'].get('close', 0) + \
           style_indicators['shot_distribution'].get('extreme_close', 0) > len(cinematography) * 0.5:
            shot_style = "intimate"
        elif style_indicators['shot_distribution'].get('wide', 0) + \
             style_indicators['shot_distribution'].get('extreme_wide', 0) > len(cinematography) * 0.5:
            shot_style = "epic/landscape"
        else:
            shot_style = "balanced"
            
        return {
            'movement_style': movement_style,
            'shot_style': shot_style,
            'color_palette': dominant_color,
            'visual_rhythm': self._analyze_visual_rhythm(cinematography),
            'stylistic_influences': self._identify_influences(style_indicators)
        }
        
    def _analyze_pacing(self, cinematography: List[Dict], scenes: List[Dict]) -> Dict[str, Any]:
        """Analyze the pacing of the video."""
        # Calculate average shot duration
        shot_durations = []
        for i in range(len(scenes)):
            shot_durations.append(scenes[i]['duration'])
            
        # Analyze motion intensity over time
        motion_intensities = []
        for frame in cinematography:
            intensity = frame['motion']['motion_intensity']['intensity']
            motion_intensities.append({
                'none': 0, 'minimal': 0.25, 'moderate': 0.5, 
                'high': 0.75, 'extreme': 1.0
            }.get(intensity, 0))
            
        # Determine pacing type
        avg_shot_duration = np.mean(shot_durations) if shot_durations else 0
        avg_motion = np.mean(motion_intensities) if motion_intensities else 0
        
        if avg_shot_duration < 3 and avg_motion > 0.5:
            pacing_type = "frenetic"
        elif avg_shot_duration > 10 and avg_motion < 0.3:
            pacing_type = "contemplative"
        elif avg_shot_duration < 5:
            pacing_type = "brisk"
        else:
            pacing_type = "measured"
            
        return {
            'pacing_type': pacing_type,
            'average_shot_duration': avg_shot_duration,
            'shortest_shot': min(shot_durations) if shot_durations else 0,
            'longest_shot': max(shot_durations) if shot_durations else 0,
            'motion_intensity_average': avg_motion,
            'rhythm_variance': np.var(shot_durations) if shot_durations else 0
        }
        
    def _identify_key_moments(self, cinematography: List[Dict], 
                            transcription: Dict) -> List[Dict[str, Any]]:
        """Identify the most significant moments in the video."""
        key_moments = []
        
        # Find moments with high cinematography scores
        for frame in cinematography:
            if frame['cinematography_score']['overall'] > 0.8:
                # Find corresponding dialogue
                dialogue = ""
                for segment in transcription.get('segments', []):
                    if segment['start'] <= frame['timestamp'] <= segment['end']:
                        dialogue = segment['text']
                        break
                        
                key_moments.append({
                    'timestamp': frame['timestamp'],
                    'type': 'high_cinematography',
                    'description': frame['directors_notes'],
                    'dialogue': dialogue,
                    'score': frame['cinematography_score']['overall']
                })
                
        # Find emotional peaks
        for frame in cinematography:
            if frame.get('is_emotional_peak'):
                key_moments.append({
                    'timestamp': frame['timestamp'],
                    'type': 'emotional_peak',
                    'emotion': frame['emotions']['dominant_emotion'],
                    'description': frame['directors_notes'],
                    'score': frame['emotions']['emotional_intensity']
                })
                
        # Sort by score and timestamp
        key_moments.sort(key=lambda x: (x['score'], x['timestamp']), reverse=True)
        
        return key_moments[:10]  # Top 10 moments
        
    def _technical_summary(self, cinematography: List[Dict]) -> Dict[str, Any]:
        """Generate technical summary of cinematographic techniques."""
        techniques_used = {
            'camera_movements': set(),
            'shot_types': set(),
            'color_grades': set(),
            'composition_techniques': set()
        }
        
        for frame in cinematography:
            techniques_used['camera_movements'].add(
                frame['motion']['camera_movement']['type']
            )
            techniques_used['shot_types'].add(
                frame['shot_type']['shot_type']
            )
            techniques_used['color_grades'].add(
                frame['color_grading']['color_grading']['style']
            )
            
            # Check composition techniques
            comp = frame['shot_type'].get('composition', {})
            if comp.get('rule_of_thirds', {}).get('follows_rule'):
                techniques_used['composition_techniques'].add('rule_of_thirds')
            if comp.get('symmetry', {}).get('is_symmetric'):
                techniques_used['composition_techniques'].add('symmetry')
            if comp.get('leading_lines'):
                techniques_used['composition_techniques'].add('leading_lines')
                
        # Convert sets to lists for JSON serialization
        for key in techniques_used:
            techniques_used[key] = list(techniques_used[key])
            
        return techniques_used
        
    def _identify_directorial_patterns(self, cinematography: List[Dict]) -> Dict[str, Any]:
        """Identify recurring directorial patterns and techniques."""
        patterns = {
            'recurring_sequences': [],
            'visual_motifs': [],
            'stylistic_consistency': 0.0
        }
        
        # Look for recurring shot patterns
        shot_sequence = [f['shot_type']['shot_type'] for f in cinematography]
        
        # Find repeating patterns of length 2-4
        for pattern_length in range(2, 5):
            for i in range(len(shot_sequence) - pattern_length * 2):
                pattern = shot_sequence[i:i+pattern_length]
                
                # Check if pattern repeats
                for j in range(i + pattern_length, len(shot_sequence) - pattern_length):
                    if shot_sequence[j:j+pattern_length] == pattern:
                        patterns['recurring_sequences'].append({
                            'pattern': ' -> '.join(pattern),
                            'occurrences': 2,
                            'significance': 'Establishes visual rhythm'
                        })
                        
        # Calculate stylistic consistency
        color_styles = [f['color_grading']['color_grading']['style'] for f in cinematography]
        if color_styles:
            most_common_style = max(set(color_styles), key=color_styles.count)
            consistency = color_styles.count(most_common_style) / len(color_styles)
            patterns['stylistic_consistency'] = consistency
            
        return patterns
        
    def _generate_overall_assessment(self, synthesis: Dict[str, Any]) -> str:
        """Generate a comprehensive assessment of the video's cinematography."""
        assessment_parts = []
        
        # Visual narrative assessment
        narrative = synthesis['visual_narrative']
        assessment_parts.append(
            f"The video follows a {narrative.get('structure', 'complex')} narrative structure"
        )
        
        # Emotional arc
        emotional = synthesis['emotional_arc']
        assessment_parts.append(
            f"with a {emotional['arc_type']} emotional arc"
        )
        
        # Cinematographic style
        style = synthesis['cinematographic_style']
        assessment_parts.append(
            f"The cinematographic style is characterized by {style['shot_style']} framing, "
            f"{style['movement_style']} camera work, and {style['color_palette']} color grading"
        )
        
        # Pacing
        pacing = synthesis['pacing']
        assessment_parts.append(
            f"The {pacing['pacing_type']} pacing "
            f"(average shot duration: {pacing['average_shot_duration']:.1f}s) "
            f"creates a {style['visual_rhythm']} visual rhythm"
        )
        
        # Technical proficiency
        if synthesis['directorial_patterns']['stylistic_consistency'] > 0.7:
            assessment_parts.append(
                "The high stylistic consistency suggests intentional directorial vision"
            )
            
        return ". ".join(assessment_parts) + "."
        
    async def _generate_outputs(self, output_format: str) -> None:
        """Generate output files in requested formats."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.ANALYSIS_DIR / f"analysis_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate outputs based on format
        if output_format in ['json', 'all']:
            await self._save_json(output_dir)
            
        if output_format in ['markdown', 'all']:
            await self._save_markdown(output_dir)
            
        if output_format in ['csv', 'all']:
            await self._save_csv(output_dir)
            
        if output_format in ['video', 'all']:
            await self._generate_annotated_video(output_dir)
            
        # Generate visualization
        self.visualizer.create_comprehensive_visualization(
            self.results, 
            output_dir / "visualization.png"
        )
        
        logger.info(f"Outputs saved to: {output_dir}")
        
    async def _save_json(self, output_dir: Path) -> None:
        """Save results as JSON."""
        output_path = output_dir / "analysis.json"
        
        async with asyncio.Lock():
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
                
        logger.info(f"JSON output saved: {output_path}")
        
    async def _save_markdown(self, output_dir: Path) -> None:
        """Generate markdown report."""
        output_path = output_dir / "analysis_report.md"
        
        md_content = self._generate_markdown_report()
        
        async with asyncio.Lock():
            with open(output_path, 'w') as f:
                f.write(md_content)
                
        logger.info(f"Markdown report saved: {output_path}")
        
    def _generate_markdown_report(self) -> str:
        """Generate detailed markdown report."""
        md = []
        
        # Header
        md.append("# Video Director Analysis Report")
        md.append(f"\nGenerated: {self.results['timestamp']}\n")
        
        # Metadata
        meta = self.results['metadata']
        md.append("## Video Information")
        md.append(f"- **File**: {meta['filename']}")
        md.append(f"- **Duration**: {meta['duration']:.2f} seconds")
        md.append(f"- **Resolution**: {meta['width']}x{meta['height']}")
        md.append(f"- **FPS**: {meta['fps']}")
        md.append("")
        
        # Overall Assessment
        md.append("## Overall Assessment")
        md.append(self.results['synthesis']['overall_assessment'])
        md.append("")
        
        # Cinematographic Style
        style = self.results['synthesis']['cinematographic_style']
        md.append("## Cinematographic Style")
        md.append(f"- **Shot Style**: {style['shot_style']}")
        md.append(f"- **Movement Style**: {style['movement_style']}")
        md.append(f"- **Color Palette**: {style['color_palette']}")
        md.append(f"- **Visual Rhythm**: {style['visual_rhythm']}")
        md.append("")
        
        # Emotional Arc
        emotional = self.results['synthesis']['emotional_arc']
        md.append("## Emotional Journey")
        md.append(f"- **Arc Type**: {emotional['arc_type']}")
        md.append(f"- **Average Intensity**: {emotional['average_intensity']:.2f}")
        md.append("\n### Emotional Peaks")
        for peak in emotional['peaks'][:3]:
            md.append(f"- **{peak['timestamp']:.1f}s**: {peak['emotion']} "
                     f"(intensity: {peak['intensity']:.2f})")
        md.append("")
        
        # Key Moments
        md.append("## Key Moments")
        for i, moment in enumerate(self.results['synthesis']['key_moments'][:5], 1):
            md.append(f"\n### {i}. {moment['timestamp']:.1f}s - {moment['type']}")
            md.append(f"{moment['description']}")
            if moment.get('dialogue'):
                md.append(f"\n> *\"{moment['dialogue']}\"*")
        md.append("")
        
        # Technical Analysis
        tech = self.results['synthesis']['technical_analysis']
        md.append("## Technical Analysis")
        md.append("\n### Camera Techniques Used")
        md.append("- **Movements**: " + ", ".join(tech['camera_movements']))
        md.append("- **Shot Types**: " + ", ".join(tech['shot_types']))
        md.append("- **Color Grades**: " + ", ".join(tech['color_grades']))
        md.append("- **Composition**: " + ", ".join(tech['composition_techniques']))
        
        return "\n".join(md)
        
    async def _save_csv(self, output_dir: Path) -> None:
        """Save timeline data as CSV."""
        import pandas as pd
        
        # Create timeline dataframe
        timeline_data = []
        
        for frame in self.results['cinematography']:
            row = {
                'timestamp': frame['timestamp'],
                'shot_type': frame['shot_type']['shot_type'],
                'camera_movement': frame['motion']['camera_movement']['type'],
                'motion_intensity': frame['motion']['motion_intensity']['intensity'],
                'dominant_emotion': frame['emotions']['dominant_emotion'],
                'emotional_intensity': frame['emotions']['emotional_intensity'],
                'color_grading': frame['color_grading']['color_grading']['style'],
                'overall_score': frame['cinematography_score']['overall']
            }
            timeline_data.append(row)
            
        df = pd.DataFrame(timeline_data)
        csv_path = output_dir / "timeline_analysis.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV timeline saved: {csv_path}")
        
    async def _generate_annotated_video(self, output_dir: Path) -> None:
        """Generate video with analysis overlay (placeholder for now)."""
        # This would require significant additional implementation
        # For now, just log the intention
        logger.info("Annotated video generation not yet implemented")
        
    # Helper methods
    def _get_dominant_shots(self, frames: List[Dict]) -> List[str]:
        """Get most common shot types."""
        shot_counts = {}
        for frame in frames:
            shot = frame['shot_type']['shot_type']
            shot_counts[shot] = shot_counts.get(shot, 0) + 1
            
        # Sort by count
        sorted_shots = sorted(shot_counts.items(), key=lambda x: x[1], reverse=True)
        return [shot[0] for shot in sorted_shots[:3]]
        
    def _get_dominant_emotions(self, frames: List[Dict]) -> List[str]:
        """Get most common emotions."""
        emotion_counts = {}
        for frame in frames:
            emotion = frame['emotions']['dominant_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return [emotion[0] for emotion in sorted_emotions[:3]]
        
    def _calculate_emotion_distribution(self, cinematography: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of emotions throughout video."""
        emotion_counts = {emotion: 0 for emotion in Config.EMOTIONS}
        
        for frame in cinematography:
            emotion = frame['emotions'].get('dominant_emotion', 'neutral')
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
                
        # Convert to percentages
        total = sum(emotion_counts.values())
        if total > 0:
            for emotion in emotion_counts:
                emotion_counts[emotion] = emotion_counts[emotion] / total * 100
                
        return emotion_counts
        
    def _analyze_visual_rhythm(self, cinematography: List[Dict]) -> str:
        """Analyze the visual rhythm of the video."""
        # Look at shot changes and motion patterns
        shot_changes = sum(1 for f in cinematography if f.get('shot_change', False))
        
        change_rate = shot_changes / len(cinematography) if cinematography else 0
        
        if change_rate > 0.3:
            return "staccato"
        elif change_rate > 0.15:
            return "dynamic"
        elif change_rate > 0.05:
            return "flowing"
        else:
            return "contemplative"
            
    def _identify_influences(self, style_indicators: Dict) -> List[str]:
        """Identify potential stylistic influences."""
        influences = []
        
        # Check for specific director signatures
        if style_indicators['color_styles'].get('Teal_Orange', 0) > 5:
            influences.append("Michael Bay / Modern Blockbuster")
            
        if style_indicators['color_styles'].get('Bleach_Bypass', 0) > 3:
            influences.append("David Fincher / Neo-noir")
            
        if style_indicators['shot_distribution'].get('extreme_close', 0) > 5:
            influences.append("Sergio Leone / Dramatic Tension")
            
        if style_indicators['movement_patterns'].get('static', 0) > 10:
            influences.append("Yasujirō Ozu / Contemplative Cinema")
            
        return influences[:3]  # Top 3 influences