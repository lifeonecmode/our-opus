#!/usr/bin/env python3
"""
Full cinematographic analysis of Instagram reel
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configure style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CompleteCinematographicAnalyzer:
    """Complete cinematographic analysis with all features."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps
        
        # Initialize analyzers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Motion tracking
        self.prev_frame = None
        self.prev_gray = None
        
    def analyze_complete(self, interval=1.0):
        """Run complete analysis."""
        results = []
        frame_interval = int(self.fps * interval)
        
        print(f"\nAnalyzing {self.total_frames} frames (every {interval}s)...")
        
        for frame_num in range(0, self.total_frames, frame_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            timestamp = frame_num / self.fps
            
            # Run all analyses
            analysis = {
                'timestamp': timestamp,
                'frame_number': frame_num,
                'shot_type': self._analyze_shot_type(frame),
                'camera_angle': self._detect_camera_angle(frame),
                'motion': self._analyze_motion(frame),
                'color_grading': self._analyze_color_grading(frame),
                'composition': self._analyze_composition(frame),
                'depth_of_field': self._estimate_depth_of_field(frame),
                'emotions': self._analyze_emotions(frame)
            }
            
            # Calculate scores
            analysis['cinematography_score'] = self._calculate_scores(analysis)
            
            # Generate director's notes
            analysis['directors_notes'] = self._generate_directors_notes(analysis)
            
            results.append(analysis)
            
            print(f"Frame {frame_num:5d} ({timestamp:6.2f}s): {analysis['shot_type']['type']:12s} | "
                  f"{analysis['motion']['camera_movement']:12s} | {analysis['color_grading']['style']:15s} | "
                  f"Score: {analysis['cinematography_score']['overall']:.2f}")
            
        self.cap.release()
        return results
        
    def _analyze_shot_type(self, frame):
        """Detect shot type based on face/body detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
        
        frame_area = frame.shape[0] * frame.shape[1]
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_area = largest_face[2] * largest_face[3]
            face_ratio = face_area / frame_area
            
            if face_ratio > 0.6:
                return {'type': 'extreme_close', 'ratio': face_ratio, 'confidence': 0.9}
            elif face_ratio > 0.3:
                return {'type': 'close', 'ratio': face_ratio, 'confidence': 0.9}
            elif face_ratio > 0.1:
                return {'type': 'medium', 'ratio': face_ratio, 'confidence': 0.8}
            elif face_ratio > 0.02:
                return {'type': 'wide', 'ratio': face_ratio, 'confidence': 0.7}
            else:
                return {'type': 'extreme_wide', 'ratio': face_ratio, 'confidence': 0.6}
                
        elif len(bodies) > 0:
            return {'type': 'wide', 'ratio': 0, 'confidence': 0.5}
        else:
            # Use edge detection for landscape shots
            edges = cv2.Canny(frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < 0.05:
                return {'type': 'extreme_wide', 'ratio': 0, 'confidence': 0.4}
            else:
                return {'type': 'wide', 'ratio': 0, 'confidence': 0.4}
                
    def _detect_camera_angle(self, frame):
        """Detect camera angle."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        height = frame.shape[0]
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_center_y = largest_face[1] + largest_face[3] / 2
            
            if face_center_y < height * 0.3:
                return 'high_angle'
            elif face_center_y > height * 0.7:
                return 'low_angle'
            else:
                return 'eye_level'
                
        # Check for Dutch angle
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
                
            avg_angle = np.mean(angles)
            if 5 < abs(avg_angle) < 85:
                return 'dutch_angle'
                
        return 'eye_level'
        
    def _analyze_motion(self, frame):
        """Analyze camera and subject motion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame
            return {
                'camera_movement': 'static',
                'motion_intensity': 'none',
                'subject_motion': False
            }
            
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Analyze flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        
        # Determine camera movement
        horizontal_flow = np.mean(flow[:, :, 0])
        vertical_flow = np.mean(flow[:, :, 1])
        
        if avg_magnitude < 1:
            camera_movement = 'static'
            motion_intensity = 'none'
        elif abs(horizontal_flow) > abs(vertical_flow) * 2:
            camera_movement = 'pan_right' if horizontal_flow > 0 else 'pan_left'
            motion_intensity = 'moderate' if avg_magnitude < 5 else 'high'
        elif abs(vertical_flow) > abs(horizontal_flow) * 2:
            camera_movement = 'tilt_down' if vertical_flow > 0 else 'tilt_up'
            motion_intensity = 'moderate' if avg_magnitude < 5 else 'high'
        else:
            camera_movement = 'complex'
            motion_intensity = 'high'
            
        # Detect subject motion
        motion_mask = magnitude > 3
        subject_motion = np.sum(motion_mask) > 0.1 * motion_mask.size
        
        self.prev_gray = gray
        self.prev_frame = frame
        
        return {
            'camera_movement': camera_movement,
            'motion_intensity': motion_intensity,
            'subject_motion': subject_motion,
            'average_flow': float(avg_magnitude)
        }
        
    def _analyze_color_grading(self, frame):
        """Analyze color grading and temperature."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Analyze color distribution
        b, g, r = cv2.split(frame)
        
        # Temperature analysis
        avg_b = np.mean(lab[:, :, 2]) - 128
        if avg_b > 10:
            temperature = 'warm'
        elif avg_b < -10:
            temperature = 'cool'
        else:
            temperature = 'neutral'
            
        # Detect specific color grading styles
        # Teal & Orange detection
        orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
        teal_mask = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([100, 255, 255]))
        
        orange_percent = np.sum(orange_mask > 0) / orange_mask.size
        teal_percent = np.sum(teal_mask > 0) / teal_mask.size
        
        if orange_percent > 0.05 and teal_percent > 0.05:
            style = 'teal_orange'
        elif np.mean(hsv[:, :, 1]) < 50:  # Low saturation
            style = 'desaturated'
        elif np.mean(g) > np.mean(b) * 1.2 and np.mean(g) > np.mean(r) * 1.2:
            style = 'matrix_green'
        else:
            style = 'standard'
            
        # Calculate saturation and contrast
        saturation = np.mean(hsv[:, :, 1]) / 255
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / 128
        
        return {
            'style': style,
            'temperature': temperature,
            'saturation': float(saturation),
            'contrast': float(contrast),
            'dominant_hue': float(np.mean(hsv[:, :, 0]))
        }
        
    def _analyze_composition(self, frame):
        """Analyze frame composition."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Rule of thirds
        thirds_v = [width // 3, 2 * width // 3]
        thirds_h = [height // 3, 2 * height // 3]
        
        # Check for subjects near thirds
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        follows_thirds = False
        if len(faces) > 0:
            for face in faces:
                face_center_x = face[0] + face[2] / 2
                face_center_y = face[1] + face[3] / 2
                
                for v in thirds_v:
                    if abs(face_center_x - v) < width * 0.1:
                        follows_thirds = True
                        break
                        
        # Symmetry check
        left_half = frame[:, :width//2]
        right_half = cv2.flip(frame[:, width//2:], 1)
        symmetry_score = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        
        # Leading lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        has_leading_lines = lines is not None and len(lines) > 5
        
        return {
            'rule_of_thirds': follows_thirds,
            'symmetry': symmetry_score > 0.7,
            'symmetry_score': float(symmetry_score),
            'leading_lines': has_leading_lines
        }
        
    def _estimate_depth_of_field(self, frame):
        """Estimate depth of field characteristics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate focus using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_measure = laplacian.var()
        
        # Divide into regions
        h, w = gray.shape
        center = gray[h//3:2*h//3, w//3:2*w//3]
        edges = np.concatenate([
            gray[:h//3, :].flatten(),
            gray[2*h//3:, :].flatten()
        ])
        
        center_sharp = cv2.Laplacian(center, cv2.CV_64F).var()
        edge_sharp = cv2.Laplacian(edges.reshape(-1, 1), cv2.CV_64F).var()
        
        if center_sharp > edge_sharp * 2:
            dof_type = 'shallow'
        elif focus_measure > 100:
            dof_type = 'deep'
        else:
            dof_type = 'medium'
            
        return {
            'type': dof_type,
            'focus_score': float(focus_measure),
            'center_sharpness': float(center_sharp)
        }
        
    def _analyze_emotions(self, frame):
        """Basic emotion analysis based on visual characteristics."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use brightness and color as proxy
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        
        if brightness > 180 and saturation > 100:
            emotion = 'energetic'
            intensity = 0.8
        elif brightness < 80:
            emotion = 'somber'
            intensity = 0.7
        elif saturation < 50:
            emotion = 'calm'
            intensity = 0.5
        else:
            emotion = 'neutral'
            intensity = 0.4
            
        return {
            'dominant_emotion': emotion,
            'intensity': intensity
        }
        
    def _calculate_scores(self, analysis):
        """Calculate cinematography scores."""
        # Emotional impact
        shot_weights = {
            'extreme_close': 1.0, 'close': 0.8, 'medium': 0.5,
            'wide': 0.3, 'extreme_wide': 0.1
        }
        emotional_impact = analysis['emotions']['intensity'] * shot_weights.get(
            analysis['shot_type']['type'], 0.5
        )
        
        # Visual dynamics
        motion_weights = {
            'none': 0, 'minimal': 0.3, 'moderate': 0.6, 'high': 1.0
        }
        visual_dynamics = motion_weights.get(analysis['motion']['motion_intensity'], 0)
        
        # Aesthetic quality
        aesthetic = 0.5
        if analysis['composition']['rule_of_thirds']:
            aesthetic += 0.2
        if analysis['composition']['symmetry']:
            aesthetic += 0.15
        if analysis['composition']['leading_lines']:
            aesthetic += 0.15
            
        # Color bonus
        if analysis['color_grading']['style'] != 'standard':
            aesthetic += 0.1
            
        return {
            'emotional_impact': float(emotional_impact),
            'visual_dynamics': float(visual_dynamics),
            'aesthetic_quality': float(min(aesthetic, 1.0)),
            'overall': float(np.mean([emotional_impact, visual_dynamics, aesthetic]))
        }
        
    def _generate_directors_notes(self, analysis):
        """Generate director-style notes."""
        notes = []
        
        # Shot description
        shot_type = analysis['shot_type']['type']
        if shot_type == 'extreme_close':
            notes.append("Extreme close-up creates intense intimacy")
        elif shot_type == 'close':
            notes.append("Close-up focuses on emotional detail")
        elif shot_type == 'medium':
            notes.append("Medium shot balances subject and environment")
        elif shot_type == 'wide':
            notes.append("Wide shot establishes spatial context")
        else:
            notes.append("Extreme wide shot emphasizes scale and isolation")
            
        # Camera movement
        movement = analysis['motion']['camera_movement']
        if movement != 'static':
            notes.append(f"Camera {movement} adds kinetic energy")
            
        # Color grading
        style = analysis['color_grading']['style']
        if style == 'teal_orange':
            notes.append("Teal & orange grading creates blockbuster aesthetic")
        elif style == 'desaturated':
            notes.append("Desaturated palette suggests realism or melancholy")
        elif style == 'matrix_green':
            notes.append("Green tint evokes digital/artificial atmosphere")
            
        # Composition
        if analysis['composition']['rule_of_thirds']:
            notes.append("Rule of thirds creates balanced composition")
        if analysis['composition']['symmetry']:
            notes.append("Symmetrical framing suggests order and control")
            
        return ". ".join(notes)
        
    def create_visualization(self, results, output_path):
        """Create comprehensive visualization."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Cinematographic Analysis - Instagram Reel', fontsize=16, fontweight='bold')
        
        timestamps = [r['timestamp'] for r in results]
        
        # 1. Shot Types Timeline
        ax = axes[0, 0]
        shot_values = {'extreme_close': 5, 'close': 4, 'medium': 3, 'wide': 2, 'extreme_wide': 1}
        shots = [shot_values.get(r['shot_type']['type'], 3) for r in results]
        colors = ['#FF0000', '#FF6600', '#FFCC00', '#66CC00', '#0066CC']
        shot_colors = [colors[s-1] for s in shots]
        
        ax.scatter(timestamps, shots, c=shot_colors, s=50, alpha=0.8)
        ax.plot(timestamps, shots, color='gray', alpha=0.3, linewidth=1)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['Extreme\nWide', 'Wide', 'Medium', 'Close', 'Extreme\nClose'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Shot Progression')
        ax.grid(True, alpha=0.3)
        
        # 2. Motion Intensity
        ax = axes[0, 1]
        motion_map = {'none': 0, 'minimal': 0.33, 'moderate': 0.66, 'high': 1.0}
        motion_values = [motion_map.get(r['motion']['motion_intensity'], 0) for r in results]
        
        ax.fill_between(timestamps, motion_values, alpha=0.5, color='#FF6B6B')
        ax.plot(timestamps, motion_values, color='#FF6B6B', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Motion Intensity')
        ax.set_title('Camera/Subject Motion')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 3. Color Temperature
        ax = axes[1, 0]
        temp_colors = {'warm': '#FF6B35', 'neutral': '#808080', 'cool': '#35A7FF'}
        temp_timeline = [temp_colors[r['color_grading']['temperature']] for r in results]
        
        for i, (t, c) in enumerate(zip(timestamps, temp_timeline)):
            ax.axvspan(t - 0.5 if i > 0 else 0, t + 0.5, color=c, alpha=0.3)
            
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Color Temperature Timeline')
        ax.set_ylim(0, 1)
        
        # 4. Cinematography Scores
        ax = axes[1, 1]
        overall_scores = [r['cinematography_score']['overall'] for r in results]
        emotional_scores = [r['cinematography_score']['emotional_impact'] for r in results]
        aesthetic_scores = [r['cinematography_score']['aesthetic_quality'] for r in results]
        
        ax.plot(timestamps, overall_scores, label='Overall', linewidth=3, color='#2C3E50')
        ax.plot(timestamps, emotional_scores, label='Emotional Impact', linewidth=2, alpha=0.7)
        ax.plot(timestamps, aesthetic_scores, label='Aesthetic Quality', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Score')
        ax.set_title('Cinematography Quality Scores')
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 5. Shot Distribution
        ax = axes[2, 0]
        shot_counts = defaultdict(int)
        for r in results:
            shot_counts[r['shot_type']['type']] += 1
            
        shots = list(shot_counts.keys())
        counts = list(shot_counts.values())
        
        ax.bar(shots, counts, color=sns.color_palette("viridis", len(shots)))
        ax.set_xlabel('Shot Type')
        ax.set_ylabel('Count')
        ax.set_title('Shot Type Distribution')
        
        # 6. Key Statistics
        ax = axes[2, 1]
        ax.axis('off')
        
        stats_text = f"""
Video Statistics:
- Duration: {self.duration:.1f} seconds
- Resolution: {self.width}x{self.height}
- FPS: {self.fps}
- Frames Analyzed: {len(results)}

Cinematographic Summary:
- Dominant Shot: {max(shot_counts.items(), key=lambda x: x[1])[0]}
- Average Score: {np.mean(overall_scores):.2f}
- Peak Score: {max(overall_scores):.2f}
- Most Dynamic Moment: {timestamps[np.argmax(motion_values)]:.1f}s

Color Analysis:
- Primary Temperature: {max(set([r['color_grading']['temperature'] for r in results]), 
                           key=[r['color_grading']['temperature'] for r in results].count)}
- Average Saturation: {np.mean([r['color_grading']['saturation'] for r in results]):.2f}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {output_path}")


def main():
    """Run complete analysis."""
    video_path = "test_downloads/Video by christianpolk.mp4"
    
    print("="*60)
    print("COMPLETE CINEMATOGRAPHIC ANALYSIS")
    print("="*60)
    
    # Create analyzer
    analyzer = CompleteCinematographicAnalyzer(video_path)
    
    print(f"\nVideo Information:")
    print(f"- File: {Path(video_path).name}")
    print(f"- Duration: {analyzer.duration:.2f} seconds")
    print(f"- Resolution: {analyzer.width}x{analyzer.height}")
    print(f"- FPS: {analyzer.fps}")
    
    # Run analysis
    results = analyzer.analyze_complete(interval=2.0)  # Analyze every 2 seconds
    
    # Save results
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(output_dir / "complete_analysis.json", 'w') as f:
        # Convert numpy bools to Python bools
        json_results = []
        for r in results:
            json_result = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    json_result[k] = {
                        kk: bool(vv) if isinstance(vv, np.bool_) else vv 
                        for kk, vv in v.items()
                    }
                else:
                    json_result[k] = bool(v) if isinstance(v, np.bool_) else v
            json_results.append(json_result)
            
        json.dump({
            'video_path': str(video_path),
            'metadata': {
                'duration': float(analyzer.duration),
                'resolution': f"{analyzer.width}x{analyzer.height}",
                'fps': float(analyzer.fps),
                'total_frames': int(analyzer.total_frames)
            },
            'analysis': json_results
        }, f, indent=2)
        
    # Create visualization
    analyzer.create_visualization(results, output_dir / "cinematography_visualization.png")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Find key moments
    high_score_moments = sorted(results, key=lambda x: x['cinematography_score']['overall'], reverse=True)[:5]
    
    print("\nTop 5 Cinematographic Moments:")
    for i, moment in enumerate(high_score_moments, 1):
        print(f"\n{i}. Time: {moment['timestamp']:.1f}s (Score: {moment['cinematography_score']['overall']:.2f})")
        print(f"   Shot: {moment['shot_type']['type']}, Movement: {moment['motion']['camera_movement']}")
        print(f"   Color: {moment['color_grading']['style']} ({moment['color_grading']['temperature']})")
        print(f"   Notes: {moment['directors_notes']}")
        
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()