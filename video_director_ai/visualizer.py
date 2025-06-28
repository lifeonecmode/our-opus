import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from .settings import Config


class CinematographyVisualizer:
    """Creates comprehensive visualizations of video analysis."""
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Color mappings
        self.emotion_colors = {
            'angry': '#FF4444',
            'disgust': '#8B4513',
            'fear': '#9370DB',
            'happy': '#FFD700',
            'sad': '#4169E1',
            'surprise': '#FF69B4',
            'neutral': '#808080'
        }
        
        self.shot_colors = {
            'extreme_close': '#FF0000',
            'close': '#FF6600',
            'medium': '#FFCC00',
            'wide': '#66CC00',
            'extreme_wide': '#0066CC'
        }
        
        self.motion_colors = {
            'static': '#808080',
            'pan_left': '#FF6B6B',
            'pan_right': '#4ECDC4',
            'tilt_up': '#45B7D1',
            'tilt_down': '#96CEB4',
            'zoom_in': '#FFA07A',
            'zoom_out': '#98D8C8'
        }
        
    def create_comprehensive_visualization(self, analysis_results: Dict[str, Any], 
                                         output_path: Path) -> None:
        """Create a comprehensive multi-panel visualization."""
        fig = plt.figure(figsize=(24, 16))
        
        # Create grid
        gs = fig.add_gridspec(4, 3, height_ratios=[1.5, 1, 1, 1], 
                            width_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Extract data
        cinematography = analysis_results.get('cinematography', [])
        synthesis = analysis_results.get('synthesis', {})
        metadata = analysis_results.get('metadata', {})
        
        if not cinematography:
            plt.text(0.5, 0.5, 'No data to visualize', ha='center', va='center')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return
            
        # Panel 1: Emotional Journey Timeline
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_emotional_journey(ax1, cinematography)
        
        # Panel 2: Shot Type Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_shot_distribution(ax2, cinematography)
        
        # Panel 3: Camera Movement Analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_camera_movements(ax3, cinematography)
        
        # Panel 4: Color Grading Timeline
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_color_grading(ax4, cinematography)
        
        # Panel 5: Cinematography Score Timeline
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_cinematography_scores(ax5, cinematography)
        
        # Panel 6: Scene Pacing
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_scene_pacing(ax6, analysis_results.get('scenes', []))
        
        # Panel 7: Emotional Distribution
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_emotion_distribution(ax7, synthesis.get('emotional_arc', {}))
        
        # Panel 8: Key Insights
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_key_insights(ax8, synthesis, metadata)
        
        # Main title
        fig.suptitle(f'Video Director Analysis: {metadata.get("filename", "Unknown")}', 
                    fontsize=20, fontweight='bold')
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _plot_emotional_journey(self, ax: plt.Axes, cinematography: List[Dict]) -> None:
        """Plot emotional intensity over time with dominant emotions."""
        timestamps = []
        intensities = []
        emotions = []
        colors = []
        
        for frame in cinematography:
            timestamps.append(frame['timestamp'])
            intensities.append(frame['emotions'].get('emotional_intensity', 0))
            emotion = frame['emotions'].get('dominant_emotion', 'neutral')
            emotions.append(emotion)
            colors.append(self.emotion_colors.get(emotion, '#808080'))
            
        # Create scatter plot with emotion colors
        scatter = ax.scatter(timestamps, intensities, c=colors, s=50, alpha=0.6)
        
        # Add trend line
        if len(timestamps) > 3:
            z = np.polyfit(timestamps, intensities, 3)
            p = np.poly1d(z)
            smooth_times = np.linspace(min(timestamps), max(timestamps), 100)
            ax.plot(smooth_times, p(smooth_times), 'k--', alpha=0.5, linewidth=2)
            
        # Mark emotional peaks
        for frame in cinematography:
            if frame.get('is_emotional_peak'):
                ax.annotate(frame['emotions']['dominant_emotion'],
                          (frame['timestamp'], frame['emotions']['emotional_intensity']),
                          xytext=(0, 10), textcoords='offset points',
                          ha='center', fontsize=8, style='italic')
                
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Emotional Intensity', fontsize=12)
        ax.set_title('Emotional Journey Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # Add emotion legend
        emotion_patches = [mpatches.Patch(color=color, label=emotion.capitalize()) 
                         for emotion, color in self.emotion_colors.items() 
                         if emotion in emotions]
        ax.legend(handles=emotion_patches, loc='upper right', ncol=3, fontsize=8)
        
    def _plot_shot_distribution(self, ax: plt.Axes, cinematography: List[Dict]) -> None:
        """Plot distribution of shot types."""
        shot_counts = {}
        
        for frame in cinematography:
            shot_type = frame['shot_type']['shot_type']
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
            
        # Sort by logical order (close to wide)
        shot_order = ['extreme_close', 'close', 'medium', 'wide', 'extreme_wide']
        shots = [s for s in shot_order if s in shot_counts]
        counts = [shot_counts[s] for s in shots]
        colors = [self.shot_colors[s] for s in shots]
        
        # Create bar chart
        bars = ax.bar(range(len(shots)), counts, color=colors, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
        ax.set_xticks(range(len(shots)))
        ax.set_xticklabels([s.replace('_', '\n') for s in shots], fontsize=10)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Shot Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
    def _plot_camera_movements(self, ax: plt.Axes, cinematography: List[Dict]) -> None:
        """Plot camera movement analysis."""
        movement_counts = {}
        
        for frame in cinematography:
            movement = frame['motion']['camera_movement']['type']
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
            
        # Sort by count
        movements = sorted(movement_counts.items(), key=lambda x: x[1], reverse=True)[:7]
        
        if movements:
            labels, counts = zip(*movements)
            colors = [self.motion_colors.get(m, '#808080') for m in labels]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
            
            # Improve text
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
                autotext.set_weight('bold')
                
        ax.set_title('Camera Movement Distribution', fontsize=14, fontweight='bold')
        
    def _plot_color_grading(self, ax: plt.Axes, cinematography: List[Dict]) -> None:
        """Plot color grading styles used."""
        grading_counts = {}
        
        for frame in cinematography:
            style = frame['color_grading']['color_grading']['style']
            grading_counts[style] = grading_counts.get(style, 0) + 1
            
        # Get top styles
        top_styles = sorted(grading_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_styles:
            styles, counts = zip(*top_styles)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(styles))
            ax.barh(y_pos, counts, color=sns.color_palette("viridis", len(styles)))
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(styles, fontsize=10)
            ax.set_xlabel('Frames', fontsize=12)
            ax.set_title('Color Grading Styles', fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add count labels
            for i, (style, count) in enumerate(top_styles):
                ax.text(count + 0.5, i, str(count), va='center', fontsize=10)
                
    def _plot_cinematography_scores(self, ax: plt.Axes, cinematography: List[Dict]) -> None:
        """Plot cinematography scores over time."""
        timestamps = []
        emotional_impact = []
        visual_dynamics = []
        aesthetic_quality = []
        overall_scores = []
        
        for frame in cinematography:
            timestamps.append(frame['timestamp'])
            scores = frame['cinematography_score']
            emotional_impact.append(scores['emotional_impact'])
            visual_dynamics.append(scores['visual_dynamics'])
            aesthetic_quality.append(scores['aesthetic_quality'])
            overall_scores.append(scores['overall'])
            
        # Plot lines
        ax.plot(timestamps, emotional_impact, label='Emotional Impact', 
                color='#FF6B6B', linewidth=2, alpha=0.8)
        ax.plot(timestamps, visual_dynamics, label='Visual Dynamics', 
                color='#4ECDC4', linewidth=2, alpha=0.8)
        ax.plot(timestamps, aesthetic_quality, label='Aesthetic Quality', 
                color='#45B7D1', linewidth=2, alpha=0.8)
        ax.plot(timestamps, overall_scores, label='Overall Score', 
                color='#2C3E50', linewidth=3, linestyle='--')
        
        # Mark high points
        high_points = [(t, s) for t, s in zip(timestamps, overall_scores) if s > 0.8]
        if high_points:
            t_high, s_high = zip(*high_points)
            ax.scatter(t_high, s_high, color='gold', s=100, zorder=5, 
                      edgecolors='black', linewidth=2)
            
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Cinematography Quality Scores', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
    def _plot_scene_pacing(self, ax: plt.Axes, scenes: List[Dict]) -> None:
        """Plot scene durations to show pacing."""
        if not scenes:
            ax.text(0.5, 0.5, 'No scene data', ha='center', va='center')
            ax.set_title('Scene Pacing', fontsize=14, fontweight='bold')
            return
            
        scene_numbers = []
        durations = []
        
        for scene in scenes:
            scene_numbers.append(scene['scene_number'])
            durations.append(scene['duration'])
            
        # Create bar chart
        bars = ax.bar(scene_numbers, durations, color=sns.color_palette("coolwarm", len(scenes)))
        
        # Add average line
        avg_duration = np.mean(durations)
        ax.axhline(y=avg_duration, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_duration:.1f}s')
        
        ax.set_xlabel('Scene Number', fontsize=12)
        ax.set_ylabel('Duration (seconds)', fontsize=12)
        ax.set_title('Scene Pacing Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
    def _plot_emotion_distribution(self, ax: plt.Axes, emotional_arc: Dict) -> None:
        """Plot distribution of emotions throughout video."""
        distribution = emotional_arc.get('emotion_distribution', {})
        
        if distribution:
            emotions = list(distribution.keys())
            percentages = list(distribution.values())
            colors = [self.emotion_colors.get(e, '#808080') for e in emotions]
            
            # Create donut chart
            wedges, texts, autotexts = ax.pie(percentages, labels=emotions, colors=colors,
                                             autopct='%1.1f%%', startangle=90,
                                             pctdistance=0.85)
            
            # Create donut hole
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            
            # Add center text
            arc_type = emotional_arc.get('arc_type', 'unknown')
            ax.text(0, 0, f'{arc_type.replace("_", " ").title()}\nArc', 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
        ax.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
        
    def _plot_key_insights(self, ax: plt.Axes, synthesis: Dict, metadata: Dict) -> None:
        """Display key insights and statistics."""
        ax.axis('off')
        
        insights = []
        
        # Add metadata
        insights.append(f"Duration: {metadata.get('duration', 0):.1f}s")
        insights.append(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
        insights.append("")
        
        # Add style insights
        style = synthesis.get('cinematographic_style', {})
        insights.append(f"Shot Style: {style.get('shot_style', 'Unknown')}")
        insights.append(f"Movement: {style.get('movement_style', 'Unknown')}")
        insights.append(f"Color Style: {style.get('color_palette', 'Unknown')}")
        insights.append("")
        
        # Add pacing insights
        pacing = synthesis.get('pacing', {})
        insights.append(f"Pacing: {pacing.get('pacing_type', 'Unknown')}")
        insights.append(f"Avg Shot: {pacing.get('average_shot_duration', 0):.1f}s")
        insights.append("")
        
        # Add emotional insights
        emotional = synthesis.get('emotional_arc', {})
        insights.append(f"Emotional Arc: {emotional.get('arc_type', 'Unknown')}")
        insights.append(f"Intensity: {emotional.get('average_intensity', 0):.2f}")
        
        # Display insights
        text = '\n'.join(insights)
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title('Key Insights', fontsize=14, fontweight='bold', y=0.98)
        
    def create_timeline_plot(self, analysis_results: Dict[str, Any], 
                           output_path: Path) -> None:
        """Create a detailed timeline visualization."""
        cinematography = analysis_results.get('cinematography', [])
        scenes = analysis_results.get('scenes', [])
        
        if not cinematography:
            return
            
        fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        
        # Timeline data
        timestamps = [f['timestamp'] for f in cinematography]
        
        # Plot 1: Emotions
        self._plot_emotion_timeline(axes[0], cinematography, timestamps)
        
        # Plot 2: Shot types
        self._plot_shot_timeline(axes[1], cinematography, timestamps)
        
        # Plot 3: Motion
        self._plot_motion_timeline(axes[2], cinematography, timestamps)
        
        # Plot 4: Overall score
        self._plot_score_timeline(axes[3], cinematography, timestamps)
        
        # Add scene boundaries
        for scene in scenes:
            for ax in axes:
                ax.axvline(x=scene['start_time'], color='red', alpha=0.3, linestyle='--')
                
        plt.xlabel('Time (seconds)', fontsize=14)
        fig.suptitle('Detailed Timeline Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_emotion_timeline(self, ax: plt.Axes, cinematography: List[Dict], 
                              timestamps: List[float]) -> None:
        """Plot emotion timeline with stacked area chart."""
        # Prepare emotion data
        emotion_data = {emotion: [] for emotion in Config.EMOTIONS}
        
        for frame in cinematography:
            emotions = frame['emotions'].get('emotions', {})
            for emotion in Config.EMOTIONS:
                emotion_data[emotion].append(emotions.get(emotion, 0))
                
        # Create stacked area chart
        ax.stackplot(timestamps, *emotion_data.values(), 
                    labels=list(emotion_data.keys()),
                    colors=[self.emotion_colors.get(e, '#808080') for e in emotion_data.keys()],
                    alpha=0.7)
        
        ax.set_ylabel('Emotion Level', fontsize=12)
        ax.set_title('Emotional Composition', fontsize=14)
        ax.legend(loc='upper right', ncol=len(Config.EMOTIONS), fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_shot_timeline(self, ax: plt.Axes, cinematography: List[Dict], 
                          timestamps: List[float]) -> None:
        """Plot shot types as colored bars."""
        shot_types = []
        shot_values = []
        
        for frame in cinematography:
            shot_type = frame['shot_type']['shot_type']
            shot_types.append(shot_type)
            
            # Convert to numeric for plotting
            shot_map = {'extreme_close': 5, 'close': 4, 'medium': 3, 
                       'wide': 2, 'extreme_wide': 1}
            shot_values.append(shot_map.get(shot_type, 3))
            
        # Create color map
        colors = [self.shot_colors.get(s, '#808080') for s in shot_types]
        
        # Plot as scatter with connecting lines
        ax.scatter(timestamps, shot_values, c=colors, s=50, alpha=0.8)
        ax.plot(timestamps, shot_values, color='gray', alpha=0.3, linewidth=1)
        
        ax.set_ylabel('Shot Type', fontsize=12)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['Extreme\nWide', 'Wide', 'Medium', 'Close', 'Extreme\nClose'])
        ax.set_title('Shot Progression', fontsize=14)
        ax.grid(True, alpha=0.3)
        
    def _plot_motion_timeline(self, ax: plt.Axes, cinematography: List[Dict], 
                            timestamps: List[float]) -> None:
        """Plot motion intensity and camera movement."""
        motion_intensity = []
        camera_movements = []
        
        for frame in cinematography:
            # Get motion intensity as numeric
            intensity_map = {'none': 0, 'minimal': 0.25, 'moderate': 0.5, 
                           'high': 0.75, 'extreme': 1.0}
            intensity = frame['motion']['motion_intensity']['intensity']
            motion_intensity.append(intensity_map.get(intensity, 0))
            
            # Track camera movement
            movement = frame['motion']['camera_movement']['type']
            camera_movements.append(0 if movement == 'static' else 1)
            
        # Plot motion intensity as area
        ax.fill_between(timestamps, motion_intensity, alpha=0.5, color='#FF6B6B', 
                       label='Motion Intensity')
        
        # Plot camera movement as scatter
        moving_times = [t for t, m in zip(timestamps, camera_movements) if m == 1]
        if moving_times:
            ax.scatter(moving_times, [1.1] * len(moving_times), marker='v', 
                      color='#4ECDC4', s=30, label='Camera Movement')
            
        ax.set_ylabel('Motion Level', fontsize=12)
        ax.set_title('Motion Analysis', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.2)
        
    def _plot_score_timeline(self, ax: plt.Axes, cinematography: List[Dict], 
                           timestamps: List[float]) -> None:
        """Plot overall cinematography score."""
        scores = [f['cinematography_score']['overall'] for f in cinematography]
        
        # Plot main score line
        ax.plot(timestamps, scores, color='#2C3E50', linewidth=2.5, label='Overall Score')
        
        # Fill area under curve
        ax.fill_between(timestamps, scores, alpha=0.3, color='#3498DB')
        
        # Mark high points
        high_indices = [i for i, s in enumerate(scores) if s > 0.8]
        if high_indices:
            high_times = [timestamps[i] for i in high_indices]
            high_scores = [scores[i] for i in high_indices]
            ax.scatter(high_times, high_scores, color='gold', s=100, 
                      edgecolors='black', linewidth=2, zorder=5, 
                      label='High Quality Moments')
            
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Quality Score', fontsize=12)
        ax.set_title('Cinematography Quality', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)