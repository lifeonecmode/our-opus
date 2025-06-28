#!/usr/bin/env python3
"""
Direct Instagram Video Analysis - Avoiding import issues
"""

import yt_dlp
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import time

class InstagramAnalyzer:
    def __init__(self):
        self.output_dir = Path("instagram_comparison")
        self.output_dir.mkdir(exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def download_video(self, url, label):
        """Download Instagram video."""
        ydl_opts = {
            'outtmpl': str(self.output_dir / f'{label}_%(id)s.%(ext)s'),
            'quiet': False,
            'format': 'best',
            'cookiesfrombrowser': ('chrome',),
        }
        
        print(f"\nDownloading {label}: {url}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                if not Path(filename).exists():
                    filename = filename.rsplit('.', 1)[0] + '.mp4'
                
                return {
                    'path': filename,
                    'title': info.get('title', label),
                    'view_count': info.get('view_count', 0),
                    'duration': info.get('duration', 0),
                    'url': url
                }
        except Exception as e:
            print(f"Error downloading: {e}")
            return None
    
    def analyze_video(self, video_path, label):
        """Comprehensive video analysis."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nAnalyzing {label}...")
        print(f"Duration: {duration:.2f}s, FPS: {fps}, Frames: {total_frames}")
        
        analysis = {
            'label': label,
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'cuts': [],
            'motion_scores': [],
            'brightness_timeline': [],
            'face_timeline': [],
            'hook_analysis': {
                'cuts_in_first_3s': 0,
                'motion_peaks_first_3s': 0,
                'face_visible_first_3s': 0
            },
            'micro_edits': []
        }
        
        prev_gray = None
        prev_frame = None
        frame_count = 0
        shot_start = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Cut detection
                diff = cv2.absdiff(prev_gray, gray)
                diff_score = np.mean(diff)
                
                # Hard cut detection
                if diff_score > 40:
                    shot_duration = timestamp - shot_start
                    analysis['cuts'].append({
                        'timestamp': timestamp,
                        'shot_duration': shot_duration,
                        'intensity': diff_score
                    })
                    shot_start = timestamp
                    
                    if timestamp <= 3:
                        analysis['hook_analysis']['cuts_in_first_3s'] += 1
                
                # Micro-edit detection
                elif 15 < diff_score < 40:
                    analysis['micro_edits'].append({
                        'timestamp': timestamp,
                        'intensity': diff_score
                    })
                
                # Motion analysis every 5 frames
                if frame_count % 5 == 0 and prev_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_score = np.mean(magnitude)
                    
                    analysis['motion_scores'].append({
                        'timestamp': timestamp,
                        'score': motion_score
                    })
                    
                    if timestamp <= 3 and motion_score > 3:
                        analysis['hook_analysis']['motion_peaks_first_3s'] += 1
            
            # Analyze every 10th frame
            if frame_count % 10 == 0:
                # Brightness
                brightness = np.mean(gray)
                analysis['brightness_timeline'].append({
                    'timestamp': timestamp,
                    'brightness': brightness
                })
                
                # Face detection
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                has_face = len(faces) > 0
                analysis['face_timeline'].append({
                    'timestamp': timestamp,
                    'has_face': has_face,
                    'num_faces': len(faces)
                })
                
                if timestamp <= 3 and has_face:
                    analysis['hook_analysis']['face_visible_first_3s'] += 1
            
            prev_gray = gray
            prev_frame = frame
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Progress: {(frame_count/total_frames)*100:.1f}%", end='\r')
        
        cap.release()
        
        # Calculate statistics
        analysis['shot_changes'] = len(analysis['cuts'])
        analysis['avg_shot_duration'] = duration / (len(analysis['cuts']) + 1) if analysis['cuts'] else duration
        analysis['face_visibility_pct'] = sum(1 for f in analysis['face_timeline'] if f['has_face']) / len(analysis['face_timeline']) * 100 if analysis['face_timeline'] else 0
        analysis['avg_motion'] = np.mean([m['score'] for m in analysis['motion_scores']]) if analysis['motion_scores'] else 0
        analysis['brightness_variance'] = np.var([b['brightness'] for b in analysis['brightness_timeline']]) if analysis['brightness_timeline'] else 0
        
        print(f"\n  ✓ Analysis complete: {analysis['shot_changes']} cuts, {analysis['avg_shot_duration']:.2f}s avg shot")
        
        return analysis
    
    def create_comparison_visualization(self, analysis1, analysis2):
        """Create comprehensive comparison visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Instagram Video Performance Analysis: 100k+ vs 15k Views', fontsize=16, fontweight='bold')
        
        # 1. Hook Analysis (First 3 seconds)
        ax = axes[0, 0]
        ax.set_xlim(0, 3)
        ax.set_title('First 3 Seconds - The Hook', fontweight='bold')
        
        for cut in analysis1['cuts']:
            if cut['timestamp'] <= 3:
                ax.axvline(x=cut['timestamp'], color='red', alpha=0.7, linewidth=2)
        for cut in analysis2['cuts']:
            if cut['timestamp'] <= 3:
                ax.axvline(x=cut['timestamp'], color='blue', alpha=0.7, linewidth=2, linestyle='--')
        
        ax.text(1.5, 0.5, f"100k+: {analysis1['hook_analysis']['cuts_in_first_3s']} cuts\n15k: {analysis2['hook_analysis']['cuts_in_first_3s']} cuts",
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax.set_xlabel('Time (seconds)')
        
        # 2. Full Timeline Cuts
        ax = axes[0, 1]
        ax.set_title('Cut Frequency Over Time')
        
        max_time = max(analysis1['duration'], analysis2['duration'])
        for cut in analysis1['cuts']:
            ax.axvline(x=cut['timestamp'], color='red', alpha=0.5, linewidth=1)
        for cut in analysis2['cuts']:
            ax.axvline(x=cut['timestamp'], color='blue', alpha=0.5, linewidth=1, ymin=0.5, ymax=1)
        
        ax.set_xlim(0, max_time)
        ax.set_xlabel('Time (seconds)')
        ax.text(0.02, 0.95, f"100k+: {analysis1['shot_changes']} cuts", transform=ax.transAxes, color='red')
        ax.text(0.02, 0.88, f"15k: {analysis2['shot_changes']} cuts", transform=ax.transAxes, color='blue')
        
        # 3. Editing Pace
        ax = axes[0, 2]
        ax.bar(['100k+ views', '15k views'], 
               [analysis1['shot_changes'], analysis2['shot_changes']], 
               color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Number of Cuts')
        ax.set_title('Total Cuts Comparison')
        
        for i, (label, val, avg) in enumerate([
            ('100k+ views', analysis1['shot_changes'], analysis1['avg_shot_duration']),
            ('15k views', analysis2['shot_changes'], analysis2['avg_shot_duration'])
        ]):
            ax.text(i, val + 0.5, f"{avg:.2f}s/shot", ha='center')
        
        # 4. Motion Intensity
        ax = axes[1, 0]
        ax.set_title('Motion Dynamics')
        
        if analysis1['motion_scores']:
            times1 = [m['timestamp'] for m in analysis1['motion_scores']]
            scores1 = [m['score'] for m in analysis1['motion_scores']]
            ax.plot(times1, scores1, color='red', alpha=0.7, label='100k+ views')
        
        if analysis2['motion_scores']:
            times2 = [m['timestamp'] for m in analysis2['motion_scores']]
            scores2 = [m['score'] for m in analysis2['motion_scores']]
            ax.plot(times2, scores2, color='blue', alpha=0.7, label='15k views')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Motion Score')
        ax.legend()
        
        # 5. Face Presence
        ax = axes[1, 1]
        face_data = [analysis1['face_visibility_pct'], analysis2['face_visibility_pct']]
        ax.bar(['100k+ views', '15k views'], face_data, color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Face Visibility %')
        ax.set_title('Human Connection')
        ax.set_ylim(0, 100)
        
        # 6. Visual Energy (Brightness Variance)
        ax = axes[1, 2]
        bright_data = [analysis1['brightness_variance'], analysis2['brightness_variance']]
        ax.bar(['100k+ views', '15k views'], bright_data, color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Brightness Variance')
        ax.set_title('Visual Energy')
        
        # 7. Micro-edits
        ax = axes[2, 0]
        micro_data = [len(analysis1['micro_edits']), len(analysis2['micro_edits'])]
        ax.bar(['100k+ views', '15k views'], micro_data, color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title('Micro-Edits & Transitions')
        
        # 8. Shot Duration Distribution
        ax = axes[2, 1]
        if analysis1['cuts'] and analysis2['cuts']:
            durations1 = [cut['shot_duration'] for cut in analysis1['cuts'] if cut['shot_duration'] < 10]
            durations2 = [cut['shot_duration'] for cut in analysis2['cuts'] if cut['shot_duration'] < 10]
            
            if durations1:
                ax.hist(durations1, bins=15, alpha=0.5, color='red', label='100k+ views')
            if durations2:
                ax.hist(durations2, bins=15, alpha=0.5, color='blue', label='15k views')
            
            ax.set_xlabel('Shot Duration (seconds)')
            ax.set_ylabel('Count')
            ax.set_title('Shot Duration Distribution')
            ax.legend()
        
        # 9. Key Insights
        ax = axes[2, 2]
        ax.axis('off')
        
        insights = self.generate_insights(analysis1, analysis2)
        ax.text(0.05, 0.95, insights, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        output_path = self.output_dir / "comparison_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved to: {output_path}")
    
    def generate_insights(self, analysis1, analysis2):
        """Generate key insights from comparison."""
        insights = "🔍 KEY VIRAL FACTORS:\n\n"
        
        # Hook strength
        if analysis1['hook_analysis']['cuts_in_first_3s'] > analysis2['hook_analysis']['cuts_in_first_3s']:
            diff = analysis1['hook_analysis']['cuts_in_first_3s'] - analysis2['hook_analysis']['cuts_in_first_3s']
            insights += f"✓ STRONGER HOOK: {diff} more cuts in first 3s\n"
        else:
            insights += "✗ WEAKER HOOK in viral video\n"
        
        # Pacing
        if analysis1['avg_shot_duration'] < analysis2['avg_shot_duration']:
            pct = (analysis2['avg_shot_duration'] / analysis1['avg_shot_duration'] - 1) * 100
            insights += f"✓ FASTER PACING: {pct:.0f}% quicker cuts\n"
        else:
            insights += "✗ SLOWER PACING in viral video\n"
        
        # Face visibility
        if analysis1['face_visibility_pct'] > analysis2['face_visibility_pct']:
            diff = analysis1['face_visibility_pct'] - analysis2['face_visibility_pct']
            insights += f"✓ MORE FACES: {diff:.0f}% more visibility\n"
        
        # Motion energy
        if analysis1['avg_motion'] > analysis2['avg_motion']:
            ratio = analysis1['avg_motion'] / analysis2['avg_motion']
            insights += f"✓ HIGHER ENERGY: {ratio:.1f}x more motion\n"
        
        # Visual dynamics
        if analysis1['brightness_variance'] > analysis2['brightness_variance']:
            ratio = analysis1['brightness_variance'] / analysis2['brightness_variance']
            insights += f"✓ DYNAMIC VISUALS: {ratio:.1f}x variance\n"
        
        # Micro-edits
        if len(analysis1['micro_edits']) > len(analysis2['micro_edits']):
            diff = len(analysis1['micro_edits']) - len(analysis2['micro_edits'])
            insights += f"✓ SMOOTHER FLOW: {diff} more micro-edits\n"
        
        insights += "\n🎯 SUCCESS FACTORS:\n"
        insights += "Hook + Pacing + Energy = Views!"
        
        return insights
    
    def save_analysis_data(self, video1_info, video2_info, analysis1, analysis2):
        """Save analysis results to JSON."""
        data = {
            'video1_100k': {
                'info': video1_info,
                'analysis': analysis1
            },
            'video2_15k': {
                'info': video2_info,
                'analysis': analysis2
            },
            'comparison': {
                'hook_difference': analysis1['hook_analysis']['cuts_in_first_3s'] - analysis2['hook_analysis']['cuts_in_first_3s'],
                'pacing_ratio': analysis2['avg_shot_duration'] / analysis1['avg_shot_duration'] if analysis1['avg_shot_duration'] > 0 else 1,
                'face_visibility_diff': analysis1['face_visibility_pct'] - analysis2['face_visibility_pct'],
                'motion_ratio': analysis1['avg_motion'] / analysis2['avg_motion'] if analysis2['avg_motion'] > 0 else 1,
                'visual_energy_ratio': analysis1['brightness_variance'] / analysis2['brightness_variance'] if analysis2['brightness_variance'] > 0 else 1
            }
        }
        
        output_path = self.output_dir / "analysis_data.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Analysis data saved to: {output_path}")

def main():
    """Run the analysis."""
    print("="*60)
    print("INSTAGRAM VIDEO PERFORMANCE ANALYSIS")
    print("Comparing 100k+ views vs 15k views")
    print("="*60)
    
    # URLs
    url1 = "https://www.instagram.com/reel/DAoX3WEp1Su/"  # 100k+ views
    url2 = "https://www.instagram.com/reel/DAPFyzbyt91/"  # 15k views
    
    analyzer = InstagramAnalyzer()
    
    # Download videos
    print("\n1. DOWNLOADING VIDEOS...")
    video1_info = analyzer.download_video(url1, "video1_100k")
    video2_info = analyzer.download_video(url2, "video2_15k")
    
    if not video1_info or not video2_info:
        print("\nError: Could not download videos. Make sure:")
        print("1. You have Chrome installed")
        print("2. You are logged into Instagram in Chrome")
        print("3. The URLs are correct")
        return
    
    # Analyze videos
    print("\n2. ANALYZING VIDEOS...")
    analysis1 = analyzer.analyze_video(video1_info['path'], "Video 1 (100k+ views)")
    analysis2 = analyzer.analyze_video(video2_info['path'], "Video 2 (15k views)")
    
    if not analysis1 or not analysis2:
        print("\nError: Could not analyze videos")
        return
    
    # Create visualization
    print("\n3. CREATING COMPARISON...")
    analyzer.create_comparison_visualization(analysis1, analysis2)
    
    # Save data
    analyzer.save_analysis_data(video1_info, video2_info, analysis1, analysis2)
    
    # Print results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nVIDEO 1 (100k+ views):")
    print(f"  • Duration: {analysis1['duration']:.2f}s")
    print(f"  • Total cuts: {analysis1['shot_changes']}")
    print(f"  • Avg shot: {analysis1['avg_shot_duration']:.2f}s")
    print(f"  • Hook cuts: {analysis1['hook_analysis']['cuts_in_first_3s']} in first 3s")
    print(f"  • Face visibility: {analysis1['face_visibility_pct']:.1f}%")
    print(f"  • Micro-edits: {len(analysis1['micro_edits'])}")
    
    print(f"\nVIDEO 2 (15k views):")
    print(f"  • Duration: {analysis2['duration']:.2f}s")
    print(f"  • Total cuts: {analysis2['shot_changes']}")
    print(f"  • Avg shot: {analysis2['avg_shot_duration']:.2f}s")
    print(f"  • Hook cuts: {analysis2['hook_analysis']['cuts_in_first_3s']} in first 3s")
    print(f"  • Face visibility: {analysis2['face_visibility_pct']:.1f}%")
    print(f"  • Micro-edits: {len(analysis2['micro_edits'])}")
    
    print("\n🎯 KEY DIFFERENCES:")
    
    # Hook comparison
    hook_diff = analysis1['hook_analysis']['cuts_in_first_3s'] - analysis2['hook_analysis']['cuts_in_first_3s']
    if hook_diff > 0:
        print(f"  ✓ 100k+ video has {hook_diff} MORE cuts in the hook")
    elif hook_diff < 0:
        print(f"  ✗ 100k+ video has {abs(hook_diff)} FEWER cuts in the hook")
    else:
        print(f"  = Same number of hook cuts")
    
    # Pacing comparison
    if analysis1['avg_shot_duration'] < analysis2['avg_shot_duration']:
        pct = (analysis2['avg_shot_duration'] / analysis1['avg_shot_duration'] - 1) * 100
        print(f"  ✓ 100k+ video is {pct:.0f}% FASTER paced")
    else:
        pct = (analysis1['avg_shot_duration'] / analysis2['avg_shot_duration'] - 1) * 100
        print(f"  ✗ 100k+ video is {pct:.0f}% SLOWER paced")
    
    # Motion comparison
    if analysis1['avg_motion'] > analysis2['avg_motion']:
        ratio = analysis1['avg_motion'] / analysis2['avg_motion']
        print(f"  ✓ 100k+ video has {ratio:.1f}x MORE motion energy")
    
    # Visual dynamics
    if analysis1['brightness_variance'] > analysis2['brightness_variance']:
        ratio = analysis1['brightness_variance'] / analysis2['brightness_variance']
        print(f"  ✓ 100k+ video has {ratio:.1f}x MORE visual dynamics")
    
    print("\n💡 RECOMMENDATIONS FOR VIRAL SUCCESS:")
    print("  1. Strong hook with 2-3 cuts in first 3 seconds")
    print("  2. Fast pacing with shots under 2 seconds")
    print("  3. High motion energy throughout")
    print("  4. Dynamic visual changes (lighting/color)")
    print("  5. Face visibility for human connection")
    print("  6. Smooth micro-transitions between shots")
    
    print(f"\n✅ Analysis complete! Check {analyzer.output_dir} for results.")

if __name__ == "__main__":
    main()