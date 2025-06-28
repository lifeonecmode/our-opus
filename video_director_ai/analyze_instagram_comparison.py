#!/usr/bin/env python3
"""
Instagram Video Comparison Analysis
Analyzing performance difference between 100k+ and 15k view videos
"""

import sys
import asyncio
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_downloader import VideoDownloader
from director_brain_lite import DirectorBrainLite
from analysis_types.viral.viral_comparison import ViralVideoComparator
from analysis_types.viral.deep_viral_analysis import analyze_micro_edits, analyze_beat_sync, analyze_hook_timing
from settings import Config

async def main():
    """Run comprehensive comparison of two Instagram videos."""
    print("="*60)
    print("INSTAGRAM VIDEO PERFORMANCE ANALYSIS")
    print("100k+ views vs 15k views - Finding the difference")
    print("="*60)
    
    # Video URLs
    url1 = "https://www.instagram.com/reel/DAoX3WEp1Su/"  # 100k+ views
    url2 = "https://www.instagram.com/reel/DAPFyzbyt91/"  # 15k views
    
    # Create output directory
    output_dir = Path("instagram_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tools
    downloader = VideoDownloader(output_dir)
    director = DirectorBrainLite()
    comparator = ViralVideoComparator()
    
    print("\n1. Downloading videos...")
    try:
        # Download video 1
        print(f"\nDownloading Video 1 (100k+ views): {url1}")
        video1_info = downloader.download(url1)
        print(f"✓ Downloaded: {video1_info['title']}")
        
        # Download video 2
        print(f"\nDownloading Video 2 (15k views): {url2}")
        video2_info = downloader.download(url2)
        print(f"✓ Downloaded: {video2_info['title']}")
        
    except Exception as e:
        print(f"Error downloading videos: {e}")
        print("\nMake sure you have Chrome installed and are logged into Instagram.")
        return
    
    # Analyze both videos
    print("\n2. Running cinematographic analysis...")
    
    # Basic viral analysis
    print("\n   Analyzing Video 1 (100k+ views)...")
    analysis1 = comparator.analyze_video(video1_info['file_path'], "Video 1 (100k+ views)")
    
    print("\n   Analyzing Video 2 (15k views)...")
    analysis2 = comparator.analyze_video(video2_info['file_path'], "Video 2 (15k views)")
    
    # Deep micro-edit analysis
    print("\n3. Performing deep viral analysis...")
    deep1 = analyze_micro_edits(video1_info['file_path'], "Video 1 (100k+)")
    deep2 = analyze_micro_edits(video2_info['file_path'], "Video 2 (15k)")
    
    # Director brain analysis for detailed insights
    print("\n4. Running director-level analysis...")
    director_results1 = await director.analyze_video(video1_info['file_path'], output_format='json')
    director_results2 = await director.analyze_video(video2_info['file_path'], output_format='json')
    
    # Compare results
    print("\n5. Comparing viral factors...")
    comparison = comparator.compare_videos(analysis1, analysis2)
    
    # Analyze beat sync
    beat_analysis = analyze_beat_sync(analysis1['cuts'], analysis2['cuts'])
    
    # Hook analysis
    hook_analysis = analyze_hook_timing(analysis1, analysis2)
    
    # Create comprehensive visualization
    create_comprehensive_comparison(
        analysis1, analysis2, 
        deep1, deep2, 
        director_results1, director_results2,
        comparison, beat_analysis, hook_analysis,
        output_dir
    )
    
    # Save all analysis data
    save_analysis_data(
        video1_info, video2_info,
        analysis1, analysis2,
        deep1, deep2,
        director_results1, director_results2,
        comparison, beat_analysis, hook_analysis,
        output_dir
    )
    
    # Print insights
    print_viral_insights(
        analysis1, analysis2,
        deep1, deep2,
        director_results1, director_results2,
        comparison, beat_analysis, hook_analysis
    )

def create_comprehensive_comparison(analysis1, analysis2, deep1, deep2, 
                                  director1, director2, comparison, 
                                  beat_analysis, hook_analysis, output_dir):
    """Create detailed comparison visualization."""
    fig = plt.figure(figsize=(20, 16))
    
    # Create main title
    fig.suptitle('Instagram Video Performance Analysis: 100k+ vs 15k Views', 
                 fontsize=20, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Hook Analysis (First 3 seconds)
    ax1 = fig.add_subplot(gs[0, :])
    plot_hook_analysis(ax1, analysis1, analysis2, deep1, deep2)
    
    # 2. Cut Frequency
    ax2 = fig.add_subplot(gs[1, 0])
    plot_cut_frequency(ax2, analysis1, analysis2)
    
    # 3. Motion Intensity
    ax3 = fig.add_subplot(gs[1, 1])
    plot_motion_intensity(ax3, analysis1, analysis2)
    
    # 4. Face Presence
    ax4 = fig.add_subplot(gs[1, 2])
    plot_face_presence(ax4, analysis1, analysis2)
    
    # 5. Visual Energy
    ax5 = fig.add_subplot(gs[2, 0])
    plot_visual_energy(ax5, analysis1, analysis2)
    
    # 6. Micro-edits
    ax6 = fig.add_subplot(gs[2, 1])
    plot_micro_edits(ax6, deep1, deep2)
    
    # 7. Beat Sync
    ax7 = fig.add_subplot(gs[2, 2])
    plot_beat_sync(ax7, beat_analysis, analysis1, analysis2)
    
    # 8. Key Insights
    ax8 = fig.add_subplot(gs[3, :])
    plot_key_insights(ax8, comparison, beat_analysis, hook_analysis, 
                     analysis1, analysis2, deep1, deep2)
    
    plt.tight_layout()
    output_path = output_dir / "comprehensive_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualization saved to: {output_path}")

def plot_hook_analysis(ax, analysis1, analysis2, deep1, deep2):
    """Detailed hook analysis for first 3 seconds."""
    ax.set_xlim(0, 3)
    ax.set_title('First 3 Seconds - The Critical Hook', fontsize=14, fontweight='bold')
    
    # Plot cuts
    for cut in analysis1['cuts']:
        if cut['timestamp'] <= 3:
            ax.axvline(x=cut['timestamp'], color='red', alpha=0.7, linewidth=2, 
                      label='100k+ cuts' if cut == analysis1['cuts'][0] else '')
    
    for cut in analysis2['cuts']:
        if cut['timestamp'] <= 3:
            ax.axvline(x=cut['timestamp'], color='blue', alpha=0.7, linewidth=2, 
                      linestyle='--', label='15k cuts' if cut == analysis2['cuts'][0] else '')
    
    # Plot micro-edits
    for mc in deep1['micro_cuts']:
        if mc['timestamp'] <= 3:
            ax.scatter(mc['timestamp'], 0.5, color='red', alpha=0.3, s=mc['intensity']*2)
    
    for mc in deep2['micro_cuts']:
        if mc['timestamp'] <= 3:
            ax.scatter(mc['timestamp'], 0.3, color='blue', alpha=0.3, s=mc['intensity']*2)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    
    # Add annotations
    cuts_100k = sum(1 for cut in analysis1['cuts'] if cut['timestamp'] <= 3)
    cuts_15k = sum(1 for cut in analysis2['cuts'] if cut['timestamp'] <= 3)
    ax.text(1.5, 0.9, f"100k+: {cuts_100k} cuts in first 3s", ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(1.5, 0.75, f"15k: {cuts_15k} cuts in first 3s", ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

def plot_cut_frequency(ax, analysis1, analysis2):
    """Plot cut frequency comparison."""
    ax.bar(['100k+ views', '15k views'], 
           [analysis1['shot_changes'], analysis2['shot_changes']], 
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Number of Cuts')
    ax.set_title('Editing Pace')
    
    # Add average shot duration
    ax.text(0, analysis1['shot_changes'] + 1, 
            f"{analysis1['avg_shot_duration']:.2f}s/shot", 
            ha='center', fontsize=10)
    ax.text(1, analysis2['shot_changes'] + 1, 
            f"{analysis2['avg_shot_duration']:.2f}s/shot", 
            ha='center', fontsize=10)

def plot_motion_intensity(ax, analysis1, analysis2):
    """Plot motion intensity over time."""
    times1 = [m['timestamp'] for m in analysis1['motion_intensity']]
    motion1 = [m['score'] for m in analysis1['motion_intensity']]
    times2 = [m['timestamp'] for m in analysis2['motion_intensity']]
    motion2 = [m['score'] for m in analysis2['motion_intensity']]
    
    ax.plot(times1, motion1, color='red', alpha=0.7, label='100k+ views')
    ax.plot(times2, motion2, color='blue', alpha=0.7, label='15k views')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motion Score')
    ax.set_title('Motion Dynamics')
    ax.legend()

def plot_face_presence(ax, analysis1, analysis2):
    """Plot face detection percentage."""
    face_pct1 = sum(1 for f in analysis1['face_detection_timeline'] if f['has_face']) / len(analysis1['face_detection_timeline']) * 100
    face_pct2 = sum(1 for f in analysis2['face_detection_timeline'] if f['has_face']) / len(analysis2['face_detection_timeline']) * 100
    
    ax.bar(['100k+ views', '15k views'], [face_pct1, face_pct2], 
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Face Visibility %')
    ax.set_title('Human Connection')
    ax.set_ylim(0, 100)

def plot_visual_energy(ax, analysis1, analysis2):
    """Plot visual energy (brightness variance)."""
    bright_var1 = np.var([b['brightness'] for b in analysis1['brightness_changes']])
    bright_var2 = np.var([b['brightness'] for b in analysis2['brightness_changes']])
    
    ax.bar(['100k+ views', '15k views'], [bright_var1, bright_var2], 
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Brightness Variance')
    ax.set_title('Visual Energy')

def plot_micro_edits(ax, deep1, deep2):
    """Plot micro-edit comparison."""
    categories = ['Micro Cuts', 'Color Shifts', 'Zoom Effects']
    video1_counts = [len(deep1['micro_cuts']), len(deep1['color_shifts']), len(deep1['zoom_effects'])]
    video2_counts = [len(deep2['micro_cuts']), len(deep2['color_shifts']), len(deep2['zoom_effects'])]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, video1_counts, width, label='100k+ views', color='red', alpha=0.7)
    ax.bar(x + width/2, video2_counts, width, label='15k views', color='blue', alpha=0.7)
    
    ax.set_xlabel('Effect Type')
    ax.set_ylabel('Count')
    ax.set_title('Micro-Level Editing')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

def plot_beat_sync(ax, beat_analysis, analysis1, analysis2):
    """Plot beat synchronization analysis."""
    if beat_analysis and 'beat_sync' in beat_analysis:
        videos = ['100k+ views', '15k views']
        sync_scores = [
            beat_analysis['beat_sync']['video1'],
            beat_analysis['beat_sync']['video2']
        ]
        
        ax.bar(videos, sync_scores, color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Beat Sync Score %')
        ax.set_title('Audio-Visual Synchronization')
        ax.set_ylim(0, 100)

def plot_key_insights(ax, comparison, beat_analysis, hook_analysis, 
                     analysis1, analysis2, deep1, deep2):
    """Display key insights."""
    ax.axis('off')
    
    insights_text = f"""
🔍 KEY PERFORMANCE FACTORS:

HOOK EFFECTIVENESS (First 3 seconds):
• 100k+ video: {sum(1 for cut in analysis1['cuts'] if cut['timestamp'] <= 3)} cuts
• 15k video: {sum(1 for cut in analysis2['cuts'] if cut['timestamp'] <= 3)} cuts
{('✓ STRONGER HOOK' if sum(1 for cut in analysis1['cuts'] if cut['timestamp'] <= 3) > 
  sum(1 for cut in analysis2['cuts'] if cut['timestamp'] <= 3) else '✗ WEAKER HOOK')}

EDITING DYNAMICS:
• 100k+ video: {analysis1['shot_changes']} total cuts ({analysis1['avg_shot_duration']:.2f}s avg)
• 15k video: {analysis2['shot_changes']} total cuts ({analysis2['avg_shot_duration']:.2f}s avg)
{('✓ FASTER PACING' if analysis1['avg_shot_duration'] < analysis2['avg_shot_duration'] else '✗ SLOWER PACING')}

MICRO-LEVEL DIFFERENCES:
• Micro-edits: {len(deep1['micro_cuts'])} vs {len(deep2['micro_cuts'])}
• Color shifts: {len(deep1['color_shifts'])} vs {len(deep2['color_shifts'])}
• Zoom effects: {len(deep1['zoom_effects'])} vs {len(deep2['zoom_effects'])}

ENGAGEMENT METRICS:
• Face visibility: {comparison['engagement_factors']['video1_face_time']:.1f}% vs {comparison['engagement_factors']['video2_face_time']:.1f}%
• Motion energy: {comparison['engagement_factors']['video1_motion_avg']:.2f} vs {comparison['engagement_factors']['video2_motion_avg']:.2f}

🎯 VIRAL SUCCESS FACTORS:
{generate_viral_factors(analysis1, analysis2, deep1, deep2, comparison, beat_analysis)}
"""
    
    ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

def generate_viral_factors(analysis1, analysis2, deep1, deep2, comparison, beat_analysis):
    """Generate specific viral success factors."""
    factors = []
    
    # Hook strength
    hooks1 = sum(1 for cut in analysis1['cuts'] if cut['timestamp'] <= 3)
    hooks2 = sum(1 for cut in analysis2['cuts'] if cut['timestamp'] <= 3)
    if hooks1 > hooks2:
        factors.append(f"✓ {hooks1-hooks2} more cuts in first 3 seconds = STRONGER HOOK")
    
    # Pacing
    if analysis1['avg_shot_duration'] < analysis2['avg_shot_duration']:
        factors.append(f"✓ {(analysis2['avg_shot_duration']/analysis1['avg_shot_duration']-1)*100:.0f}% faster pacing")
    
    # Micro-edits
    if len(deep1['micro_cuts']) > len(deep2['micro_cuts']):
        factors.append(f"✓ {len(deep1['micro_cuts'])-len(deep2['micro_cuts'])} more micro-transitions")
    
    # Face time
    if comparison['engagement_factors']['video1_face_time'] > comparison['engagement_factors']['video2_face_time']:
        factors.append("✓ More face visibility = better human connection")
    
    # Visual energy
    if comparison['visual_dynamics']['video1_brightness_variance'] > comparison['visual_dynamics']['video2_brightness_variance']:
        factors.append("✓ Higher visual energy maintains attention")
    
    return '\n'.join(factors) if factors else "⚡ Success likely due to content/audio factors"

def save_analysis_data(video1_info, video2_info, analysis1, analysis2,
                      deep1, deep2, director1, director2,
                      comparison, beat_analysis, hook_analysis, output_dir):
    """Save all analysis data to JSON."""
    data = {
        'video1': {
            'info': video1_info,
            'basic_analysis': {k: v for k, v in analysis1.items() if k not in ['frame_analysis']},
            'deep_analysis': deep1,
            'director_insights': {
                'metadata': director1.get('metadata', {}),
                'synthesis': director1.get('synthesis', {}),
                'key_moments': director1.get('synthesis', {}).get('key_moments', [])[:3]
            }
        },
        'video2': {
            'info': video2_info,
            'basic_analysis': {k: v for k, v in analysis2.items() if k not in ['frame_analysis']},
            'deep_analysis': deep2,
            'director_insights': {
                'metadata': director2.get('metadata', {}),
                'synthesis': director2.get('synthesis', {}),
                'key_moments': director2.get('synthesis', {}).get('key_moments', [])[:3]
            }
        },
        'comparison': comparison,
        'beat_analysis': beat_analysis,
        'hook_analysis': hook_analysis
    }
    
    output_path = output_dir / "analysis_data.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✓ Analysis data saved to: {output_path}")

def print_viral_insights(analysis1, analysis2, deep1, deep2, 
                        director1, director2, comparison, 
                        beat_analysis, hook_analysis):
    """Print detailed insights about performance difference."""
    print("\n" + "="*60)
    print("🔥 PERFORMANCE ANALYSIS RESULTS")
    print("="*60)
    
    print("\n📊 VIDEO METRICS:")
    print(f"\nVideo 1 (100k+ views):")
    print(f"  • Duration: {analysis1['duration']:.2f}s")
    print(f"  • Total cuts: {analysis1['shot_changes']}")
    print(f"  • Avg shot duration: {analysis1['avg_shot_duration']:.2f}s")
    print(f"  • Face visibility: {comparison['engagement_factors']['video1_face_time']:.1f}%")
    
    print(f"\nVideo 2 (15k views):")
    print(f"  • Duration: {analysis2['duration']:.2f}s")
    print(f"  • Total cuts: {analysis2['shot_changes']}")
    print(f"  • Avg shot duration: {analysis2['avg_shot_duration']:.2f}s")
    print(f"  • Face visibility: {comparison['engagement_factors']['video2_face_time']:.1f}%")
    
    print("\n🎬 HOOK ANALYSIS (First 3 seconds):")
    cuts1_hook = sum(1 for cut in analysis1['cuts'] if cut['timestamp'] <= 3)
    cuts2_hook = sum(1 for cut in analysis2['cuts'] if cut['timestamp'] <= 3)
    print(f"  • 100k+ video: {cuts1_hook} cuts")
    print(f"  • 15k video: {cuts2_hook} cuts")
    if cuts1_hook > cuts2_hook:
        print(f"  ✓ The viral video has {cuts1_hook - cuts2_hook} MORE cuts in the hook!")
    
    print("\n🎯 KEY DIFFERENCES:")
    
    # Pacing difference
    if analysis1['avg_shot_duration'] < analysis2['avg_shot_duration']:
        pacing_diff = (analysis2['avg_shot_duration'] / analysis1['avg_shot_duration'] - 1) * 100
        print(f"  ✓ PACING: 100k+ video is {pacing_diff:.0f}% faster")
    
    # Micro-edits
    micro_diff = len(deep1['micro_cuts']) - len(deep2['micro_cuts'])
    if micro_diff > 0:
        print(f"  ✓ MICRO-EDITS: {micro_diff} more subtle transitions")
    
    # Visual dynamics
    bright_ratio = comparison['visual_dynamics']['video1_brightness_variance'] / comparison['visual_dynamics']['video2_brightness_variance']
    if bright_ratio > 1:
        print(f"  ✓ VISUAL ENERGY: {bright_ratio:.1f}x more dynamic")
    
    # Beat sync
    if beat_analysis and 'beat_sync' in beat_analysis:
        if beat_analysis['beat_sync']['video1'] > beat_analysis['beat_sync']['video2']:
            print(f"  ✓ BEAT SYNC: Better audio-visual synchronization")
    
    print("\n💡 RECOMMENDATIONS FOR VIRAL SUCCESS:")
    print("  1. Create a stronger hook with 2-3 cuts in the first 3 seconds")
    print("  2. Maintain faster pacing with shots under 2 seconds")
    print("  3. Use micro-transitions for smoother flow")
    print("  4. Increase face visibility for human connection")
    print("  5. Add visual dynamics through lighting and color changes")
    print("  6. Sync cuts to audio beats for better rhythm")
    
    print("\n✅ Analysis complete! Check the output folder for detailed visualizations.")

if __name__ == "__main__":
    asyncio.run(main())