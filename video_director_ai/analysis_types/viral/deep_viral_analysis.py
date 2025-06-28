#!/usr/bin/env python3
"""
Deep Viral Analysis - Finding the subtle differences that made 10x views
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_micro_edits(video_path, label):
    """Analyze micro-level editing differences."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    analysis = {
        'label': label,
        'micro_cuts': [],
        'rhythm_changes': [],
        'color_shifts': [],
        'zoom_effects': [],
        'transition_types': []
    }
    
    prev_frame = None
    prev_hsv = None
    frame_count = 0
    
    print(f"\nDeep analysis of {label}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_count / fps
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if prev_frame is not None:
            # 1. Micro-cut detection (smaller threshold)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            diff_score = np.mean(diff)
            
            if 15 < diff_score < 40:  # Micro cuts/transitions
                analysis['micro_cuts'].append({
                    'timestamp': timestamp,
                    'intensity': diff_score,
                    'type': 'soft_cut'
                })
                
            # 2. Color shift detection
            if prev_hsv is not None:
                hue_shift = np.mean(np.abs(hsv[:,:,0].astype(float) - prev_hsv[:,:,0].astype(float)))
                sat_shift = np.mean(np.abs(hsv[:,:,1].astype(float) - prev_hsv[:,:,1].astype(float)))
                
                if hue_shift > 5 or sat_shift > 10:
                    analysis['color_shifts'].append({
                        'timestamp': timestamp,
                        'hue_change': hue_shift,
                        'saturation_change': sat_shift
                    })
                    
            # 3. Zoom/Scale detection
            if frame_count % 5 == 0 and prev_frame is not None:
                # Simple zoom detection via feature matching
                orb = cv2.ORB_create(nfeatures=50)
                kp1, des1 = orb.detectAndCompute(prev_gray, None)
                kp2, des2 = orb.detectAndCompute(gray, None)
                
                if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    
                    if len(matches) > 5:
                        # Calculate scale change
                        scale_changes = []
                        for i in range(min(5, len(matches))):
                            for j in range(i+1, min(5, len(matches))):
                                p1_1 = kp1[matches[i].queryIdx].pt
                                p2_1 = kp1[matches[j].queryIdx].pt
                                dist1 = np.sqrt((p1_1[0] - p2_1[0])**2 + (p1_1[1] - p2_1[1])**2)
                                
                                p1_2 = kp2[matches[i].trainIdx].pt
                                p2_2 = kp2[matches[j].trainIdx].pt
                                dist2 = np.sqrt((p1_2[0] - p2_2[0])**2 + (p1_2[1] - p2_2[1])**2)
                                
                                if dist1 > 0:
                                    scale_changes.append(dist2 / dist1)
                                    
                        if scale_changes:
                            avg_scale = np.mean(scale_changes)
                            if abs(1 - avg_scale) > 0.02:
                                analysis['zoom_effects'].append({
                                    'timestamp': timestamp,
                                    'scale': avg_scale,
                                    'type': 'zoom_in' if avg_scale > 1 else 'zoom_out'
                                })
                                
        prev_frame = frame
        prev_hsv = hsv
        frame_count += 1
        
    cap.release()
    return analysis

def analyze_beat_sync(cuts1, cuts2):
    """Analyze if cuts are synchronized to beat/rhythm."""
    # Calculate inter-cut intervals
    intervals1 = [cuts1[i+1]['timestamp'] - cuts1[i]['timestamp'] 
                  for i in range(len(cuts1)-1)]
    intervals2 = [cuts2[i+1]['timestamp'] - cuts2[i]['timestamp'] 
                  for i in range(len(cuts2)-1)]
    
    # Check for rhythmic patterns
    rhythm1 = np.std(intervals1) if intervals1 else 999
    rhythm2 = np.std(intervals2) if intervals2 else 999
    
    # Find common beat intervals (0.5s, 1s, etc.)
    beat_sync1 = sum(1 for i in intervals1 if abs(i - round(i*2)/2) < 0.1) / len(intervals1) if intervals1 else 0
    beat_sync2 = sum(1 for i in intervals2 if abs(i - round(i*2)/2) < 0.1) / len(intervals2) if intervals2 else 0
    
    return {
        'rhythm_consistency': {
            'video1': rhythm1,
            'video2': rhythm2,
            'more_consistent': '2' if rhythm2 < rhythm1 else '1'
        },
        'beat_sync': {
            'video1': beat_sync1 * 100,
            'video2': beat_sync2 * 100,
            'difference': (beat_sync2 - beat_sync1) * 100
        }
    }

def analyze_hook_timing(analysis1, analysis2):
    """Analyze the first 3 seconds - the crucial hook."""
    hook_analysis = {
        'video1_hook': {
            'cuts_in_first_3s': 0,
            'motion_peaks': 0,
            'color_changes': 0
        },
        'video2_hook': {
            'cuts_in_first_3s': 0,
            'motion_peaks': 0,
            'color_changes': 0
        }
    }
    
    # Count events in first 3 seconds
    for cut in analysis1.get('cuts', []):
        if cut['timestamp'] <= 3:
            hook_analysis['video1_hook']['cuts_in_first_3s'] += 1
            
    for cut in analysis2.get('cuts', []):
        if cut['timestamp'] <= 3:
            hook_analysis['video2_hook']['cuts_in_first_3s'] += 1
            
    return hook_analysis

def create_detailed_comparison(data1, data2, cuts1, cuts2):
    """Create detailed comparison visualization."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Deep Viral Analysis: The Subtle Differences', fontsize=16, fontweight='bold')
    
    # 1. Micro-edits timeline
    ax = axes[0, 0]
    
    # Plot micro cuts
    for mc in data1['micro_cuts']:
        ax.scatter(mc['timestamp'], 1, color='red', alpha=0.5, s=mc['intensity'])
    for mc in data2['micro_cuts']:
        ax.scatter(mc['timestamp'], 2, color='green', alpha=0.5, s=mc['intensity'])
        
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['8.8M', '87.5M'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Micro-Edits & Transitions')
    ax.text(0.02, 0.95, f"Video 1: {len(data1['micro_cuts'])} micro-edits", 
            transform=ax.transAxes, color='red', fontsize=9)
    ax.text(0.02, 0.88, f"Video 2: {len(data2['micro_cuts'])} micro-edits", 
            transform=ax.transAxes, color='green', fontsize=9)
    
    # 2. Color shifts
    ax = axes[0, 1]
    
    times1 = [cs['timestamp'] for cs in data1['color_shifts']]
    intensity1 = [cs['hue_change'] + cs['saturation_change'] for cs in data1['color_shifts']]
    times2 = [cs['timestamp'] for cs in data2['color_shifts']]
    intensity2 = [cs['hue_change'] + cs['saturation_change'] for cs in data2['color_shifts']]
    
    if times1:
        ax.scatter(times1, intensity1, color='red', alpha=0.6, label='8.8M')
    if times2:
        ax.scatter(times2, intensity2, color='green', alpha=0.6, label='87.5M')
        
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Color Change Intensity')
    ax.set_title('Color Grading Shifts')
    ax.legend()
    
    # 3. Zoom effects
    ax = axes[1, 0]
    
    zoom1 = [z for z in data1['zoom_effects']]
    zoom2 = [z for z in data2['zoom_effects']]
    
    for z in zoom1:
        color = 'blue' if z['type'] == 'zoom_in' else 'orange'
        ax.scatter(z['timestamp'], 1, color=color, s=abs(1-z['scale'])*1000, alpha=0.7)
    for z in zoom2:
        color = 'blue' if z['type'] == 'zoom_in' else 'orange'
        ax.scatter(z['timestamp'], 2, color=color, s=abs(1-z['scale'])*1000, alpha=0.7)
        
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['8.8M', '87.5M'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Zoom/Scale Effects (Blue=In, Orange=Out)')
    
    # 4. Beat sync analysis
    ax = axes[1, 1]
    
    # Inter-cut intervals histogram
    intervals1 = [cuts1[i+1]['timestamp'] - cuts1[i]['timestamp'] 
                  for i in range(len(cuts1)-1)]
    intervals2 = [cuts2[i+1]['timestamp'] - cuts2[i]['timestamp'] 
                  for i in range(len(cuts2)-1)]
    
    if intervals1 and intervals2:
        bins = np.linspace(0, 2, 20)
        ax.hist(intervals1, bins=bins, alpha=0.5, color='red', label='8.8M')
        ax.hist(intervals2, bins=bins, alpha=0.5, color='green', label='87.5M')
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3, label='Beat markers')
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Cut Interval (seconds)')
        ax.set_ylabel('Count')
        ax.set_title('Cut Rhythm Distribution')
        ax.legend()
    
    # 5. First 3 seconds analysis
    ax = axes[2, 0]
    
    # Zoom into first 3 seconds
    ax.set_xlim(0, 3)
    
    # Plot all events in first 3 seconds
    for cut in cuts1:
        if cut['timestamp'] <= 3:
            ax.axvline(x=cut['timestamp'], color='red', alpha=0.7, linewidth=2)
    for cut in cuts2:
        if cut['timestamp'] <= 3:
            ax.axvline(x=cut['timestamp'], color='green', alpha=0.7, linewidth=2, linestyle='--')
            
    ax.set_xlabel('Time (seconds)')
    ax.set_title('First 3 Seconds - The Hook')
    ax.text(0.5, 0.9, 'Critical engagement window', transform=ax.transAxes, 
            ha='center', fontsize=12, style='italic')
    
    # 6. Summary insights
    ax = axes[2, 1]
    ax.axis('off')
    
    beat_analysis = analyze_beat_sync(cuts1, cuts2)
    
    insights = f"""
🔍 DEEP ANALYSIS INSIGHTS:

MICRO-EDITS:
Video 1: {len(data1['micro_cuts'])} subtle transitions
Video 2: {len(data2['micro_cuts'])} subtle transitions
Δ = {len(data2['micro_cuts']) - len(data1['micro_cuts'])} more micro-edits

COLOR DYNAMICS:
Video 1: {len(data1['color_shifts'])} color shifts
Video 2: {len(data2['color_shifts'])} color shifts
Δ = {len(data2['color_shifts']) - len(data1['color_shifts'])} more color changes

ZOOM/SCALE EFFECTS:
Video 1: {len(data1['zoom_effects'])} zoom effects
Video 2: {len(data2['zoom_effects'])} zoom effects

RHYTHM & TIMING:
Beat Sync Score:
- Video 1: {beat_analysis['beat_sync']['video1']:.1f}%
- Video 2: {beat_analysis['beat_sync']['video2']:.1f}%
{('✓ BETTER BEAT SYNC!' if beat_analysis['beat_sync']['difference'] > 5 else '')}

🎯 VIRAL FACTORS:
{generate_insights(data1, data2, beat_analysis)}
    """
    
    ax.text(0.05, 0.95, insights, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig

def generate_insights(data1, data2, beat_analysis):
    """Generate specific insights about viral success."""
    insights = []
    
    # Micro-edits
    if len(data2['micro_cuts']) > len(data1['micro_cuts']) * 1.2:
        insights.append("✓ MORE SUBTLE TRANSITIONS: Smoother flow")
        
    # Color dynamics
    if len(data2['color_shifts']) > len(data1['color_shifts']):
        insights.append("✓ DYNAMIC COLOR GRADING: More visual interest")
        
    # Zoom effects
    if len(data2['zoom_effects']) > len(data1['zoom_effects']):
        insights.append("✓ ZOOM DYNAMICS: Creates energy and focus")
        
    # Beat sync
    if beat_analysis['beat_sync']['difference'] > 10:
        insights.append("✓ BEAT-SYNCED CUTS: Better audio-visual sync")
        
    if not insights:
        insights.append("⚡ Similar technical execution")
        insights.append("🎵 Success likely due to AUDIO differences!")
        
    return '\n'.join(insights)

def main():
    """Run deep viral analysis."""
    print("="*60)
    print("DEEP VIRAL ANALYSIS")
    print("Finding the subtle differences...")
    print("="*60)
    
    # Load previous analysis
    with open("viral_comparison/viral_comparison_data.json", 'r') as f:
        prev_data = json.load(f)
        
    # Video paths
    video1_path = "viral_comparison/Video by mike850822_C7L-cFbvLuo.mp4"
    video2_path = "viral_comparison/Video by mike850822_DJBxRgNzBtl.mp4"
    
    # Deep analysis
    deep1 = analyze_micro_edits(video1_path, "Video 1 (8.8M)")
    deep2 = analyze_micro_edits(video2_path, "Video 2 (87.5M)")
    
    # Create visualization
    fig = create_detailed_comparison(
        deep1, deep2,
        prev_data['video1']['analysis']['cuts'],
        prev_data['video2']['analysis']['cuts']
    )
    
    output_path = Path("viral_comparison") / "deep_viral_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Deep analysis complete!")
    print(f"Visualization saved to: {output_path}")
    
    # Print final insights
    print("\n" + "="*60)
    print("🎯 FINAL VIRAL SUCCESS FACTORS:")
    print("="*60)
    
    print(f"\n1. MICRO-EDITS: {len(deep2['micro_cuts']) - len(deep1['micro_cuts'])} more subtle transitions")
    print(f"2. COLOR SHIFTS: {len(deep2['color_shifts']) - len(deep1['color_shifts'])} more color changes")
    print(f"3. ZOOM EFFECTS: {len(deep2['zoom_effects'])} vs {len(deep1['zoom_effects'])}")
    
    print("\n🎵 CRITICAL INSIGHT: With similar cut counts, the viral video")
    print("   likely succeeded through AUDIO (music timing, sound effects)")
    print("   and MICRO-EDITS that create smoother flow!")
    
    # Save deep analysis
    with open(Path("viral_comparison") / "deep_analysis_data.json", 'w') as f:
        json.dump({
            'video1_deep': deep1,
            'video2_deep': deep2
        }, f, indent=2)


if __name__ == "__main__":
    main()