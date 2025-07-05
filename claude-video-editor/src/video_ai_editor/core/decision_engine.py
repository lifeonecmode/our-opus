#!/usr/bin/env python3
"""
Edit Decision Engine - Converts analysis data into editing decisions
Uses AI-driven rules to create compelling edits
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class EditDecision:
    """Single editing decision"""
    type: str  # emphasis, cut, transition, text_overlay, effect, speed_change
    timestamp: float
    duration: float
    layer: str
    priority: int  # 1-10, higher = more important
    parameters: Dict[str, Any]
    reason: str  # Why this edit was chosen

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "layer": self.layer,
            "priority": self.priority,
            "parameters": self.parameters,
            "reason": self.reason
        }


class EditDecisionEngine:
    """
    Converts frame-by-frame analysis into editing decisions
    Implements various editing styles and techniques
    """
    
    def __init__(self):
        # Style-specific rules
        self.style_rules = {
            "viral": self._viral_rules,
            "cinematic": self._cinematic_rules,
            "documentary": self._documentary_rules,
            "fast_paced": self._fast_paced_rules
        }
        
        # Platform specifications
        self.platform_specs = {
            "tiktok": {
                "max_duration": 60,
                "optimal_duration": 15,
                "hook_duration": 3,
                "attention_span": 2
            },
            "instagram": {
                "max_duration": 90,
                "optimal_duration": 30,
                "hook_duration": 3,
                "attention_span": 3
            },
            "youtube": {
                "max_duration": 60,
                "optimal_duration": 45,
                "hook_duration": 5,
                "attention_span": 5
            }
        }
    
    def generate_decisions(self, analysis_data: Dict[str, Any], 
                         style: str = "viral",
                         platform: str = "tiktok") -> List[Dict]:
        """
        Generate editing decisions from analysis data
        
        Args:
            analysis_data: Frame-by-frame analysis
            style: Editing style
            platform: Target platform
            
        Returns:
            List of editing decisions
        """
        logger.info(f"Generating {style} edit decisions for {platform}")
        
        # Extract key data
        frames = analysis_data.get('frame_data', [])
        scenes = analysis_data.get('scenes', [])
        metadata = analysis_data.get('metadata', {})
        
        if not frames:
            logger.warning("No frame data available")
            return []
        
        # Initialize decisions list
        decisions = []
        
        # Apply style-specific rules
        style_func = self.style_rules.get(style, self._viral_rules)
        decisions.extend(style_func(frames, scenes, metadata))
        
        # Add platform-specific optimizations
        decisions.extend(self._platform_optimizations(
            frames, platform, self.platform_specs[platform]
        ))
        
        # Add universal techniques
        decisions.extend(self._universal_edits(frames, scenes))
        
        # Sort by timestamp and priority
        decisions.sort(key=lambda x: (x.timestamp, -x.priority))
        
        # Resolve conflicts
        decisions = self._resolve_conflicts(decisions)
        
        # Convert to dict format
        return [d.to_dict() for d in decisions]
    
    def _viral_rules(self, frames: List[Dict], scenes: List[Dict], 
                    metadata: Dict) -> List[EditDecision]:
        """Generate viral-style editing decisions"""
        decisions = []
        
        # 1. Hook Creation (first 3 seconds)
        decisions.append(EditDecision(
            type="text_overlay",
            timestamp=0.5,
            duration=2.5,
            layer="Text",
            priority=10,
            parameters={
                "text": "WAIT FOR IT...",
                "style": "bold_yellow",
                "animation": "bounce_in",
                "position": "center"
            },
            reason="Viral hook - create anticipation"
        ))
        
        # 2. Emotional peaks - add zoom emphasis
        emotional_peaks = self._find_emotional_peaks(frames)
        for peak_time, emotion, intensity in emotional_peaks[:5]:  # Top 5 peaks
            decisions.append(EditDecision(
                type="emphasis",
                timestamp=peak_time,
                duration=0.8,
                layer="Source Video",
                priority=8,
                parameters={
                    "effect": "punch_zoom",
                    "scale": 110 + (intensity * 10),
                    "shake": intensity > 0.8
                },
                reason=f"Emotional peak: {emotion} at {intensity:.0%}"
            ))
        
        # 3. Low energy sections - speed up
        low_energy_sections = self._find_low_energy_sections(frames)
        for start, end in low_energy_sections:
            if end - start > 2:  # Only speed up sections > 2 seconds
                decisions.append(EditDecision(
                    type="speed_change",
                    timestamp=start,
                    duration=end - start,
                    layer="Source Video",
                    priority=6,
                    parameters={
                        "speed": 1.5,
                        "maintain_pitch": True
                    },
                    reason="Low energy section - increase pace"
                ))
        
        # 4. Pattern interrupts
        attention_drops = self._detect_attention_drops(frames)
        for drop_time in attention_drops[:3]:  # Max 3 interrupts
            decisions.append(EditDecision(
                type="effect",
                timestamp=drop_time,
                duration=0.5,
                layer="Global",
                priority=7,
                parameters={
                    "effect": "glitch",
                    "intensity": 0.7
                },
                reason="Pattern interrupt to regain attention"
            ))
        
        # 5. Social proof moments (if faces detected)
        face_moments = self._find_face_moments(frames)
        for moment in face_moments[:2]:  # Show 2 best moments
            decisions.append(EditDecision(
                type="text_overlay",
                timestamp=moment['timestamp'],
                duration=2,
                layer="Text",
                priority=6,
                parameters={
                    "text": "CELEBRITIES LOVE THIS",
                    "style": "trust_yellow",
                    "position": "bottom_third"
                },
                reason="Social proof overlay"
            ))
        
        # 6. CTA at end
        video_duration = metadata.get('duration', 30)
        decisions.append(EditDecision(
            type="text_overlay",
            timestamp=video_duration - 3,
            duration=3,
            layer="Text",
            priority=9,
            parameters={
                "text": "FOLLOW FOR MORE",
                "style": "cta_style",
                "animation": "pulse",
                "arrow": True
            },
            reason="Call to action"
        ))
        
        return decisions
    
    def _cinematic_rules(self, frames: List[Dict], scenes: List[Dict], 
                        metadata: Dict) -> List[EditDecision]:
        """Generate cinematic-style editing decisions"""
        decisions = []
        
        # 1. Letterbox
        decisions.append(EditDecision(
            type="effect",
            timestamp=0,
            duration=metadata.get('duration', 30),
            layer="Global",
            priority=9,
            parameters={
                "effect": "letterbox",
                "ratio": "2.35:1"
            },
            reason="Cinematic aspect ratio"
        ))
        
        # 2. Color grading
        decisions.append(EditDecision(
            type="effect",
            timestamp=0,
            duration=metadata.get('duration', 30),
            layer="Global",
            priority=8,
            parameters={
                "effect": "color_grade",
                "style": "teal_orange",
                "intensity": 0.6
            },
            reason="Cinematic color grading"
        ))
        
        # 3. Slow motion on dramatic moments
        dramatic_moments = self._find_dramatic_moments(frames)
        for moment in dramatic_moments[:3]:
            decisions.append(EditDecision(
                type="speed_change",
                timestamp=moment['timestamp'],
                duration=2,
                layer="Source Video",
                priority=7,
                parameters={
                    "speed": 0.5,
                    "interpolation": "optical_flow"
                },
                reason="Dramatic slow motion"
            ))
        
        # 4. Smooth transitions between scenes
        for i, scene in enumerate(scenes[1:], 1):
            decisions.append(EditDecision(
                type="transition",
                timestamp=scene['start_time'] - 0.5,
                duration=1,
                layer="Source Video",
                priority=6,
                parameters={
                    "type": "cross_dissolve",
                    "ease": "ease_in_out"
                },
                reason="Scene transition"
            ))
        
        return decisions
    
    def _platform_optimizations(self, frames: List[Dict], platform: str, 
                              specs: Dict) -> List[EditDecision]:
        """Add platform-specific optimizations"""
        decisions = []
        
        # Hook optimization based on platform
        hook_style = {
            "tiktok": {"text": "POV:", "style": "tiktok_text"},
            "instagram": {"text": "You won't believe...", "style": "instagram_text"},
            "youtube": {"text": "Here's why...", "style": "youtube_text"}
        }
        
        if platform in hook_style:
            decisions.append(EditDecision(
                type="text_overlay",
                timestamp=0,
                duration=specs['hook_duration'],
                layer="Text",
                priority=9,
                parameters=hook_style[platform],
                reason=f"{platform} specific hook"
            ))
        
        # Attention retention checkpoints
        checkpoint_interval = specs['attention_span']
        video_duration = frames[-1]['timestamp'] if frames else 30
        
        for checkpoint in range(checkpoint_interval, 
                               int(video_duration), 
                               checkpoint_interval):
            decisions.append(EditDecision(
                type="effect",
                timestamp=checkpoint,
                duration=0.3,
                layer="Global",
                priority=5,
                parameters={
                    "effect": "flash_frame",
                    "intensity": 0.3
                },
                reason="Attention retention checkpoint"
            ))
        
        return decisions
    
    def _universal_edits(self, frames: List[Dict], scenes: List[Dict]) -> List[EditDecision]:
        """Add universal editing techniques"""
        decisions = []
        
        # 1. Motion matching cuts
        motion_matches = self._find_motion_matches(frames)
        for match in motion_matches[:5]:
            decisions.append(EditDecision(
                type="cut",
                timestamp=match['timestamp'],
                duration=0,
                layer="Source Video",
                priority=4,
                parameters={
                    "type": "match_cut",
                    "motion_vector": match['vector']
                },
                reason="Motion matched cut"
            ))
        
        # 2. Rule of thirds reframing
        poor_composition = self._find_poor_composition(frames)
        for frame in poor_composition[:3]:
            decisions.append(EditDecision(
                type="effect",
                timestamp=frame['timestamp'],
                duration=3,
                layer="Source Video",
                priority=3,
                parameters={
                    "effect": "reframe",
                    "adjustment": frame['adjustment']
                },
                reason="Composition improvement"
            ))
        
        return decisions
    
    def _find_emotional_peaks(self, frames: List[Dict]) -> List[Tuple[float, str, float]]:
        """Find emotional peaks in the video"""
        peaks = []
        
        for frame in frames:
            emotion_data = frame.get('emotional_tone', {})
            confidence = emotion_data.get('confidence', 0)
            
            if confidence > 0.7:  # High confidence emotions
                peaks.append((
                    frame['timestamp'],
                    emotion_data.get('dominant', 'unknown'),
                    confidence
                ))
        
        # Sort by confidence
        peaks.sort(key=lambda x: x[2], reverse=True)
        return peaks
    
    def _find_low_energy_sections(self, frames: List[Dict]) -> List[Tuple[float, float]]:
        """Find sections with low visual energy"""
        sections = []
        start = None
        
        for i, frame in enumerate(frames):
            score = frame.get('cinematography_score', 0.5)
            
            if score < 0.3:  # Low energy threshold
                if start is None:
                    start = frame['timestamp']
            else:
                if start is not None:
                    end = frames[i-1]['timestamp']
                    if end - start > 1:  # Minimum 1 second
                        sections.append((start, end))
                    start = None
        
        return sections
    
    def _detect_attention_drops(self, frames: List[Dict]) -> List[float]:
        """Detect where attention might drop"""
        drops = []
        
        # Simple heuristic: low motion + neutral emotion
        for i in range(10, len(frames) - 10):
            # Look at 10-frame window
            window = frames[i-5:i+5]
            
            avg_motion = np.mean([f.get('motion', {}).get('motion_score', 0) 
                                for f in window])
            avg_emotion = np.mean([f.get('emotional_tone', {}).get('confidence', 0) 
                                 for f in window])
            
            if avg_motion < 10 and avg_emotion < 0.3:
                drops.append(frames[i]['timestamp'])
        
        return drops
    
    def _find_face_moments(self, frames: List[Dict]) -> List[Dict]:
        """Find moments with clear face shots"""
        moments = []
        
        for frame in frames:
            faces = frame.get('faces', {})
            if faces.get('count', 0) > 0:
                # Check for good face shots (medium/close)
                if frame.get('shot_type') in ['close', 'medium']:
                    moments.append({
                        'timestamp': frame['timestamp'],
                        'face_count': faces['count'],
                        'shot_type': frame['shot_type']
                    })
        
        return moments
    
    def _find_dramatic_moments(self, frames: List[Dict]) -> List[Dict]:
        """Find dramatic moments for cinematic treatment"""
        moments = []
        
        for i in range(1, len(frames) - 1):
            # Look for high contrast changes
            curr = frames[i]
            prev = frames[i-1]
            
            # Check for dramatic lighting changes
            brightness_change = abs(
                curr.get('technical', {}).get('brightness', 0) - 
                prev.get('technical', {}).get('brightness', 0)
            )
            
            if brightness_change > 50:  # Significant change
                moments.append({
                    'timestamp': curr['timestamp'],
                    'type': 'lighting_change',
                    'intensity': brightness_change / 255
                })
        
        return moments
    
    def _find_motion_matches(self, frames: List[Dict]) -> List[Dict]:
        """Find opportunities for motion-matched cuts"""
        matches = []
        
        for i in range(1, len(frames) - 1):
            curr_motion = frames[i].get('motion', {})
            
            # Look for consistent motion direction
            if curr_motion.get('camera_movement', {}).get('type') in ['pan_left', 'pan_right']:
                matches.append({
                    'timestamp': frames[i]['timestamp'],
                    'vector': [curr_motion.get('flow_x', 0), 
                              curr_motion.get('flow_y', 0)]
                })
        
        return matches
    
    def _find_poor_composition(self, frames: List[Dict]) -> List[Dict]:
        """Find frames with poor composition"""
        poor_frames = []
        
        for frame in frames:
            composition = frame.get('composition', {})
            
            # Check rule of thirds score
            if composition.get('rule_of_thirds_score', 100) < 20:
                poor_frames.append({
                    'timestamp': frame['timestamp'],
                    'adjustment': self._calculate_reframe(composition)
                })
        
        return poor_frames
    
    def _calculate_reframe(self, composition: Dict) -> Dict:
        """Calculate reframing adjustment"""
        # Simple reframing logic
        weight = composition.get('visual_weight_distribution', 'balanced')
        
        adjustments = {
            'left_heavy': {'x': 10, 'y': 0},
            'right_heavy': {'x': -10, 'y': 0},
            'top_heavy': {'x': 0, 'y': 10},
            'bottom_heavy': {'x': 0, 'y': -10},
            'balanced': {'x': 0, 'y': 0}
        }
        
        return adjustments.get(weight, {'x': 0, 'y': 0})
    
    def _resolve_conflicts(self, decisions: List[EditDecision]) -> List[EditDecision]:
        """Resolve overlapping edit decisions"""
        if not decisions:
            return decisions
        
        resolved = [decisions[0]]
        
        for decision in decisions[1:]:
            # Check for overlap with last added decision
            last = resolved[-1]
            
            # If overlapping timestamps
            if (decision.timestamp < last.timestamp + last.duration and
                decision.layer == last.layer):
                
                # Keep higher priority
                if decision.priority > last.priority:
                    resolved[-1] = decision
                # If same priority, adjust timing
                elif decision.priority == last.priority:
                    decision.timestamp = last.timestamp + last.duration + 0.1
                    resolved.append(decision)
            else:
                resolved.append(decision)
        
        return resolved
    
    def _fast_paced_rules(self, frames: List[Dict], scenes: List[Dict], 
                         metadata: Dict) -> List[EditDecision]:
        """Generate fast-paced editing style"""
        decisions = []
        
        # Quick cuts every 2-3 seconds
        cut_interval = 2.5
        duration = metadata.get('duration', 30)
        
        for t in np.arange(cut_interval, duration, cut_interval):
            decisions.append(EditDecision(
                type="cut",
                timestamp=t,
                duration=0,
                layer="Source Video",
                priority=5,
                parameters={
                    "type": "hard_cut",
                    "transition": "none"
                },
                reason="Fast-paced rhythm"
            ))
        
        # Speed ramping
        decisions.append(EditDecision(
            type="speed_change",
            timestamp=0,
            duration=duration,
            layer="Source Video",
            priority=6,
            parameters={
                "speed": 1.2,
                "variable": True,
                "sync_to_beat": True
            },
            reason="Overall pace increase"
        ))
        
        return decisions
    
    def _documentary_rules(self, frames: List[Dict], scenes: List[Dict], 
                          metadata: Dict) -> List[EditDecision]:
        """Generate documentary-style editing"""
        decisions = []
        
        # Lower thirds for context
        key_moments = self._find_key_informational_moments(frames)
        for moment in key_moments[:5]:
            decisions.append(EditDecision(
                type="text_overlay",
                timestamp=moment['timestamp'],
                duration=5,
                layer="Text",
                priority=6,
                parameters={
                    "text": moment.get('context', 'Context'),
                    "style": "lower_third",
                    "subtitle": True
                },
                reason="Documentary context"
            ))
        
        # Ken Burns effect for still moments
        still_sections = self._find_still_sections(frames)
        for start, end in still_sections:
            decisions.append(EditDecision(
                type="effect",
                timestamp=start,
                duration=end - start,
                layer="Source Video",
                priority=5,
                parameters={
                    "effect": "ken_burns",
                    "start_scale": 100,
                    "end_scale": 110,
                    "pan_direction": "center_out"
                },
                reason="Ken Burns for visual interest"
            ))
        
        return decisions
    
    def _find_key_informational_moments(self, frames: List[Dict]) -> List[Dict]:
        """Find moments that need context"""
        # Placeholder - would analyze for scene changes, new subjects, etc.
        moments = []
        
        # For now, just mark scene starts
        seen_timestamps = set()
        for frame in frames:
            if frame['timestamp'] not in seen_timestamps:
                if frame.get('shot_type') == 'wide':  # Wide shots often establish context
                    moments.append({
                        'timestamp': frame['timestamp'],
                        'context': 'Location established'
                    })
                seen_timestamps.add(frame['timestamp'])
        
        return moments[:5]  # Limit to 5
    
    def _find_still_sections(self, frames: List[Dict]) -> List[Tuple[float, float]]:
        """Find sections with little motion"""
        sections = []
        start = None
        
        for i, frame in enumerate(frames):
            motion_score = frame.get('motion', {}).get('motion_score', 0)
            
            if motion_score < 5:  # Very low motion
                if start is None:
                    start = frame['timestamp']
            else:
                if start is not None and i > 0:
                    end = frames[i-1]['timestamp']
                    if end - start > 3:  # Minimum 3 seconds
                        sections.append((start, end))
                    start = None
        
        return sections