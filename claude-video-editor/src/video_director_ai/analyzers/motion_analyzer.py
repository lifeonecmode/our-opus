import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from collections import deque
from ..settings import Config

logger = logging.getLogger(__name__)


class MotionAnalyzer:
    """Analyzes camera movement and motion characteristics in video frames."""
    
    def __init__(self, motion_threshold: float = None):
        self.motion_threshold = motion_threshold or Config.MOTION_THRESHOLD
        self.prev_frame = None
        self.prev_gray = None
        self.motion_history = deque(maxlen=10)  # Keep last 10 frames of motion data
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
    def analyze_motion(self, frame: np.ndarray, timestamp: float = 0) -> Dict[str, any]:
        """
        Analyze motion between current and previous frame.
        
        Args:
            frame: Current BGR frame
            timestamp: Current timestamp in seconds
            
        Returns:
            Dict containing motion analysis
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame
            return self._get_static_result()
            
        # Analyze different types of motion
        motion_data = {
            'camera_movement': self._detect_camera_movement(gray),
            'optical_flow': self._analyze_optical_flow(gray),
            'motion_intensity': self._calculate_motion_intensity(gray),
            'shake_detection': self._detect_camera_shake(gray),
            'zoom_detection': self._detect_zoom(frame),
            'subject_motion': self._detect_subject_motion(gray),
            'timestamp': timestamp
        }
        
        # Update motion history
        self.motion_history.append(motion_data)
        
        # Analyze motion patterns over time
        motion_data['motion_pattern'] = self._analyze_motion_pattern()
        
        # Update previous frame
        self.prev_gray = gray.copy()
        self.prev_frame = frame.copy()
        
        return motion_data
        
    def _detect_camera_movement(self, gray: np.ndarray) -> Dict[str, any]:
        """Detect type of camera movement."""
        # Find features to track
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
        
        if prev_pts is None or len(prev_pts) < 10:
            return {'type': 'static', 'confidence': 0.5}
            
        # Track features using optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )
        
        # Keep only good matches
        good_prev = prev_pts[status == 1]
        good_next = next_pts[status == 1]
        
        if len(good_prev) < 5:
            return {'type': 'static', 'confidence': 0.5}
            
        # Calculate average motion vector
        motion_vectors = good_next - good_prev
        avg_motion = np.mean(motion_vectors, axis=0)
        
        # Determine movement type
        horizontal_motion = abs(avg_motion[0])
        vertical_motion = abs(avg_motion[1])
        
        # Calculate homography for more advanced detection
        if len(good_prev) >= 4:
            M, mask = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 5.0)
            
            if M is not None:
                movement_type = self._classify_movement_from_homography(M, avg_motion)
            else:
                movement_type = self._classify_movement_from_vectors(horizontal_motion, vertical_motion, avg_motion)
        else:
            movement_type = self._classify_movement_from_vectors(horizontal_motion, vertical_motion, avg_motion)
            
        return movement_type
        
    def _classify_movement_from_homography(self, M: np.ndarray, avg_motion: np.ndarray) -> Dict[str, any]:
        """Classify camera movement from homography matrix."""
        # Extract scale from homography
        scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
        avg_scale = (scale_x + scale_y) / 2
        
        # Extract rotation
        rotation = np.arctan2(M[1, 0], M[0, 0])
        
        # Determine movement type
        if avg_scale > 1.02:
            return {
                'type': 'zoom_in',
                'confidence': min((avg_scale - 1) * 10, 1.0),
                'scale_factor': avg_scale,
                'description': 'Camera zooming in or dolly forward'
            }
        elif avg_scale < 0.98:
            return {
                'type': 'zoom_out',
                'confidence': min((1 - avg_scale) * 10, 1.0),
                'scale_factor': avg_scale,
                'description': 'Camera zooming out or dolly backward'
            }
        elif abs(rotation) > 0.02:
            return {
                'type': 'rotation',
                'confidence': min(abs(rotation) * 10, 1.0),
                'rotation_angle': np.degrees(rotation),
                'description': 'Camera rotation or roll'
            }
        else:
            return self._classify_movement_from_vectors(
                abs(avg_motion[0]), abs(avg_motion[1]), avg_motion
            )
            
    def _classify_movement_from_vectors(self, h_motion: float, v_motion: float, 
                                      avg_motion: np.ndarray) -> Dict[str, any]:
        """Classify movement from motion vectors."""
        if h_motion < self.motion_threshold and v_motion < self.motion_threshold:
            return {
                'type': 'static',
                'confidence': 0.9,
                'description': 'Camera is stationary'
            }
            
        if h_motion > v_motion * 2:
            if avg_motion[0] > 0:
                movement_type = 'pan_right'
                description = 'Camera panning right'
            else:
                movement_type = 'pan_left'
                description = 'Camera panning left'
            confidence = min(h_motion / 10, 1.0)
            
        elif v_motion > h_motion * 2:
            if avg_motion[1] > 0:
                movement_type = 'tilt_down'
                description = 'Camera tilting down'
            else:
                movement_type = 'tilt_up'
                description = 'Camera tilting up'
            confidence = min(v_motion / 10, 1.0)
            
        else:
            # Diagonal movement
            if avg_motion[0] > 0 and avg_motion[1] > 0:
                movement_type = 'diagonal_down_right'
            elif avg_motion[0] > 0 and avg_motion[1] < 0:
                movement_type = 'diagonal_up_right'
            elif avg_motion[0] < 0 and avg_motion[1] > 0:
                movement_type = 'diagonal_down_left'
            else:
                movement_type = 'diagonal_up_left'
                
            description = f'Camera moving {movement_type.replace("_", " ")}'
            confidence = min((h_motion + v_motion) / 20, 1.0)
            
        return {
            'type': movement_type,
            'confidence': confidence,
            'description': description,
            'motion_vector': {'x': float(avg_motion[0]), 'y': float(avg_motion[1])}
        }
        
    def _analyze_optical_flow(self, gray: np.ndarray) -> Dict[str, any]:
        """Analyze dense optical flow for overall motion patterns."""
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate flow statistics
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Analyze flow characteristics
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        
        # Detect motion regions
        motion_mask = magnitude > self.motion_threshold
        motion_percentage = np.sum(motion_mask) / motion_mask.size
        
        # Analyze flow direction
        avg_angle = np.mean(angle[motion_mask]) if np.any(motion_mask) else 0
        
        # Detect flow patterns
        flow_pattern = self._detect_flow_pattern(flow, magnitude, angle)
        
        return {
            'average_magnitude': float(avg_magnitude),
            'max_magnitude': float(max_magnitude),
            'motion_percentage': float(motion_percentage),
            'dominant_direction': float(np.degrees(avg_angle)),
            'flow_pattern': flow_pattern
        }
        
    def _detect_flow_pattern(self, flow: np.ndarray, magnitude: np.ndarray, 
                           angle: np.ndarray) -> str:
        """Detect specific optical flow patterns."""
        h, w = flow.shape[:2]
        
        # Divide frame into regions
        regions = {
            'center': (h//3, 2*h//3, w//3, 2*w//3),
            'edges': [(0, h//3, 0, w), (2*h//3, h, 0, w), 
                     (0, h, 0, w//3), (0, h, 2*w//3, w)]
        }
        
        # Check for radial flow (zoom)
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Calculate expected radial flow
        expected_x = x_coords - center_x
        expected_y = y_coords - center_y
        
        # Normalize
        norm = np.sqrt(expected_x**2 + expected_y**2)
        norm[norm == 0] = 1
        expected_x /= norm
        expected_y /= norm
        
        # Compare with actual flow
        flow_normalized = flow.copy()
        flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        flow_mag[flow_mag == 0] = 1
        flow_normalized[:,:,0] /= flow_mag
        flow_normalized[:,:,1] /= flow_mag
        
        radial_similarity = np.mean(
            expected_x * flow_normalized[:,:,0] + 
            expected_y * flow_normalized[:,:,1]
        )
        
        if radial_similarity > 0.5:
            return "radial_expansion"
        elif radial_similarity < -0.5:
            return "radial_contraction"
        
        # Check for rotational flow
        expected_rot_x = -expected_y
        expected_rot_y = expected_x
        
        rotation_similarity = np.mean(
            expected_rot_x * flow_normalized[:,:,0] + 
            expected_rot_y * flow_normalized[:,:,1]
        )
        
        if abs(rotation_similarity) > 0.5:
            return "rotational"
            
        # Check for uniform flow
        flow_std = np.std(flow, axis=(0, 1))
        if np.all(flow_std < 1.0):
            return "uniform"
            
        return "complex"
        
    def _calculate_motion_intensity(self, gray: np.ndarray) -> Dict[str, any]:
        """Calculate overall motion intensity."""
        # Frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        
        # Threshold to remove noise
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion metrics
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_ratio = motion_pixels / total_pixels
        
        # Classify intensity
        if motion_ratio < 0.01:
            intensity = "none"
        elif motion_ratio < 0.05:
            intensity = "minimal"
        elif motion_ratio < 0.15:
            intensity = "moderate"
        elif motion_ratio < 0.3:
            intensity = "high"
        else:
            intensity = "extreme"
            
        return {
            'intensity': intensity,
            'motion_ratio': float(motion_ratio),
            'motion_pixels': int(motion_pixels)
        }
        
    def _detect_camera_shake(self, gray: np.ndarray) -> Dict[str, any]:
        """Detect camera shake or handheld movement."""
        if len(self.motion_history) < 3:
            return {'has_shake': False, 'shake_level': 0}
            
        # Analyze motion vectors over recent frames
        recent_movements = [m['camera_movement'].get('motion_vector', {'x': 0, 'y': 0}) 
                          for m in list(self.motion_history)[-5:] 
                          if 'camera_movement' in m]
        
        if len(recent_movements) < 3:
            return {'has_shake': False, 'shake_level': 0}
            
        # Calculate variance in motion
        x_motions = [m['x'] for m in recent_movements]
        y_motions = [m['y'] for m in recent_movements]
        
        x_variance = np.var(x_motions)
        y_variance = np.var(y_motions)
        
        total_variance = x_variance + y_variance
        
        # Detect high-frequency changes
        direction_changes = 0
        for i in range(1, len(x_motions)):
            if np.sign(x_motions[i]) != np.sign(x_motions[i-1]):
                direction_changes += 1
            if np.sign(y_motions[i]) != np.sign(y_motions[i-1]):
                direction_changes += 1
                
        # Classify shake
        if total_variance > 10 and direction_changes > 3:
            has_shake = True
            if total_variance > 50:
                shake_level = "severe"
            elif total_variance > 20:
                shake_level = "moderate"
            else:
                shake_level = "mild"
        else:
            has_shake = False
            shake_level = "none"
            
        return {
            'has_shake': has_shake,
            'shake_level': shake_level,
            'variance': float(total_variance),
            'direction_changes': direction_changes,
            'style': 'handheld' if has_shake else 'stabilized'
        }
        
    def _detect_zoom(self, frame: np.ndarray) -> Dict[str, any]:
        """Detect zoom based on scale changes."""
        if self.prev_frame is None:
            return {'is_zooming': False, 'zoom_type': 'none'}
            
        # Detect features in both frames
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find matching features
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return {'is_zooming': False, 'zoom_type': 'none'}
            
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10:
            return {'is_zooming': False, 'zoom_type': 'none'}
            
        # Calculate scale change
        scale_changes = []
        for i in range(min(20, len(matches))):
            for j in range(i+1, min(20, len(matches))):
                # Distance in first frame
                p1_1 = kp1[matches[i].queryIdx].pt
                p2_1 = kp1[matches[j].queryIdx].pt
                dist1 = np.sqrt((p1_1[0] - p2_1[0])**2 + (p1_1[1] - p2_1[1])**2)
                
                # Distance in second frame
                p1_2 = kp2[matches[i].trainIdx].pt
                p2_2 = kp2[matches[j].trainIdx].pt
                dist2 = np.sqrt((p1_2[0] - p2_2[0])**2 + (p1_2[1] - p2_2[1])**2)
                
                if dist1 > 0:
                    scale_changes.append(dist2 / dist1)
                    
        if not scale_changes:
            return {'is_zooming': False, 'zoom_type': 'none'}
            
        avg_scale = np.median(scale_changes)
        
        # Determine zoom type
        if avg_scale > 1.02:
            zoom_type = 'zoom_in'
            is_zooming = True
        elif avg_scale < 0.98:
            zoom_type = 'zoom_out'
            is_zooming = True
        else:
            zoom_type = 'none'
            is_zooming = False
            
        return {
            'is_zooming': is_zooming,
            'zoom_type': zoom_type,
            'scale_factor': float(avg_scale),
            'confidence': float(min(abs(1 - avg_scale) * 10, 1.0))
        }
        
    def _detect_subject_motion(self, gray: np.ndarray) -> Dict[str, any]:
        """Detect motion of subjects vs camera motion."""
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Get camera motion from previous analysis
        if len(self.motion_history) > 0:
            camera_motion = self.motion_history[-1].get('camera_movement', {})
            camera_vector = camera_motion.get('motion_vector', {'x': 0, 'y': 0})
        else:
            camera_vector = {'x': 0, 'y': 0}
            
        # Subtract camera motion from flow
        compensated_flow = flow.copy()
        compensated_flow[:,:,0] -= camera_vector['x']
        compensated_flow[:,:,1] -= camera_vector['y']
        
        # Find regions with significant motion after compensation
        magnitude = np.sqrt(compensated_flow[:,:,0]**2 + compensated_flow[:,:,1]**2)
        motion_mask = magnitude > self.motion_threshold * 2
        
        # Find contours of moving regions
        motion_mask_uint8 = (motion_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(motion_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze moving subjects
        moving_subjects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate average motion in this region
                region_flow = compensated_flow[y:y+h, x:x+w]
                avg_motion = np.mean(region_flow, axis=(0, 1))
                
                moving_subjects.append({
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'area': float(area),
                    'motion_vector': {'x': float(avg_motion[0]), 'y': float(avg_motion[1])},
                    'speed': float(np.linalg.norm(avg_motion))
                })
                
        # Sort by area (largest first)
        moving_subjects.sort(key=lambda x: x['area'], reverse=True)
        
        return {
            'num_moving_subjects': len(moving_subjects),
            'moving_subjects': moving_subjects[:5],  # Top 5 largest
            'has_subject_motion': len(moving_subjects) > 0,
            'motion_coverage': float(np.sum(motion_mask) / motion_mask.size)
        }
        
    def _analyze_motion_pattern(self) -> Dict[str, any]:
        """Analyze motion patterns over recent frames."""
        if len(self.motion_history) < 5:
            return {'pattern': 'insufficient_data'}
            
        # Get recent camera movements
        recent_movements = [m['camera_movement']['type'] 
                          for m in list(self.motion_history)[-5:] 
                          if 'camera_movement' in m]
        
        # Check for consistent patterns
        if all(m == 'static' for m in recent_movements):
            pattern = 'static_shot'
        elif all(m == recent_movements[0] for m in recent_movements):
            pattern = f'continuous_{recent_movements[0]}'
        elif 'zoom_in' in recent_movements or 'zoom_out' in recent_movements:
            pattern = 'dynamic_zoom'
        else:
            # Check for specific sequences
            movement_sequence = '->'.join(recent_movements[-3:])
            
            if 'pan_left->pan_right' in movement_sequence or 'pan_right->pan_left' in movement_sequence:
                pattern = 'whip_pan'
            elif 'tilt_up->tilt_down' in movement_sequence or 'tilt_down->tilt_up' in movement_sequence:
                pattern = 'vertical_scan'
            else:
                pattern = 'complex_movement'
                
        return {
            'pattern': pattern,
            'recent_movements': recent_movements,
            'consistency': len(set(recent_movements)) == 1
        }
        
    def _get_static_result(self) -> Dict[str, any]:
        """Return result for static/first frame."""
        return {
            'camera_movement': {'type': 'static', 'confidence': 1.0},
            'optical_flow': {
                'average_magnitude': 0,
                'max_magnitude': 0,
                'motion_percentage': 0,
                'flow_pattern': 'none'
            },
            'motion_intensity': {
                'intensity': 'none',
                'motion_ratio': 0,
                'motion_pixels': 0
            },
            'shake_detection': {
                'has_shake': False,
                'shake_level': 'none'
            },
            'zoom_detection': {
                'is_zooming': False,
                'zoom_type': 'none'
            },
            'subject_motion': {
                'num_moving_subjects': 0,
                'has_subject_motion': False
            }
        }