import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from ..settings import Config

logger = logging.getLogger(__name__)


class ShotTypeDetector:
    """Detects cinematographic shot types based on subject framing."""
    
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        
        # Shot type thresholds based on face/frame ratio
        self.shot_thresholds = {
            'extreme_close': 0.6,   # Face takes up >60% of frame
            'close': 0.3,           # Face takes up 30-60% of frame
            'medium': 0.1,          # Face takes up 10-30% of frame
            'wide': 0.02,           # Face takes up 2-10% of frame
            'extreme_wide': 0.0     # Face takes up <2% of frame
        }
        
    def detect_shot_type(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Detect the cinematographic shot type of a frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            Dict containing shot type and confidence
        """
        # Detect faces and bodies
        faces = self._detect_faces(frame)
        bodies = self._detect_bodies(frame)
        upper_bodies = self._detect_upper_bodies(frame)
        
        # Calculate shot type
        if len(faces) > 0:
            shot_info = self._analyze_by_face(frame, faces)
        elif len(upper_bodies) > 0:
            shot_info = self._analyze_by_upper_body(frame, upper_bodies)
        elif len(bodies) > 0:
            shot_info = self._analyze_by_body(frame, bodies)
        else:
            shot_info = self._analyze_by_scene(frame)
            
        # Add additional shot characteristics
        shot_info.update({
            'camera_angle': self._detect_camera_angle(frame, faces, bodies),
            'composition': self._analyze_composition(frame),
            'depth_of_field': self._estimate_depth_of_field(frame),
            'subject_position': self._get_subject_position(frame, faces, bodies)
        })
        
        return shot_info
        
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
        
    def _detect_bodies(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect full bodies in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100))
        return bodies
        
    def _detect_upper_bodies(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect upper bodies in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        upper_bodies = self.upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 80))
        return upper_bodies
        
    def _analyze_by_face(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> Dict[str, any]:
        """Analyze shot type based on face detection."""
        # Get the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Calculate face area ratio
        frame_area = frame.shape[0] * frame.shape[1]
        face_area = w * h
        face_ratio = face_area / frame_area
        
        # Determine shot type
        if face_ratio > self.shot_thresholds['extreme_close']:
            shot_type = "extreme_close"
            description = "Extreme Close-up (ECU): Face fills entire frame, showing intense emotion"
        elif face_ratio > self.shot_thresholds['close']:
            shot_type = "close"
            description = "Close-up (CU): Face and shoulders visible, intimate and personal"
        elif face_ratio > self.shot_thresholds['medium']:
            shot_type = "medium"
            description = "Medium Shot (MS): Waist up, good for dialogue and gestures"
        elif face_ratio > self.shot_thresholds['wide']:
            shot_type = "wide"
            description = "Wide Shot (WS): Full body visible with environment"
        else:
            shot_type = "extreme_wide"
            description = "Extreme Wide Shot (EWS): Subject small in frame, emphasizing location"
            
        return {
            'shot_type': shot_type,
            'description': description,
            'confidence': self._calculate_confidence(face_ratio, shot_type),
            'face_ratio': face_ratio,
            'num_faces': len(faces),
            'main_subject_bbox': {'x': x, 'y': y, 'width': w, 'height': h}
        }
        
    def _analyze_by_upper_body(self, frame: np.ndarray, upper_bodies: List[Tuple[int, int, int, int]]) -> Dict[str, any]:
        """Analyze shot type based on upper body detection."""
        largest_body = max(upper_bodies, key=lambda b: b[2] * b[3])
        x, y, w, h = largest_body
        
        frame_area = frame.shape[0] * frame.shape[1]
        body_area = w * h
        body_ratio = body_area / frame_area
        
        # Upper body detection usually indicates medium shots
        if body_ratio > 0.4:
            shot_type = "medium"
            description = "Medium Shot (MS): Upper body visible, good for character interaction"
        elif body_ratio > 0.15:
            shot_type = "medium"
            description = "Medium Wide Shot (MWS): Most of body visible"
        else:
            shot_type = "wide"
            description = "Wide Shot (WS): Full figure in environment"
            
        return {
            'shot_type': shot_type,
            'description': description,
            'confidence': 0.7,  # Lower confidence without face
            'body_ratio': body_ratio,
            'num_subjects': len(upper_bodies)
        }
        
    def _analyze_by_body(self, frame: np.ndarray, bodies: List[Tuple[int, int, int, int]]) -> Dict[str, any]:
        """Analyze shot type based on full body detection."""
        if len(bodies) == 0:
            return self._analyze_by_scene(frame)
            
        largest_body = max(bodies, key=lambda b: b[2] * b[3])
        x, y, w, h = largest_body
        
        frame_area = frame.shape[0] * frame.shape[1]
        body_area = w * h
        body_ratio = body_area / frame_area
        
        if body_ratio > 0.5:
            shot_type = "medium"
            description = "Medium Full Shot (MFS): Full body fills most of frame"
        elif body_ratio > 0.2:
            shot_type = "wide"
            description = "Wide Shot (WS): Full body with surrounding space"
        else:
            shot_type = "extreme_wide"
            description = "Extreme Wide Shot (EWS): Figure small in landscape"
            
        return {
            'shot_type': shot_type,
            'description': description,
            'confidence': 0.6,
            'body_ratio': body_ratio,
            'num_subjects': len(bodies)
        }
        
    def _analyze_by_scene(self, frame: np.ndarray) -> Dict[str, any]:
        """Analyze shot type when no people are detected."""
        # Use edge detection and scene analysis
        edges = cv2.Canny(frame, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Analyze color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # Check for landscape characteristics
        horizon_line = self._detect_horizon(frame)
        
        if horizon_line is not None:
            shot_type = "extreme_wide"
            description = "Landscape/Establishing Shot: Wide view of location"
            confidence = 0.8
        elif edge_density > 0.1:
            shot_type = "close"
            description = "Detail/Insert Shot: Close view of object or detail"
            confidence = 0.6
        else:
            shot_type = "wide"
            description = "Environmental Shot: General view of space"
            confidence = 0.5
            
        return {
            'shot_type': shot_type,
            'description': description,
            'confidence': confidence,
            'scene_type': 'no_people',
            'edge_density': edge_density
        }
        
    def _detect_camera_angle(self, frame: np.ndarray, faces: List, bodies: List) -> Dict[str, any]:
        """Detect camera angle based on subject position."""
        height, width = frame.shape[:2]
        
        # Default angle
        angle = "eye_level"
        confidence = 0.5
        
        if len(faces) > 0:
            # Analyze face position
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_center_y = largest_face[1] + largest_face[3] / 2
            
            if face_center_y < height * 0.3:
                angle = "high_angle"
                confidence = 0.8
            elif face_center_y > height * 0.7:
                angle = "low_angle"
                confidence = 0.8
            else:
                angle = "eye_level"
                confidence = 0.9
                
        # Check for Dutch angle
        if self._detect_dutch_angle(frame):
            angle = "dutch_angle"
            confidence = 0.7
            
        return {
            'angle': angle,
            'confidence': confidence,
            'description': self._get_angle_description(angle)
        }
        
    def _analyze_composition(self, frame: np.ndarray) -> Dict[str, any]:
        """Analyze frame composition."""
        height, width = frame.shape[:2]
        
        # Rule of thirds analysis
        thirds_vertical = [width // 3, 2 * width // 3]
        thirds_horizontal = [height // 3, 2 * height // 3]
        
        # Find strong edges
        edges = cv2.Canny(frame, 50, 150)
        
        # Check alignment with rule of thirds
        vertical_alignment = 0
        horizontal_alignment = 0
        
        for x in thirds_vertical:
            column = edges[:, max(0, x-5):min(width, x+5)]
            if np.sum(column) > 0:
                vertical_alignment += 1
                
        for y in thirds_horizontal:
            row = edges[max(0, y-5):min(height, y+5), :]
            if np.sum(row) > 0:
                horizontal_alignment += 1
                
        # Symmetry detection
        left_half = frame[:, :width//2]
        right_half = cv2.flip(frame[:, width//2:], 1)
        symmetry_score = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        
        # Leading lines detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        has_leading_lines = lines is not None and len(lines) > 3
        
        return {
            'rule_of_thirds': {
                'vertical_alignment': vertical_alignment,
                'horizontal_alignment': horizontal_alignment,
                'follows_rule': (vertical_alignment + horizontal_alignment) >= 2
            },
            'symmetry': {
                'score': symmetry_score,
                'is_symmetric': symmetry_score > 0.7
            },
            'leading_lines': has_leading_lines,
            'balance': self._analyze_visual_balance(frame)
        }
        
    def _estimate_depth_of_field(self, frame: np.ndarray) -> Dict[str, any]:
        """Estimate depth of field characteristics."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate focus metrics using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_measure = laplacian.var()
        
        # Divide frame into regions
        h, w = gray.shape
        regions = {
            'center': gray[h//3:2*h//3, w//3:2*w//3],
            'edges': np.concatenate([
                gray[:h//3, :].flatten(),
                gray[2*h//3:, :].flatten(),
                gray[:, :w//3].flatten(),
                gray[:, 2*w//3:].flatten()
            ])
        }
        
        # Calculate sharpness for each region
        center_sharp = cv2.Laplacian(regions['center'], cv2.CV_64F).var()
        edge_sharp = cv2.Laplacian(regions['edges'].reshape(-1, 1), cv2.CV_64F).var()
        
        # Determine DOF characteristics
        if center_sharp > edge_sharp * 2:
            dof_type = "shallow"
            description = "Shallow DOF: Subject in focus, background blurred"
        elif edge_sharp > center_sharp * 1.5:
            dof_type = "reverse"
            description = "Reverse focus: Background sharp, foreground soft"
        elif focus_measure > 100:
            dof_type = "deep"
            description = "Deep DOF: Everything in focus"
        else:
            dof_type = "soft"
            description = "Soft focus: Overall dreamy/soft look"
            
        return {
            'type': dof_type,
            'description': description,
            'focus_measure': focus_measure,
            'center_sharpness': center_sharp,
            'edge_sharpness': edge_sharp
        }
        
    def _get_subject_position(self, frame: np.ndarray, faces: List, bodies: List) -> Dict[str, any]:
        """Get subject position in frame."""
        height, width = frame.shape[:2]
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            center_x = x + w / 2
            center_y = y + h / 2
        elif len(bodies) > 0:
            largest_body = max(bodies, key=lambda b: b[2] * b[3])
            x, y, w, h = largest_body
            center_x = x + w / 2
            center_y = y + h / 2
        else:
            return {'position': 'no_subject', 'quadrant': 'none'}
            
        # Determine position
        if center_x < width / 3:
            h_pos = "left"
        elif center_x > 2 * width / 3:
            h_pos = "right"
        else:
            h_pos = "center"
            
        if center_y < height / 3:
            v_pos = "top"
        elif center_y > 2 * height / 3:
            v_pos = "bottom"
        else:
            v_pos = "middle"
            
        # Quadrant
        if center_x < width / 2 and center_y < height / 2:
            quadrant = "top_left"
        elif center_x >= width / 2 and center_y < height / 2:
            quadrant = "top_right"
        elif center_x < width / 2 and center_y >= height / 2:
            quadrant = "bottom_left"
        else:
            quadrant = "bottom_right"
            
        return {
            'position': f"{v_pos}_{h_pos}",
            'quadrant': quadrant,
            'center': {'x': center_x / width, 'y': center_y / height},
            'offset_from_center': {
                'x': (center_x - width/2) / width,
                'y': (center_y - height/2) / height
            }
        }
        
    def _calculate_confidence(self, ratio: float, shot_type: str) -> float:
        """Calculate confidence score for shot type detection."""
        # Define ideal ratios for each shot type
        ideal_ratios = {
            'extreme_close': 0.8,
            'close': 0.45,
            'medium': 0.2,
            'wide': 0.06,
            'extreme_wide': 0.01
        }
        
        ideal = ideal_ratios.get(shot_type, 0.1)
        difference = abs(ratio - ideal)
        
        # Convert difference to confidence (0-1)
        confidence = max(0, 1 - difference * 2)
        
        return confidence
        
    def _detect_horizon(self, frame: np.ndarray) -> Optional[int]:
        """Detect horizon line in landscape shots."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Use Hough transform to find horizontal lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            horizontal_lines = []
            for rho, theta in lines[:, 0]:
                # Check if line is roughly horizontal
                if abs(theta - np.pi/2) < 0.1:
                    horizontal_lines.append(rho)
                    
            if horizontal_lines:
                # Return average position of horizontal lines
                return int(np.mean(horizontal_lines))
                
        return None
        
    def _detect_dutch_angle(self, frame: np.ndarray) -> bool:
        """Detect if frame has Dutch angle (tilted camera)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Calculate average angle of lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
                
            avg_angle = np.mean(angles)
            
            # Dutch angle if average line angle is tilted
            return 5 < abs(avg_angle) < 85
            
        return False
        
    def _get_angle_description(self, angle: str) -> str:
        """Get description for camera angle."""
        descriptions = {
            'high_angle': "High angle: Camera looking down, subject appears vulnerable",
            'low_angle': "Low angle: Camera looking up, subject appears powerful",
            'eye_level': "Eye level: Neutral perspective, natural view",
            'dutch_angle': "Dutch angle: Tilted frame creating unease or dynamism"
        }
        
        return descriptions.get(angle, "Unknown angle")
        
    def _analyze_visual_balance(self, frame: np.ndarray) -> str:
        """Analyze visual balance of the frame."""
        height, width = frame.shape[:2]
        
        # Convert to grayscale for weight calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate visual weight for each half
        left_weight = np.mean(gray[:, :width//2])
        right_weight = np.mean(gray[:, width//2:])
        
        balance_ratio = min(left_weight, right_weight) / max(left_weight, right_weight)
        
        if balance_ratio > 0.9:
            return "balanced"
        elif left_weight > right_weight:
            return "left_heavy"
        else:
            return "right_heavy"