import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from ..settings import Config

logger = logging.getLogger(__name__)


class ColorGradingAnalyzer:
    """Analyzes color grading, LUTs, and film stock emulation in video frames."""
    
    def __init__(self):
        self.color_signatures = Config.COLOR_GRADING_SIGNATURES
        self.film_stocks = Config.FILM_STOCKS
        
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive color analysis of a single frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            Dict containing color grading analysis
        """
        analysis = {
            'color_grading': self.detect_color_grading(frame),
            'temperature': self.detect_color_temperature(frame),
            'lut_detection': self.detect_lut_type(frame),
            'film_stock': self.detect_film_stock_emulation(frame),
            'histogram_stats': self.analyze_histogram(frame),
            'color_palette': self.extract_dominant_colors(frame),
            'contrast_ratio': self.calculate_contrast_ratio(frame),
            'saturation_level': self.calculate_saturation(frame)
        }
        
        return analysis
        
    def detect_color_grading(self, frame: np.ndarray) -> Dict[str, any]:
        """Detect common color grading styles."""
        b, g, r = cv2.split(frame)
        
        # Calculate histograms
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist_b = hist_b.flatten() / hist_b.sum()
        hist_g = hist_g.flatten() / hist_g.sum()
        hist_r = hist_r.flatten() / hist_r.sum()
        
        # Detect specific color grading patterns
        grading_style = "Standard"
        confidence = 0.0
        
        # Teal & Orange detection
        if self._is_teal_orange(hist_b, hist_r, frame):
            grading_style = "Teal_Orange"
            confidence = self._calculate_teal_orange_confidence(frame)
            
        # Bleach Bypass detection
        elif self._is_bleach_bypass(hist_b, hist_g, hist_r):
            grading_style = "Bleach_Bypass"
            confidence = self._calculate_bleach_bypass_confidence(frame)
            
        # Day for Night detection
        elif self._is_day_for_night(frame):
            grading_style = "Day_for_Night"
            confidence = self._calculate_day_for_night_confidence(frame)
            
        # Matrix Green detection
        elif self._is_matrix_green(frame):
            grading_style = "Matrix_Green"
            confidence = self._calculate_matrix_confidence(frame)
            
        # Mexico Filter detection
        elif self._is_mexico_filter(frame):
            grading_style = "Mexico_Filter"
            confidence = self._calculate_mexico_confidence(frame)
            
        return {
            'style': grading_style,
            'confidence': confidence,
            'description': self.color_signatures.get(grading_style, {}).get('description', 'Standard color grading')
        }
        
    def detect_color_temperature(self, frame: np.ndarray) -> Dict[str, any]:
        """Detect color temperature and tint."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate average values
        avg_a = np.mean(a) - 128  # Green-Magenta axis
        avg_b = np.mean(b) - 128  # Blue-Yellow axis
        
        # Determine temperature
        if avg_b > 10:
            temp = "Warm"
            kelvin = "3200K-4000K"
        elif avg_b < -10:
            temp = "Cool"
            kelvin = "6000K-8000K"
        else:
            temp = "Neutral"
            kelvin = "5000K-5500K"
            
        # Detect specific looks
        look = "Standard"
        if avg_a > 5 and avg_b > 10:
            look = "Golden Hour/Magic Hour"
        elif avg_a < -5 and avg_b < -10:
            look = "Moonlight/Night"
        elif abs(avg_a) < 2 and abs(avg_b) < 2:
            look = "Neutral/Documentary"
            
        return {
            'temperature': temp,
            'kelvin': kelvin,
            'tint': {'green_magenta': avg_a, 'blue_yellow': avg_b},
            'look': look
        }
        
    def detect_lut_type(self, frame: np.ndarray) -> Dict[str, any]:
        """Detect potential LUT usage based on color curves."""
        # Extract color curves
        curves = self._extract_color_curves(frame)
        
        # Common LUT signatures
        lut_signatures = {
            "LOG_to_Rec709": {"lift": 0.1, "gamma": 0.8, "gain": 0.9},
            "Fuji_3510": {"lift": 0.05, "gamma": 0.85, "gain": 0.95},
            "Kodak_2393": {"lift": 0.0, "gamma": 0.9, "gain": 1.0},
            "Cinematic": {"lift": -0.05, "gamma": 0.85, "gain": 0.9},
            "Instagram_Nashville": {"lift": 0.1, "gamma": 0.9, "gain": 1.1}
        }
        
        best_match = "None"
        best_score = 0.0
        
        for lut_name, signature in lut_signatures.items():
            score = self._compare_curves_to_signature(curves, signature)
            if score > best_score:
                best_match = lut_name
                best_score = score
                
        return {
            'detected_lut': best_match,
            'confidence': best_score,
            'curves': curves
        }
        
    def detect_film_stock_emulation(self, frame: np.ndarray) -> Dict[str, any]:
        """Detect film stock emulation characteristics."""
        # Analyze grain
        grain_level = self._analyze_grain(frame)
        
        # Get color characteristics
        saturation = self.calculate_saturation(frame)
        contrast = self.calculate_contrast_ratio(frame)
        
        # Check for halation
        has_halation = self._detect_halation(frame)
        
        # Match to film stocks
        best_match = "Digital"
        best_score = 0.0
        
        for stock_name, characteristics in Config.FILM_STOCKS.items():
            score = 0.0
            
            # Compare grain
            if abs(grain_level - characteristics['grain']) < 0.1:
                score += 0.3
                
            # Compare saturation
            if abs(saturation - characteristics['saturation']) < 0.1:
                score += 0.3
                
            # Compare contrast
            if abs(contrast - characteristics['contrast']) < 0.1:
                score += 0.3
                
            # Bonus for halation match
            if has_halation and characteristics.get('halation', False):
                score += 0.1
                
            if score > best_score:
                best_match = stock_name
                best_score = score
                
        return {
            'film_stock': best_match,
            'confidence': best_score,
            'grain_level': grain_level,
            'has_halation': has_halation
        }
        
    def analyze_histogram(self, frame: np.ndarray) -> Dict[str, any]:
        """Analyze histogram statistics."""
        # Convert to grayscale for overall luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Calculate statistics
        total_pixels = hist.sum()
        mean = np.sum(hist * np.arange(256)) / total_pixels
        
        # Calculate percentiles
        cumsum = np.cumsum(hist)
        percentiles = {}
        for p in [1, 5, 25, 50, 75, 95, 99]:
            idx = np.searchsorted(cumsum, total_pixels * p / 100)
            percentiles[f'p{p}'] = min(idx, 255)
            
        # Detect crushed blacks/blown highlights
        black_point = percentiles['p1']
        white_point = percentiles['p99']
        
        crushed_blacks = black_point > 10
        blown_highlights = white_point < 245
        
        return {
            'mean': mean,
            'percentiles': percentiles,
            'crushed_blacks': crushed_blacks,
            'blown_highlights': blown_highlights,
            'black_point': black_point,
            'white_point': white_point
        }
        
    def extract_dominant_colors(self, frame: np.ndarray, k: int = 5) -> List[Dict[str, any]]:
        """Extract dominant colors from frame."""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (100, 100))
        data = small_frame.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels
        unique, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        
        dominant_colors = []
        for idx in sorted_indices:
            color = centers[idx]
            percentage = counts[idx] / len(labels) * 100
            
            dominant_colors.append({
                'bgr': color.tolist(),
                'rgb': color[::-1].tolist(),
                'percentage': percentage,
                'hex': '#{:02x}{:02x}{:02x}'.format(int(color[2]), int(color[1]), int(color[0]))
            })
            
        return dominant_colors
        
    def calculate_contrast_ratio(self, frame: np.ndarray) -> float:
        """Calculate overall contrast ratio."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get 95th and 5th percentile values
        flat = gray.flatten()
        p95 = np.percentile(flat, 95)
        p5 = np.percentile(flat, 5)
        
        # Avoid division by zero
        if p5 == 0:
            p5 = 1
            
        contrast_ratio = p95 / p5
        return min(contrast_ratio, 10.0)  # Cap at 10 for normalization
        
    def calculate_saturation(self, frame: np.ndarray) -> float:
        """Calculate average saturation level."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        return np.mean(saturation) / 255.0
        
    # Helper methods
    def _is_teal_orange(self, hist_b: np.ndarray, hist_r: np.ndarray, frame: np.ndarray) -> bool:
        """Check if frame has teal & orange color grading."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        teal_lower = np.array([85, 100, 100])
        teal_upper = np.array([100, 255, 255])
        
        # Create masks
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        teal_mask = cv2.inRange(hsv, teal_lower, teal_upper)
        
        # Calculate percentages
        orange_percent = np.sum(orange_mask > 0) / orange_mask.size
        teal_percent = np.sum(teal_mask > 0) / teal_mask.size
        
        return orange_percent > 0.05 and teal_percent > 0.05
        
    def _calculate_teal_orange_confidence(self, frame: np.ndarray) -> float:
        """Calculate confidence score for teal & orange grading."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Similar to above but return confidence score
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        teal_lower = np.array([85, 100, 100])
        teal_upper = np.array([100, 255, 255])
        
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        teal_mask = cv2.inRange(hsv, teal_lower, teal_upper)
        
        orange_percent = np.sum(orange_mask > 0) / orange_mask.size
        teal_percent = np.sum(teal_mask > 0) / teal_mask.size
        
        return min((orange_percent + teal_percent) * 2, 1.0)
        
    def _is_bleach_bypass(self, hist_b: np.ndarray, hist_g: np.ndarray, hist_r: np.ndarray) -> bool:
        """Check for bleach bypass characteristics."""
        # Low saturation, high contrast
        saturation = np.std(np.array([hist_b.mean(), hist_g.mean(), hist_r.mean()]))
        return saturation < 0.1
        
    def _calculate_bleach_bypass_confidence(self, frame: np.ndarray) -> float:
        """Calculate confidence for bleach bypass."""
        saturation = self.calculate_saturation(frame)
        contrast = self.calculate_contrast_ratio(frame)
        
        if saturation < 0.4 and contrast > 1.5:
            return min((2 - saturation) * contrast / 3, 1.0)
        return 0.0
        
    def _is_day_for_night(self, frame: np.ndarray) -> bool:
        """Check for day-for-night color grading."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        avg_l = np.mean(l)
        avg_b = np.mean(b) - 128
        
        return avg_l < 100 and avg_b < -10
        
    def _calculate_day_for_night_confidence(self, frame: np.ndarray) -> float:
        """Calculate confidence for day-for-night."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        avg_l = np.mean(l)
        avg_b = np.mean(b) - 128
        
        if avg_l < 100 and avg_b < -10:
            return min((100 - avg_l) / 50 + abs(avg_b) / 20, 1.0)
        return 0.0
        
    def _is_matrix_green(self, frame: np.ndarray) -> bool:
        """Check for Matrix-style green tint."""
        b, g, r = cv2.split(frame)
        
        green_dominance = np.mean(g) / (np.mean(b) + np.mean(r) + 1)
        return green_dominance > 0.6
        
    def _calculate_matrix_confidence(self, frame: np.ndarray) -> float:
        """Calculate confidence for Matrix green."""
        b, g, r = cv2.split(frame)
        
        green_dominance = np.mean(g) / (np.mean(b) + np.mean(r) + 1)
        if green_dominance > 0.6:
            return min((green_dominance - 0.6) * 2.5, 1.0)
        return 0.0
        
    def _is_mexico_filter(self, frame: np.ndarray) -> bool:
        """Check for Mexico/desert yellow-orange filter."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow-orange range
        lower = np.array([20, 100, 100])
        upper = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        percent = np.sum(mask > 0) / mask.size
        
        return percent > 0.3
        
    def _calculate_mexico_confidence(self, frame: np.ndarray) -> float:
        """Calculate confidence for Mexico filter."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([20, 100, 100])
        upper = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        percent = np.sum(mask > 0) / mask.size
        
        if percent > 0.3:
            return min(percent * 2, 1.0)
        return 0.0
        
    def _extract_color_curves(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract RGB curves from frame."""
        b, g, r = cv2.split(frame)
        
        curves = {}
        for channel, name in [(b, 'blue'), (g, 'green'), (r, 'red')]:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Create cumulative distribution
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            
            curves[name] = cdf_normalized
            
        return curves
        
    def _compare_curves_to_signature(self, curves: Dict[str, np.ndarray], signature: Dict[str, float]) -> float:
        """Compare extracted curves to known LUT signature."""
        # Simplified comparison - in reality would be more complex
        score = 0.0
        
        for color in ['red', 'green', 'blue']:
            if color in curves:
                curve = curves[color]
                
                # Check lift (shadows)
                lift_actual = curve[32] - curve[0]
                lift_expected = signature.get('lift', 0.0)
                score += max(0, 1 - abs(lift_actual - lift_expected))
                
                # Check gamma (midtones)
                gamma_actual = curve[128] - curve[64]
                gamma_expected = signature.get('gamma', 0.5)
                score += max(0, 1 - abs(gamma_actual - gamma_expected))
                
                # Check gain (highlights)
                gain_actual = curve[255] - curve[192]
                gain_expected = signature.get('gain', 0.9)
                score += max(0, 1 - abs(gain_actual - gain_expected))
                
        return score / 9.0  # Normalize to 0-1
        
    def _analyze_grain(self, frame: np.ndarray) -> float:
        """Analyze film grain level."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate difference (grain)
        grain = cv2.absdiff(gray, blurred)
        
        # Calculate grain metric
        grain_level = np.std(grain) / 255.0
        
        return min(grain_level * 10, 1.0)  # Normalize to 0-1
        
    def _detect_halation(self, frame: np.ndarray) -> bool:
        """Detect film halation effect around bright areas."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find bright areas
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Dilate to check surrounding areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(bright_mask, kernel)
        
        # Check for color bleeding in bright areas
        if np.sum(bright_mask) > 0:
            bright_regions = cv2.bitwise_and(frame, frame, mask=dilated)
            
            # Check for red/orange tint around bright areas (typical halation)
            b, g, r = cv2.split(bright_regions)
            
            # Halation typically shows as red/orange glow
            red_dominance = np.mean(r[dilated > 0]) > np.mean(g[dilated > 0]) * 1.2
            
            return red_dominance
            
        return False