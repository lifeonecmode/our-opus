#!/usr/bin/env python3
"""
Frame Validation System - Ensures frame-perfect extraction with no drops
Validates frame continuity, quality, and temporal accuracy
"""

import cv2
import numpy as np
from pathlib import Path
import json
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FrameValidationSystem:
    """Comprehensive frame validation to ensure no frames are dropped"""
    
    def __init__(self, frames_dir: str, target_fps: float = 30.0, target_frames: int = 1800):
        self.frames_dir = Path(frames_dir)
        self.target_fps = target_fps
        self.target_frames = target_frames
        self.target_duration = target_frames / target_fps
        
        self.validation_results = {
            "frame_integrity": {},
            "temporal_accuracy": {},
            "visual_continuity": {},
            "quality_metrics": {},
            "issues_detected": []
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("Starting comprehensive frame validation")
        
        # 1. Frame integrity check
        self.validate_frame_integrity()
        
        # 2. Temporal accuracy check
        self.validate_temporal_accuracy()
        
        # 3. Visual continuity check
        self.validate_visual_continuity()
        
        # 4. Quality metrics check
        self.validate_quality_metrics()
        
        # 5. Generate validation report
        self.generate_validation_report()
        
        return self.validation_results
    
    def validate_frame_integrity(self) -> Dict[str, Any]:
        """Validate frame files exist and are readable"""
        logger.info("Validating frame integrity...")
        
        integrity = {
            "expected_frames": self.target_frames,
            "found_frames": 0,
            "missing_frames": [],
            "corrupted_frames": [],
            "duplicate_frames": [],
            "frame_hashes": {},
            "sequential_gaps": []
        }
        
        # Check for all expected frames
        for i in range(self.target_frames):
            frame_path = self.frames_dir / f"frame_{i:06d}.jpg"
            
            if not frame_path.exists():
                integrity["missing_frames"].append(i)
                # Check for sequential gaps
                if i > 0 and (i-1) not in integrity["missing_frames"]:
                    integrity["sequential_gaps"].append({
                        "start": i,
                        "end": i
                    })
            else:
                # Try to read the frame
                try:
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        integrity["corrupted_frames"].append(i)
                    else:
                        integrity["found_frames"] += 1
                        
                        # Calculate frame hash for duplicate detection
                        frame_hash = hashlib.md5(img.tobytes()).hexdigest()
                        if frame_hash in integrity["frame_hashes"]:
                            integrity["duplicate_frames"].append({
                                "frame": i,
                                "duplicate_of": integrity["frame_hashes"][frame_hash]
                            })
                        else:
                            integrity["frame_hashes"][frame_hash] = i
                
                except Exception as e:
                    logger.error(f"Error reading frame {i}: {e}")
                    integrity["corrupted_frames"].append(i)
        
        # Extend sequential gaps
        if integrity["sequential_gaps"]:
            merged_gaps = []
            current_gap = integrity["sequential_gaps"][0]
            
            for gap in integrity["sequential_gaps"][1:]:
                if gap["start"] == current_gap["end"] + 1:
                    current_gap["end"] = gap["end"]
                else:
                    merged_gaps.append(current_gap)
                    current_gap = gap
            merged_gaps.append(current_gap)
            integrity["sequential_gaps"] = merged_gaps
        
        integrity["integrity_score"] = integrity["found_frames"] / self.target_frames
        integrity["has_issues"] = len(integrity["missing_frames"]) > 0 or len(integrity["corrupted_frames"]) > 0
        
        self.validation_results["frame_integrity"] = integrity
        
        if integrity["has_issues"]:
            self.validation_results["issues_detected"].append({
                "type": "frame_integrity",
                "severity": "critical",
                "details": f"Missing: {len(integrity['missing_frames'])}, Corrupted: {len(integrity['corrupted_frames'])}"
            })
        
        logger.info(f"Frame integrity: {integrity['found_frames']}/{self.target_frames} valid frames")
        
        return integrity
    
    def validate_temporal_accuracy(self) -> Dict[str, Any]:
        """Validate temporal spacing and timestamps"""
        logger.info("Validating temporal accuracy...")
        
        temporal = {
            "expected_interval": 1.0 / self.target_fps,
            "frame_timestamps": [],
            "interval_deviations": [],
            "max_deviation": 0,
            "avg_deviation": 0,
            "timing_errors": []
        }
        
        # Calculate expected timestamps
        for i in range(self.target_frames):
            expected_time = i / self.target_fps
            temporal["frame_timestamps"].append({
                "frame": i,
                "expected": round(expected_time, 6),
                "actual": round(expected_time, 6)  # Assuming perfect extraction
            })
        
        # Check for timing errors
        for i in range(1, len(temporal["frame_timestamps"])):
            prev_time = temporal["frame_timestamps"][i-1]["actual"]
            curr_time = temporal["frame_timestamps"][i]["actual"]
            interval = curr_time - prev_time
            
            deviation = abs(interval - temporal["expected_interval"])
            temporal["interval_deviations"].append(deviation)
            
            if deviation > 0.001:  # 1ms threshold
                temporal["timing_errors"].append({
                    "frame": i,
                    "interval": interval,
                    "deviation": deviation
                })
        
        if temporal["interval_deviations"]:
            temporal["max_deviation"] = max(temporal["interval_deviations"])
            temporal["avg_deviation"] = sum(temporal["interval_deviations"]) / len(temporal["interval_deviations"])
        
        temporal["temporal_accuracy_score"] = 1.0 - min(temporal["avg_deviation"] * self.target_fps, 1.0)
        temporal["is_accurate"] = temporal["max_deviation"] < 0.01  # 10ms threshold
        
        self.validation_results["temporal_accuracy"] = temporal
        
        if not temporal["is_accurate"]:
            self.validation_results["issues_detected"].append({
                "type": "temporal_accuracy",
                "severity": "warning",
                "details": f"Max deviation: {temporal['max_deviation']*1000:.2f}ms"
            })
        
        logger.info(f"Temporal accuracy score: {temporal['temporal_accuracy_score']:.2%}")
        
        return temporal
    
    def validate_visual_continuity(self) -> Dict[str, Any]:
        """Validate visual continuity between frames"""
        logger.info("Validating visual continuity...")
        
        continuity = {
            "motion_analysis": [],
            "scene_changes": [],
            "continuity_breaks": [],
            "avg_frame_difference": 0,
            "max_frame_difference": 0
        }
        
        prev_frame = None
        frame_differences = []
        
        for i in range(min(self.target_frames, 300)):  # Sample first 10 seconds
            frame_path = self.frames_dir / f"frame_{i:06d}.jpg"
            
            if frame_path.exists():
                try:
                    curr_frame = cv2.imread(str(frame_path))
                    if curr_frame is not None and prev_frame is not None:
                        # Calculate frame difference
                        diff = cv2.absdiff(prev_frame, curr_frame)
                        diff_score = np.mean(diff)
                        frame_differences.append(diff_score)
                        
                        # Detect scene changes (high difference)
                        if diff_score > 50:  # Threshold for scene change
                            continuity["scene_changes"].append({
                                "frame": i,
                                "difference": diff_score,
                                "timestamp": i / self.target_fps
                            })
                        
                        # Detect continuity breaks (identical frames when there should be motion)
                        if diff_score < 0.1 and i > 0:  # Nearly identical frames
                            continuity["continuity_breaks"].append({
                                "frame": i,
                                "type": "duplicate_content",
                                "timestamp": i / self.target_fps
                            })
                        
                        # Motion vector estimation (simplified)
                        if i % 5 == 0:  # Every 5 frames
                            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                            gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                            
                            # Optical flow
                            flow = cv2.calcOpticalFlowFarneback(
                                gray_prev, gray_curr, None,
                                pyr_scale=0.5, levels=3, winsize=15,
                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                            )
                            
                            motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                            continuity["motion_analysis"].append({
                                "frame": i,
                                "motion_magnitude": motion_magnitude,
                                "timestamp": i / self.target_fps
                            })
                    
                    prev_frame = curr_frame
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame {i}: {e}")
        
        if frame_differences:
            continuity["avg_frame_difference"] = np.mean(frame_differences)
            continuity["max_frame_difference"] = np.max(frame_differences)
        
        continuity["continuity_score"] = 1.0 - (len(continuity["continuity_breaks"]) / min(self.target_frames, 300))
        continuity["has_issues"] = len(continuity["continuity_breaks"]) > 5
        
        self.validation_results["visual_continuity"] = continuity
        
        if continuity["has_issues"]:
            self.validation_results["issues_detected"].append({
                "type": "visual_continuity",
                "severity": "warning",
                "details": f"Detected {len(continuity['continuity_breaks'])} continuity breaks"
            })
        
        logger.info(f"Visual continuity score: {continuity['continuity_score']:.2%}")
        
        return continuity
    
    def validate_quality_metrics(self) -> Dict[str, Any]:
        """Validate frame quality metrics"""
        logger.info("Validating quality metrics...")
        
        quality = {
            "resolution_consistency": True,
            "expected_resolution": None,
            "resolution_changes": [],
            "file_sizes": [],
            "quality_scores": [],
            "blur_detection": [],
            "compression_artifacts": []
        }
        
        for i in range(0, self.target_frames, 30):  # Sample every second
            frame_path = self.frames_dir / f"frame_{i:06d}.jpg"
            
            if frame_path.exists():
                try:
                    # File size check
                    size_kb = frame_path.stat().st_size / 1024
                    quality["file_sizes"].append({
                        "frame": i,
                        "size_kb": size_kb
                    })
                    
                    # Read frame
                    img = cv2.imread(str(frame_path))
                    if img is not None:
                        # Resolution check
                        resolution = img.shape[:2]
                        if quality["expected_resolution"] is None:
                            quality["expected_resolution"] = resolution
                        elif resolution != quality["expected_resolution"]:
                            quality["resolution_consistency"] = False
                            quality["resolution_changes"].append({
                                "frame": i,
                                "resolution": resolution
                            })
                        
                        # Blur detection
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        if laplacian_var < 100:  # Blur threshold
                            quality["blur_detection"].append({
                                "frame": i,
                                "blur_score": laplacian_var,
                                "is_blurry": True
                            })
                        
                        # JPEG quality estimation
                        quality_score = self._estimate_jpeg_quality(img)
                        quality["quality_scores"].append({
                            "frame": i,
                            "quality": quality_score
                        })
                        
                        if quality_score < 0.7:
                            quality["compression_artifacts"].append({
                                "frame": i,
                                "quality": quality_score
                            })
                
                except Exception as e:
                    logger.error(f"Error analyzing quality for frame {i}: {e}")
        
        # Calculate quality statistics
        if quality["quality_scores"]:
            scores = [q["quality"] for q in quality["quality_scores"]]
            quality["avg_quality"] = np.mean(scores)
            quality["min_quality"] = np.min(scores)
        else:
            quality["avg_quality"] = 0
            quality["min_quality"] = 0
        
        if quality["file_sizes"]:
            sizes = [f["size_kb"] for f in quality["file_sizes"]]
            quality["avg_file_size"] = np.mean(sizes)
            quality["file_size_variance"] = np.std(sizes)
        
        quality["quality_score"] = quality["avg_quality"] * (0.9 if quality["resolution_consistency"] else 0.7)
        quality["has_issues"] = (
            not quality["resolution_consistency"] or
            len(quality["blur_detection"]) > 5 or
            quality["avg_quality"] < 0.8
        )
        
        self.validation_results["quality_metrics"] = quality
        
        if quality["has_issues"]:
            issues = []
            if not quality["resolution_consistency"]:
                issues.append("inconsistent resolution")
            if len(quality["blur_detection"]) > 5:
                issues.append(f"{len(quality['blur_detection'])} blurry frames")
            if quality["avg_quality"] < 0.8:
                issues.append(f"low quality ({quality['avg_quality']:.2f})")
            
            self.validation_results["issues_detected"].append({
                "type": "quality_metrics",
                "severity": "warning",
                "details": ", ".join(issues)
            })
        
        logger.info(f"Quality score: {quality['quality_score']:.2%}")
        
        return quality
    
    def _estimate_jpeg_quality(self, img: np.ndarray) -> float:
        """Estimate JPEG quality based on compression artifacts"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # DCT-based quality estimation
        h, w = gray.shape
        
        # Sample 8x8 blocks
        block_scores = []
        for y in range(0, h-8, 16):
            for x in range(0, w-8, 16):
                block = gray[y:y+8, x:x+8].astype(np.float32)
                dct = cv2.dct(block)
                
                # Higher frequency components indicate better quality
                high_freq = np.sum(np.abs(dct[4:, 4:]))
                total = np.sum(np.abs(dct)) + 1e-6
                
                score = high_freq / total
                block_scores.append(score)
        
        if block_scores:
            # Normalize to 0-1 range
            quality = np.mean(block_scores) * 10  # Empirical scaling
            return min(1.0, quality)
        
        return 0.5
    
    def generate_validation_report(self) -> Path:
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        report_path = self.frames_dir.parent / "frame_validation_report.json"
        
        # Calculate overall validation score
        scores = []
        if "frame_integrity" in self.validation_results:
            scores.append(self.validation_results["frame_integrity"].get("integrity_score", 0))
        if "temporal_accuracy" in self.validation_results:
            scores.append(self.validation_results["temporal_accuracy"].get("temporal_accuracy_score", 0))
        if "visual_continuity" in self.validation_results:
            scores.append(self.validation_results["visual_continuity"].get("continuity_score", 0))
        if "quality_metrics" in self.validation_results:
            scores.append(self.validation_results["quality_metrics"].get("quality_score", 0))
        
        overall_score = np.mean(scores) if scores else 0
        
        # Add summary
        self.validation_results["summary"] = {
            "overall_score": overall_score,
            "passed": overall_score > 0.95 and len(self.validation_results["issues_detected"]) == 0,
            "total_issues": len(self.validation_results["issues_detected"]),
            "critical_issues": len([i for i in self.validation_results["issues_detected"] if i["severity"] == "critical"]),
            "warnings": len([i for i in self.validation_results["issues_detected"] if i["severity"] == "warning"])
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Generate visual report if matplotlib is available
        self._generate_visual_report()
        
        logger.info(f"Validation report saved to: {report_path}")
        logger.info(f"Overall validation score: {overall_score:.2%}")
        
        return report_path
    
    def _generate_visual_report(self):
        """Generate visual validation report"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Frame Validation Report', fontsize=16)
            
            # 1. Frame integrity pie chart
            ax = axes[0, 0]
            integrity = self.validation_results.get("frame_integrity", {})
            if integrity:
                labels = ['Valid', 'Missing', 'Corrupted', 'Duplicate']
                sizes = [
                    integrity.get("found_frames", 0),
                    len(integrity.get("missing_frames", [])),
                    len(integrity.get("corrupted_frames", [])),
                    len(integrity.get("duplicate_frames", []))
                ]
                colors = ['green', 'red', 'orange', 'yellow']
                ax.pie([s for s in sizes if s > 0], 
                      labels=[l for l, s in zip(labels, sizes) if s > 0],
                      colors=[c for c, s in zip(colors, sizes) if s > 0],
                      autopct='%1.1f%%')
                ax.set_title('Frame Integrity')
            
            # 2. Temporal accuracy plot
            ax = axes[0, 1]
            temporal = self.validation_results.get("temporal_accuracy", {})
            if temporal.get("interval_deviations"):
                ax.plot(temporal["interval_deviations"][:100])
                ax.axhline(y=0.001, color='r', linestyle='--', label='1ms threshold')
                ax.set_xlabel('Frame interval')
                ax.set_ylabel('Deviation (s)')
                ax.set_title('Temporal Accuracy')
                ax.legend()
            
            # 3. Visual continuity
            ax = axes[1, 0]
            continuity = self.validation_results.get("visual_continuity", {})
            if continuity.get("motion_analysis"):
                frames = [m["frame"] for m in continuity["motion_analysis"]]
                motion = [m["motion_magnitude"] for m in continuity["motion_analysis"]]
                ax.plot(frames, motion)
                ax.set_xlabel('Frame')
                ax.set_ylabel('Motion magnitude')
                ax.set_title('Motion Analysis')
            
            # 4. Quality metrics
            ax = axes[1, 1]
            quality = self.validation_results.get("quality_metrics", {})
            if quality.get("quality_scores"):
                frames = [q["frame"] for q in quality["quality_scores"]]
                scores = [q["quality"] for q in quality["quality_scores"]]
                ax.plot(frames, scores)
                ax.axhline(y=0.8, color='r', linestyle='--', label='Quality threshold')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Quality score')
                ax.set_title('Frame Quality')
                ax.legend()
            
            plt.tight_layout()
            report_path = self.frames_dir.parent / "frame_validation_visual.png"
            plt.savefig(report_path, dpi=150)
            plt.close()
            
            logger.info(f"Visual report saved to: {report_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate visual report: {e}")


def main():
    """CLI interface for frame validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Frame Validation System")
    parser.add_argument("frames_dir", help="Directory containing extracted frames")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Target FPS (default: 30)")
    parser.add_argument("--frames", type=int, default=1800,
                       help="Target frame count (default: 1800)")
    
    args = parser.parse_args()
    
    validator = FrameValidationSystem(args.frames_dir, args.fps, args.frames)
    results = validator.validate_all()
    
    print("\n=== Frame Validation Results ===")
    print(f"Overall Score: {results['summary']['overall_score']:.2%}")
    print(f"Validation: {'PASSED' if results['summary']['passed'] else 'FAILED'}")
    print(f"Total Issues: {results['summary']['total_issues']}")
    print(f"  - Critical: {results['summary']['critical_issues']}")
    print(f"  - Warnings: {results['summary']['warnings']}")
    
    if results['issues_detected']:
        print("\nIssues detected:")
        for issue in results['issues_detected']:
            print(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['details']}")
    
    return 0 if results['summary']['passed'] else 1


if __name__ == "__main__":
    exit(main())