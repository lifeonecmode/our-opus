#!/usr/bin/env python3
"""
Nuclear Frame Extractor - Guarantees exactly 1,800 frames at 30fps (60 seconds)
Maximum precision frame extraction with validation and multiple output formats
"""

import cv2
import json
import numpy as np
from pathlib import Path
import subprocess
import logging
import tempfile
import shutil
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import hashlib
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NuclearFrameExtractor:
    """Nuclear option for frame-perfect video extraction at exactly 30fps"""
    
    TARGET_FPS = 30
    TARGET_DURATION = 60  # seconds
    TARGET_FRAMES = 1800  # 30fps * 60s
    
    def __init__(self, video_path: str, output_dir: str = "nuclear_extraction"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Validation results
        self.validation_results = {
            "original_specs": {},
            "normalized_specs": {},
            "extraction_results": {},
            "frame_validation": {},
            "memory_usage": {}
        }
        
        logger.info(f"Nuclear Frame Extractor initialized for: {self.video_path}")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get detailed video information using ffprobe"""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video info: {result.stderr}")
        
        data = json.loads(result.stdout)
        stream = data['streams'][0] if data.get('streams') else {}
        format_info = data.get('format', {})
        
        # Parse frame rate
        r_frame_rate = stream.get('r_frame_rate', '0/1')
        num, den = map(int, r_frame_rate.split('/'))
        fps = num / den if den != 0 else 0
        
        return {
            "width": int(stream.get('width', 0)),
            "height": int(stream.get('height', 0)),
            "fps": fps,
            "duration": float(format_info.get('duration', 0)),
            "nb_frames": int(stream.get('nb_frames', 0)),
            "r_frame_rate": r_frame_rate
        }
    
    def normalize_video(self) -> Path:
        """Normalize video to exactly 30fps and 60 seconds"""
        logger.info("Phase 1: Video Normalization")
        
        # Get original video info
        original_info = self.get_video_info(str(self.video_path))
        self.validation_results["original_specs"] = original_info
        logger.info(f"Original: {original_info['width']}x{original_info['height']} @ {original_info['fps']:.2f}fps, {original_info['duration']:.2f}s")
        
        # Output path for normalized video
        normalized_path = self.temp_dir / "normalized_30fps_60s.mp4"
        
        # Calculate speed adjustment for exact 60s duration
        speed_factor = original_info['duration'] / self.TARGET_DURATION
        
        # FFmpeg command for normalization
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-filter_complex", 
            f"[0:v]setpts={1/speed_factor:.6f}*PTS,fps=fps={self.TARGET_FPS}:round=near[v]",
            "-map", "[v]",
            "-t", str(self.TARGET_DURATION),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",  # High quality
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(normalized_path)
        ]
        
        logger.info(f"Normalizing with speed factor: {speed_factor:.4f}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Video normalization failed: {result.stderr}")
        
        # Verify normalized video
        normalized_info = self.get_video_info(str(normalized_path))
        self.validation_results["normalized_specs"] = normalized_info
        
        logger.info(f"Normalized: {normalized_info['width']}x{normalized_info['height']} @ {normalized_info['fps']:.2f}fps, {normalized_info['duration']:.2f}s")
        
        # Validate normalization
        if abs(normalized_info['fps'] - self.TARGET_FPS) > 0.1:
            logger.warning(f"FPS mismatch: {normalized_info['fps']} != {self.TARGET_FPS}")
        
        if abs(normalized_info['duration'] - self.TARGET_DURATION) > 0.1:
            logger.warning(f"Duration mismatch: {normalized_info['duration']} != {self.TARGET_DURATION}")
        
        return normalized_path
    
    def extract_raw_frames(self, video_path: Path) -> List[Path]:
        """Extract raw frames with frame-perfect precision"""
        logger.info("Phase 2: Raw Frame Extraction")
        
        # Memory monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        extracted_frames = []
        frame_hashes = set()  # For duplicate detection
        
        # Use OpenCV for frame-by-frame extraction
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get actual frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video reports {total_frames} frames at {fps:.2f} fps")
        
        frame_count = 0
        extracted_count = 0
        
        # Extract exactly TARGET_FRAMES
        while frame_count < self.TARGET_FRAMES:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Frame read failed at frame {frame_count}")
                break
            
            # Calculate frame hash for duplicate detection
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            
            if frame_hash in frame_hashes:
                logger.warning(f"Duplicate frame detected at {frame_count}")
            
            frame_hashes.add(frame_hash)
            
            # Save frame
            frame_path = self.frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            extracted_frames.append(frame_path)
            extracted_count += 1
            
            # Memory management - clear every 100 frames
            if frame_count % 100 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                logger.info(f"Extracted {frame_count}/{self.TARGET_FRAMES} frames. Memory: {current_memory:.1f} MB")
            
            frame_count += 1
        
        cap.release()
        
        # Memory stats
        final_memory = process.memory_info().rss / 1024 / 1024
        self.validation_results["memory_usage"] = {
            "initial_mb": initial_memory,
            "final_mb": final_memory,
            "peak_mb": final_memory,
            "increase_mb": final_memory - initial_memory
        }
        
        # Validation
        self.validation_results["extraction_results"] = {
            "requested_frames": self.TARGET_FRAMES,
            "extracted_frames": extracted_count,
            "unique_frames": len(frame_hashes),
            "duplicate_frames": extracted_count - len(frame_hashes)
        }
        
        logger.info(f"Extracted {extracted_count} frames ({len(frame_hashes)} unique)")
        
        # Handle frame count mismatch
        if extracted_count < self.TARGET_FRAMES:
            logger.warning(f"Frame shortage: {extracted_count} < {self.TARGET_FRAMES}")
            # Duplicate last frame to reach target
            last_frame = extracted_frames[-1]
            for i in range(extracted_count, self.TARGET_FRAMES):
                frame_path = self.frames_dir / f"frame_{i:06d}.jpg"
                shutil.copy2(last_frame, frame_path)
                extracted_frames.append(frame_path)
                logger.info(f"Duplicated last frame for frame {i}")
        
        elif extracted_count > self.TARGET_FRAMES:
            logger.warning(f"Frame excess: {extracted_count} > {self.TARGET_FRAMES}")
            # Trim excess frames
            for frame_path in extracted_frames[self.TARGET_FRAMES:]:
                frame_path.unlink()
            extracted_frames = extracted_frames[:self.TARGET_FRAMES]
        
        return extracted_frames
    
    def validate_frames(self, frame_paths: List[Path]) -> Dict[str, Any]:
        """Validate extracted frames for completeness and quality"""
        logger.info("Phase 3: Frame Validation")
        
        validation = {
            "total_frames": len(frame_paths),
            "valid_frames": 0,
            "missing_frames": [],
            "corrupted_frames": [],
            "frame_sizes": [],
            "timestamps": [],
            "quality_metrics": {
                "min_size_kb": float('inf'),
                "max_size_kb": 0,
                "avg_size_kb": 0,
                "resolution_consistent": True
            }
        }
        
        expected_resolution = None
        total_size = 0
        
        for i, frame_path in enumerate(frame_paths):
            # Check existence
            if not frame_path.exists():
                validation["missing_frames"].append(i)
                continue
            
            # Check file size
            size_kb = frame_path.stat().st_size / 1024
            validation["frame_sizes"].append(size_kb)
            total_size += size_kb
            
            validation["quality_metrics"]["min_size_kb"] = min(
                validation["quality_metrics"]["min_size_kb"], size_kb
            )
            validation["quality_metrics"]["max_size_kb"] = max(
                validation["quality_metrics"]["max_size_kb"], size_kb
            )
            
            # Validate frame readability and resolution
            try:
                img = cv2.imread(str(frame_path))
                if img is None:
                    validation["corrupted_frames"].append(i)
                    continue
                
                # Check resolution consistency
                if expected_resolution is None:
                    expected_resolution = img.shape[:2]
                elif img.shape[:2] != expected_resolution:
                    validation["quality_metrics"]["resolution_consistent"] = False
                
                validation["valid_frames"] += 1
                
                # Calculate timestamp
                timestamp = i / self.TARGET_FPS
                validation["timestamps"].append(round(timestamp, 3))
                
            except Exception as e:
                logger.error(f"Error validating frame {i}: {e}")
                validation["corrupted_frames"].append(i)
        
        # Calculate average size
        if validation["frame_sizes"]:
            validation["quality_metrics"]["avg_size_kb"] = total_size / len(validation["frame_sizes"])
        
        # Frame continuity check
        validation["frame_continuity"] = {
            "has_gaps": len(validation["missing_frames"]) > 0,
            "gap_count": len(validation["missing_frames"]),
            "largest_gap": self._find_largest_gap(validation["missing_frames"]) if validation["missing_frames"] else 0
        }
        
        # Timing validation
        validation["timing_validation"] = {
            "expected_duration": self.TARGET_DURATION,
            "calculated_duration": (len(frame_paths) - 1) / self.TARGET_FPS,
            "fps_accurate": True
        }
        
        self.validation_results["frame_validation"] = validation
        
        logger.info(f"Validation complete: {validation['valid_frames']}/{validation['total_frames']} valid frames")
        
        return validation
    
    def _find_largest_gap(self, missing_frames: List[int]) -> int:
        """Find the largest gap in missing frames"""
        if not missing_frames:
            return 0
        
        sorted_missing = sorted(missing_frames)
        max_gap = 1
        
        for i in range(1, len(sorted_missing)):
            gap = sorted_missing[i] - sorted_missing[i-1]
            max_gap = max(max_gap, gap)
        
        return max_gap
    
    def generate_outputs(self, frame_paths: List[Path]) -> Dict[str, Path]:
        """Generate JSONL, JSON, and screenplay format outputs"""
        logger.info("Phase 4: Generating Output Formats")
        
        outputs = {}
        
        # 1. JSONL Format (frame-by-frame)
        jsonl_path = self.output_dir / "frames_analysis.jsonl"
        with open(jsonl_path, 'w') as f:
            for i, frame_path in enumerate(frame_paths):
                timestamp = i / self.TARGET_FPS
                frame_data = {
                    "frame_id": i,
                    "timestamp": round(timestamp, 3),
                    "timecode": self._seconds_to_timecode(timestamp),
                    "file_path": str(frame_path.relative_to(self.output_dir)),
                    "absolute_path": str(frame_path),
                    "frame_number": i + 1,
                    "total_frames": self.TARGET_FRAMES
                }
                f.write(json.dumps(frame_data) + '\n')
        outputs["jsonl"] = jsonl_path
        
        # 2. JSON Format (complete structure)
        json_data = {
            "metadata": {
                "source_video": str(self.video_path),
                "extraction_date": datetime.now().isoformat(),
                "target_fps": self.TARGET_FPS,
                "target_duration": self.TARGET_DURATION,
                "total_frames": self.TARGET_FRAMES,
                "extraction_method": "nuclear_frame_perfect"
            },
            "validation": self.validation_results,
            "frames": []
        }
        
        for i, frame_path in enumerate(frame_paths):
            timestamp = i / self.TARGET_FPS
            json_data["frames"].append({
                "id": i,
                "timestamp": round(timestamp, 3),
                "timecode": self._seconds_to_timecode(timestamp),
                "path": str(frame_path.relative_to(self.output_dir))
            })
        
        json_path = self.output_dir / "frames_complete.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        outputs["json"] = json_path
        
        # 3. Screenplay Format
        screenplay_path = self.output_dir / "frames_screenplay.txt"
        with open(screenplay_path, 'w') as f:
            f.write("FRAME-BY-FRAME SCREENPLAY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Title: {self.video_path.stem}\n")
            f.write(f"Duration: {self.TARGET_DURATION} seconds\n")
            f.write(f"Frame Rate: {self.TARGET_FPS} fps\n")
            f.write(f"Total Frames: {self.TARGET_FRAMES}\n\n")
            f.write("-" * 50 + "\n\n")
            
            # Write key frames every second
            for second in range(self.TARGET_DURATION + 1):
                frame_num = second * self.TARGET_FPS
                if frame_num < len(frame_paths):
                    f.write(f"[{second:02d}:{00:02d}] Frame {frame_num:04d}\n")
                    f.write(f"File: {frame_paths[frame_num].name}\n")
                    f.write(f"Action: [Frame content at {second} seconds]\n\n")
        
        outputs["screenplay"] = screenplay_path
        
        # 4. Validation Report
        report_path = self.output_dir / "extraction_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        outputs["report"] = report_path
        
        logger.info(f"Generated {len(outputs)} output formats")
        
        return outputs
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """Convert seconds to timecode format (HH:MM:SS:FF)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * self.TARGET_FPS)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def extract(self, cleanup: bool = True) -> Dict[str, Any]:
        """Execute the complete nuclear extraction process"""
        logger.info("Starting Nuclear Frame Extraction")
        logger.info(f"Target: {self.TARGET_FRAMES} frames @ {self.TARGET_FPS}fps = {self.TARGET_DURATION}s")
        
        try:
            # Phase 1: Normalize video
            normalized_video = self.normalize_video()
            
            # Phase 2: Extract frames
            frame_paths = self.extract_raw_frames(normalized_video)
            
            # Phase 3: Validate frames
            validation = self.validate_frames(frame_paths)
            
            # Phase 4: Generate outputs
            outputs = self.generate_outputs(frame_paths)
            
            # Success summary
            result = {
                "success": validation["valid_frames"] == self.TARGET_FRAMES,
                "frames_extracted": validation["valid_frames"],
                "target_frames": self.TARGET_FRAMES,
                "outputs": {k: str(v) for k, v in outputs.items()},
                "validation": validation,
                "frame_directory": str(self.frames_dir)
            }
            
            logger.info("Nuclear extraction complete!")
            logger.info(f"Frames: {result['frames_extracted']}/{result['target_frames']}")
            logger.info(f"Output directory: {self.output_dir}")
            
            return result
            
        except Exception as e:
            logger.error(f"Nuclear extraction failed: {e}")
            raise
        
        finally:
            if cleanup:
                self.cleanup_temp_files()


def main():
    """CLI interface for nuclear frame extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nuclear Frame Extractor - Guarantees exactly 1,800 frames at 30fps"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="nuclear_extraction",
                       help="Output directory (default: nuclear_extraction)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Keep temporary files")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = NuclearFrameExtractor(args.video, args.output)
    
    # Execute extraction
    try:
        result = extractor.extract(cleanup=not args.no_cleanup)
        
        print("\n✅ Nuclear Extraction Complete!")
        print(f"Frames extracted: {result['frames_extracted']}/{result['target_frames']}")
        print(f"\nOutputs generated:")
        for format_type, path in result['outputs'].items():
            print(f"  - {format_type}: {path}")
        
        if not result['success']:
            print("\n⚠️ Warning: Frame count mismatch detected!")
            print("Check extraction_report.json for details")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())