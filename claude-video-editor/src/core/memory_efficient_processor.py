#!/usr/bin/env python3
"""
Memory-Efficient Video Processor
Handles large videos with minimal memory footprint using chunked processing
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
import psutil
import gc
import tempfile
import subprocess
from typing import Dict, List, Any, Generator, Optional, Tuple
from contextlib import contextmanager
import mmap
from multiprocessing import Process, Queue, cpu_count
import queue
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryEfficientProcessor:
    """Process large videos with minimal memory usage"""
    
    def __init__(self, max_memory_mb: int = 2048, chunk_size: int = 100):
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size
        self.process = psutil.Process()
        
        # Memory monitoring
        self.memory_stats = {
            "initial_mb": self.get_memory_usage(),
            "peak_mb": 0,
            "average_mb": 0,
            "measurements": []
        }
        
        # Processing stats
        self.processing_stats = {
            "frames_processed": 0,
            "chunks_processed": 0,
            "gc_collections": 0
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        current_mb = self.get_memory_usage()
        self.memory_stats["measurements"].append(current_mb)
        self.memory_stats["peak_mb"] = max(self.memory_stats["peak_mb"], current_mb)
        
        if current_mb > self.max_memory_mb:
            logger.warning(f"Memory usage ({current_mb:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
            return False
        return True
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        gc.collect()  # Second collection for circular references
        self.processing_stats["gc_collections"] += 1
        logger.debug(f"Forced cleanup. Memory: {self.get_memory_usage():.1f} MB")
    
    @contextmanager
    def memory_mapped_video(self, video_path: Path) -> Generator:
        """Memory-mapped video file access"""
        with open(video_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                yield mmapped_file
    
    def process_video_chunks(self, video_path: Path, output_dir: Path, 
                           target_fps: float = 30.0) -> Dict[str, Any]:
        """Process video in memory-efficient chunks"""
        logger.info(f"Starting chunked processing of {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {total_frames} frames @ {fps} fps")
        
        results = {
            "chunks": [],
            "total_frames": total_frames,
            "processed_frames": 0,
            "failed_frames": 0
        }
        
        try:
            # Process in chunks
            chunk_id = 0
            while True:
                chunk_start = chunk_id * self.chunk_size
                if chunk_start >= total_frames:
                    break
                
                logger.info(f"Processing chunk {chunk_id} (frames {chunk_start}-{chunk_start + self.chunk_size})")
                
                # Process chunk
                chunk_result = self._process_chunk(
                    cap, chunk_id, chunk_start, 
                    min(self.chunk_size, total_frames - chunk_start),
                    output_dir
                )
                
                results["chunks"].append(chunk_result)
                results["processed_frames"] += chunk_result["processed"]
                results["failed_frames"] += chunk_result["failed"]
                
                # Memory management
                if not self.check_memory():
                    self.force_cleanup()
                
                # Progress update
                progress = results["processed_frames"] / total_frames
                logger.info(f"Progress: {progress:.1%} ({results['processed_frames']}/{total_frames} frames)")
                
                chunk_id += 1
                self.processing_stats["chunks_processed"] = chunk_id
        
        finally:
            cap.release()
            self.force_cleanup()
        
        # Calculate final stats
        if self.memory_stats["measurements"]:
            self.memory_stats["average_mb"] = np.mean(self.memory_stats["measurements"])
        
        results["memory_stats"] = self.memory_stats
        results["processing_stats"] = self.processing_stats
        
        return results
    
    def _process_chunk(self, cap: cv2.VideoCapture, chunk_id: int, 
                      start_frame: int, chunk_size: int, output_dir: Path) -> Dict[str, Any]:
        """Process a single chunk of frames"""
        chunk_result = {
            "chunk_id": chunk_id,
            "start_frame": start_frame,
            "end_frame": start_frame + chunk_size,
            "processed": 0,
            "failed": 0,
            "output_files": []
        }
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames in chunk
        for i in range(chunk_size):
            frame_num = start_frame + i
            ret, frame = cap.read()
            
            if not ret:
                chunk_result["failed"] += 1
                logger.warning(f"Failed to read frame {frame_num}")
                continue
            
            # Save frame
            output_path = output_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            chunk_result["processed"] += 1
            chunk_result["output_files"].append(str(output_path))
            
            # Release frame memory immediately
            del frame
            
            # Periodic memory check
            if i % 10 == 0 and not self.check_memory():
                self.force_cleanup()
        
        self.processing_stats["frames_processed"] += chunk_result["processed"]
        
        return chunk_result
    
    def parallel_process_video(self, video_path: Path, output_dir: Path,
                             num_workers: Optional[int] = None) -> Dict[str, Any]:
        """Process video using parallel workers for maximum efficiency"""
        if num_workers is None:
            num_workers = min(cpu_count(), 4)  # Limit to 4 workers
        
        logger.info(f"Starting parallel processing with {num_workers} workers")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create work queue
        work_queue = Queue()
        results_queue = Queue()
        
        # Divide work among workers
        frames_per_worker = total_frames // num_workers
        for i in range(num_workers):
            start_frame = i * frames_per_worker
            end_frame = (i + 1) * frames_per_worker if i < num_workers - 1 else total_frames
            
            work_queue.put({
                "worker_id": i,
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps
            })
        
        # Start workers
        workers = []
        for i in range(num_workers):
            worker = Process(target=self._worker_process, args=(work_queue, results_queue))
            worker.start()
            workers.append(worker)
        
        # Collect results
        results = {
            "worker_results": [],
            "total_processed": 0,
            "total_failed": 0
        }
        
        for _ in range(num_workers):
            try:
                worker_result = results_queue.get(timeout=300)  # 5 minute timeout
                results["worker_results"].append(worker_result)
                results["total_processed"] += worker_result["processed"]
                results["total_failed"] += worker_result["failed"]
            except queue.Empty:
                logger.error("Worker timeout")
        
        # Wait for workers to finish
        for worker in workers:
            worker.join()
        
        return results
    
    @staticmethod
    def _worker_process(work_queue: Queue, results_queue: Queue):
        """Worker process for parallel video processing"""
        try:
            work = work_queue.get()
            
            cap = cv2.VideoCapture(work["video_path"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, work["start_frame"])
            
            output_dir = Path(work["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = {
                "worker_id": work["worker_id"],
                "processed": 0,
                "failed": 0
            }
            
            for frame_num in range(work["start_frame"], work["end_frame"]):
                ret, frame = cap.read()
                
                if not ret:
                    result["failed"] += 1
                    continue
                
                output_path = output_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result["processed"] += 1
                
                # Memory management
                del frame
                if frame_num % 50 == 0:
                    gc.collect()
            
            cap.release()
            results_queue.put(result)
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
            results_queue.put({"worker_id": work.get("worker_id", -1), "error": str(e)})
    
    def stream_process_video(self, video_path: Path, output_dir: Path,
                           buffer_size: int = 10) -> Generator[Dict[str, Any], None, None]:
        """Stream process video with minimal memory using generator pattern"""
        logger.info(f"Starting stream processing of {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            frame_buffer = []
            frame_num = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output_path = output_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Add to buffer
                frame_info = {
                    "frame_num": frame_num,
                    "timestamp": frame_num / fps,
                    "path": str(output_path),
                    "size_kb": output_path.stat().st_size / 1024
                }
                frame_buffer.append(frame_info)
                
                # Yield buffer when full
                if len(frame_buffer) >= buffer_size:
                    yield {
                        "frames": frame_buffer,
                        "progress": frame_num / total_frames,
                        "memory_mb": self.get_memory_usage()
                    }
                    frame_buffer = []
                
                # Memory management
                del frame
                frame_num += 1
                
                if frame_num % 100 == 0:
                    self.force_cleanup()
            
            # Yield remaining frames
            if frame_buffer:
                yield {
                    "frames": frame_buffer,
                    "progress": 1.0,
                    "memory_mb": self.get_memory_usage()
                }
        
        finally:
            cap.release()
            self.force_cleanup()
    
    def extract_with_ffmpeg_pipe(self, video_path: Path, output_dir: Path,
                                target_fps: float = 30.0) -> Dict[str, Any]:
        """Extract frames using FFmpeg pipe for minimal memory usage"""
        logger.info("Using FFmpeg pipe for memory-efficient extraction")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command to output raw frames
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={target_fps}",
            "-f", "image2pipe",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        
        # Get video dimensions
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video_path)
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        probe_data = json.loads(probe_result.stdout)
        width = probe_data['streams'][0]['width']
        height = probe_data['streams'][0]['height']
        
        # Start FFmpeg process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frame_size = width * height * 3  # BGR24
        frame_num = 0
        processed = 0
        failed = 0
        
        try:
            while True:
                # Read one frame
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                
                # Save frame
                output_path = output_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                processed += 1
                frame_num += 1
                
                # Memory management
                del frame
                del raw_frame
                
                if frame_num % 100 == 0:
                    self.force_cleanup()
                    logger.info(f"Processed {frame_num} frames. Memory: {self.get_memory_usage():.1f} MB")
        
        except Exception as e:
            logger.error(f"Error during FFmpeg extraction: {e}")
            failed = frame_num - processed
        
        finally:
            process.terminate()
            process.wait()
        
        return {
            "method": "ffmpeg_pipe",
            "processed_frames": processed,
            "failed_frames": failed,
            "memory_stats": self.memory_stats
        }


def main():
    """CLI interface for memory-efficient processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-Efficient Video Processor")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("-o", "--output", default="efficient_frames",
                       help="Output directory")
    parser.add_argument("--max-memory", type=int, default=2048,
                       help="Maximum memory usage in MB (default: 2048)")
    parser.add_argument("--chunk-size", type=int, default=100,
                       help="Frames per chunk (default: 100)")
    parser.add_argument("--method", choices=["chunks", "parallel", "stream", "ffmpeg"],
                       default="chunks", help="Processing method")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    processor = MemoryEfficientProcessor(
        max_memory_mb=args.max_memory,
        chunk_size=args.chunk_size
    )
    
    video_path = Path(args.video)
    output_dir = Path(args.output)
    
    print(f"Processing {video_path} with {args.method} method")
    print(f"Max memory: {args.max_memory} MB")
    
    try:
        if args.method == "chunks":
            result = processor.process_video_chunks(video_path, output_dir)
        elif args.method == "parallel":
            result = processor.parallel_process_video(video_path, output_dir, args.workers)
        elif args.method == "stream":
            # Stream processing example
            for batch in processor.stream_process_video(video_path, output_dir):
                print(f"Progress: {batch['progress']:.1%}, Memory: {batch['memory_mb']:.1f} MB")
            result = {"method": "stream", "completed": True}
        elif args.method == "ffmpeg":
            result = processor.extract_with_ffmpeg_pipe(video_path, output_dir)
        
        print("\n=== Processing Complete ===")
        if "processed_frames" in result:
            print(f"Processed: {result['processed_frames']} frames")
        if "memory_stats" in result:
            stats = result["memory_stats"]
            print(f"Memory usage:")
            print(f"  - Initial: {stats['initial_mb']:.1f} MB")
            print(f"  - Peak: {stats['peak_mb']:.1f} MB")
            print(f"  - Average: {stats['average_mb']:.1f} MB")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())