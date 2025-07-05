#!/usr/bin/env python3
"""
Input Processor - Automatic video processing for input directory
Watches input directory and automatically processes new videos
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import argparse
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from analyze_all_footage import ProjectAnalyzer
from claude_decision_engine import ClaudeDecisionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputVideoHandler(FileSystemEventHandler):
    """Handles new video files in the input directory"""
    
    def __init__(self, processor):
        self.processor = processor
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        
    def on_created(self, event):
        """Called when a new file is created in the watched directory"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.video_extensions:
                logger.info(f"📹 New video detected: {file_path.name}")
                # Add to processing queue
                asyncio.create_task(self.processor.process_new_video(file_path))
                
    def on_moved(self, event):
        """Called when a file is moved into the watched directory"""
        if not event.is_directory:
            file_path = Path(event.dest_path)
            if file_path.suffix.lower() in self.video_extensions:
                logger.info(f"📹 Video moved in: {file_path.name}")
                # Add to processing queue
                asyncio.create_task(self.processor.process_new_video(file_path))


class InputProcessor:
    """Processes videos in the input directory"""
    
    def __init__(self, project_dir: str = ".", auto_edit: bool = False, style: str = "viral"):
        self.project_dir = Path(project_dir)
        self.input_dir = self.project_dir / "input"
        self.processed_dir = self.project_dir / "processed"
        self.analysis_dir = self.project_dir / "analysis"
        self.auto_edit = auto_edit
        self.style = style
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Track processed files
        self.processed_log_file = self.analysis_dir / "processed_videos.json"
        self.processed_videos = self._load_processed_log()
        
    def _load_processed_log(self) -> Dict[str, Any]:
        """Load the log of processed videos"""
        if self.processed_log_file.exists():
            with open(self.processed_log_file, 'r') as f:
                return json.load(f)
        return {"videos": {}, "last_updated": None}
    
    def _save_processed_log(self):
        """Save the processed videos log"""
        self.processed_videos["last_updated"] = datetime.now().isoformat()
        with open(self.processed_log_file, 'w') as f:
            json.dump(self.processed_videos, f, indent=2)
    
    async def process_new_video(self, video_path: Path):
        """Process a newly added video"""
        try:
            # Check if already processed
            if str(video_path) in self.processed_videos.get("videos", {}):
                logger.info(f"⏭️ Video already processed: {video_path.name}")
                return
            
            logger.info(f"🔄 Processing video: {video_path.name}")
            
            # Wait a moment to ensure file is fully written
            await asyncio.sleep(2)
            
            # Analyze the video
            analyzer = ProjectAnalyzer(self.project_dir)
            analysis = await analyzer.analyze_input_directory()
            
            # Find this video in the analysis
            video_info = None
            for video in analysis.get("input_videos", []):
                if video["filename"] == video_path.name:
                    video_info = video
                    break
            
            if video_info:
                # Log processing
                self.processed_videos["videos"][str(video_path)] = {
                    "filename": video_path.name,
                    "processed_at": datetime.now().isoformat(),
                    "category": video_info["category"],
                    "duration": video_info["analysis"].get("metadata", {}).get("duration", 0),
                    "suggested_use": video_info["suggested_use"]["primary_use"]
                }
                self._save_processed_log()
                
                logger.info(f"✅ Video analyzed: {video_path.name}")
                logger.info(f"   Category: {video_info['category']}")
                logger.info(f"   Suggested use: {video_info['suggested_use']['primary_use']}")
                
                # Auto-edit if enabled
                if self.auto_edit:
                    await self._create_auto_edit(video_info)
                    
                # Move to processed directory
                processed_path = self.processed_dir / f"{video_info['category']}_{video_path.name}"
                video_path.rename(processed_path)
                logger.info(f"📁 Moved to processed: {processed_path.name}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {video_path.name}: {e}")
    
    async def _create_auto_edit(self, video_info: Dict[str, Any]):
        """Create an automatic edit for the video"""
        logger.info(f"🎬 Creating auto-edit for {video_info['filename']}")
        
        # Create a minimal project structure for this video
        temp_project = self.project_dir / "temp_edits" / video_info['filename'].replace('.', '_')
        temp_project.mkdir(parents=True, exist_ok=True)
        
        # Link or copy the video to appropriate location based on category
        if video_info['category'] == 'speaker':
            footage_dir = temp_project / "footage"
            footage_dir.mkdir(exist_ok=True)
            target = footage_dir / "main-speaker.mp4"
        else:
            footage_dir = temp_project / "footage" / "b-roll" / video_info['category']
            footage_dir.mkdir(parents=True, exist_ok=True)
            target = footage_dir / video_info['filename']
        
        # Create symlink to original video
        if not target.exists():
            target.symlink_to(Path(video_info['path']).absolute())
        
        # Use Claude to create edit
        engine = ClaudeDecisionEngine(str(temp_project))
        
        # Quick edit based on video type
        if video_info['category'] == 'speaker':
            # Create a short viral edit
            decisions = {
                "style": "viral",
                "duration": min(30, video_info['analysis'].get('metadata', {}).get('duration', 30)),
                "structure": [
                    {
                        "section": "highlight",
                        "start": 0,
                        "duration": min(30, video_info['analysis'].get('metadata', {}).get('duration', 30)),
                        "clips": [
                            {
                                "source": str(target),
                                "trim": {"start": 0, "duration": min(30, video_info['analysis'].get('metadata', {}).get('duration', 30))}
                            }
                        ],
                        "text": {
                            "content": "AUTO-EDIT PREVIEW",
                            "style": {"fontSize": 48, "color": "#FFD700"},
                            "timing": {"start": 0, "duration": 3}
                        }
                    }
                ],
                "audio_tracks": []
            }
        else:
            # Create a montage for other types
            decisions = {
                "style": "montage",
                "duration": 15,
                "structure": [
                    {
                        "section": "showcase",
                        "start": 0,
                        "duration": 15,
                        "clips": [
                            {
                                "source": str(target),
                                "trim": {"start": 0, "duration": 15}
                            }
                        ],
                        "effects": [
                            {"type": "fade", "params": {"direction": "in", "duration": 0.5}},
                            {"type": "fade", "params": {"direction": "out", "duration": 0.5, "start": 14.5}}
                        ]
                    }
                ],
                "audio_tracks": []
            }
        
        # Save decisions
        decisions_path = temp_project / f"auto_edit_decisions.json"
        with open(decisions_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        
        # Execute edit
        output_name = f"auto_edit_{video_info['category']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = self.project_dir / "outputs" / output_name
        output_path.parent.mkdir(exist_ok=True)
        
        success = await engine.execute_edit(decisions, str(output_path))
        
        if success:
            logger.info(f"✅ Auto-edit created: {output_path.name}")
        else:
            logger.error(f"❌ Auto-edit failed for {video_info['filename']}")
    
    async def process_existing_videos(self):
        """Process all existing videos in the input directory"""
        logger.info("📥 Processing existing videos in input directory...")
        
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(ext))
        
        if not video_files:
            logger.info("No videos found in input directory")
            return
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_file in video_files:
            await self.process_new_video(video_file)
    
    def watch_directory(self):
        """Start watching the input directory for new videos"""
        logger.info(f"👀 Watching directory: {self.input_dir}")
        logger.info("Press Ctrl+C to stop watching")
        
        event_handler = InputVideoHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.input_dir), recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("\n👋 Stopped watching directory")
        observer.join()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process videos in input directory")
    parser.add_argument("project", nargs="?", default=".",
                       help="Project directory path")
    parser.add_argument("--watch", action="store_true",
                       help="Watch directory for new videos")
    parser.add_argument("--auto-edit", action="store_true",
                       help="Automatically create edits for processed videos")
    parser.add_argument("--style", default="viral",
                       choices=["viral", "cinematic", "standard"],
                       help="Edit style for auto-edits")
    
    args = parser.parse_args()
    
    processor = InputProcessor(args.project, args.auto_edit, args.style)
    
    # Process existing videos first
    await processor.process_existing_videos()
    
    # Start watching if requested
    if args.watch:
        # Run watch in a separate thread while keeping async loop running
        import threading
        watch_thread = threading.Thread(target=processor.watch_directory)
        watch_thread.start()
        
        # Keep async loop running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Input processor stopped")