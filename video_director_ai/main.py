#!/usr/bin/env python3
"""
Video Director AI - Professional cinematographic analysis tool
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from director_brain import DirectorBrain
from settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for Video Director AI."""
    parser = argparse.ArgumentParser(
        description='Analyze videos with professional cinematographic insights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a YouTube video
  python main.py https://www.youtube.com/watch?v=VIDEO_ID
  
  # Analyze a local video file
  python main.py /path/to/video.mp4
  
  # Analyze and output only JSON
  python main.py video.mp4 --format json
  
  # Analyze with custom frame interval
  python main.py video.mp4 --interval 2
        """
    )
    
    parser.add_argument(
        'video',
        help='Video URL (YouTube, TikTok, Instagram) or local file path'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'markdown', 'csv', 'video', 'all'],
        default='all',
        help='Output format (default: all)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=1.0,
        help='Frame analysis interval in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--whisper-model', '-w',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        default=Config.WHISPER_MODEL,
        help=f'Whisper model size (default: {Config.WHISPER_MODEL})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Config.OUTPUT_DIR,
        help=f'Output directory (default: {Config.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--no-clips',
        action='store_true',
        help='Skip extracting video clips'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Update configuration
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if args.interval:
        Config.FRAME_INTERVAL = args.interval
        
    if args.whisper_model:
        Config.WHISPER_MODEL = args.whisper_model
        
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    # Initialize Director Brain
    logger.info("Initializing Video Director AI...")
    director = DirectorBrain()
    
    try:
        # Run analysis
        logger.info(f"Starting analysis of: {args.video}")
        logger.info(f"Frame interval: {Config.FRAME_INTERVAL}s")
        logger.info(f"Output format: {args.format}")
        
        results = await director.analyze_video(
            args.video,
            output_format=args.format
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        # Display key findings
        synthesis = results.get('synthesis', {})
        metadata = results.get('metadata', {})
        
        print(f"\nVideo: {metadata.get('filename', 'Unknown')}")
        print(f"Duration: {metadata.get('duration', 0):.2f} seconds")
        print(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
        
        print(f"\n{synthesis.get('overall_assessment', 'No assessment available')}")
        
        # Show top key moments
        key_moments = synthesis.get('key_moments', [])[:3]
        if key_moments:
            print("\nTop Key Moments:")
            for i, moment in enumerate(key_moments, 1):
                print(f"{i}. {moment['timestamp']:.1f}s - {moment['type']}")
                print(f"   {moment.get('description', '')[:80]}...")
                
        # Show output location
        print(f"\nFull analysis saved to: {Config.ANALYSIS_DIR}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


def run():
    """Run the main async function."""
    asyncio.run(main())


if __name__ == "__main__":
    run()