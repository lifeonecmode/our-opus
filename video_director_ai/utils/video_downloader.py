import os
import yt_dlp
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from ..settings import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class VideoDownloader:
    """Handles video downloading from various platforms using yt-dlp."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Config.OUTPUT_DIR / "downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self, url: str, quality: str = "best") -> Dict[str, Any]:
        """
        Download video from URL.
        
        Args:
            url: Video URL (YouTube, TikTok, Instagram, etc.)
            quality: Video quality (best, 1080p, 720p, etc.)
            
        Returns:
            Dict with download info including local file path
        """
        output_template = str(self.output_dir / "%(title)s_%(id)s.%(ext)s")
        
        ydl_opts = {
            'format': f'bestvideo[height<={quality}]+bestaudio/best' if quality.isdigit() else quality,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            # Add cookies support for Instagram
            'cookiesfrombrowser': ('chrome',),
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'logger': logger,
            'progress_hooks': [self._progress_hook],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video from: {url}")
                info = ydl.extract_info(url, download=True)
                
                # Get the downloaded file path
                filename = ydl.prepare_filename(info)
                # Handle post-processed filename
                if not os.path.exists(filename):
                    filename = filename.rsplit('.', 1)[0] + '.mp4'
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'description': info.get('description', ''),
                    'file_path': filename,
                    'thumbnail': info.get('thumbnail', ''),
                    'platform': info.get('extractor_key', 'Unknown'),
                    'original_url': url
                }
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
            
    def _progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress callback for download tracking."""
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            logger.info(f"Downloading: {percent} at {speed}")
        elif d['status'] == 'finished':
            logger.info("Download completed, processing...")
            
    def download_audio_only(self, url: str) -> Dict[str, Any]:
        """Download only audio track for transcription."""
        output_template = str(self.output_dir / "%(title)s_%(id)s.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'logger': logger,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading audio from: {url}")
                info = ydl.extract_info(url, download=True)
                
                filename = ydl.prepare_filename(info)
                audio_filename = filename.rsplit('.', 1)[0] + '.mp3'
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'file_path': audio_filename,
                    'original_url': url
                }
                
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise