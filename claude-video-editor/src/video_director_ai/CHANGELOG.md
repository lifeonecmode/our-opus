# Changelog

All notable changes to the Video Director AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-16

### Added
- Initial MVP release of Video Director AI
- Video ingestion pipeline with yt-dlp support for YouTube, TikTok, and Instagram
- Frame extraction at configurable intervals (default: 1 second)
- Emotion analysis using FER (Facial Expression Recognition)
- Shot type detection based on face/body ratios
- Camera movement analysis using optical flow
- Color grading and LUT detection
- Scene detection using PySceneDetect
- Audio transcription integration with Whisper
- Unified analysis output with director-style descriptions
- Async processing for improved performance
- Configurable settings with dynamic paths
- Enhanced visualization with cinematography timeline plots

### Features
- **EmotionAnalyzer**: Detects 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **ShotTypeDetector**: Identifies shot scales (extreme close-up to extreme wide)
- **MotionAnalyzer**: Detects camera movements (pan, tilt, zoom, static)
- **ColorGradingAnalyzer**: Analyzes color temperature, LUTs, and film emulation
- **DirectorBrain**: Master controller orchestrating all analyses
- **CinematographyVisualizer**: Creates comprehensive timeline visualizations

### Technical Details
- Python 3.8+ required
- Supports async/await for parallel processing
- Modular architecture for easy extension
- Comprehensive error handling and logging
- Output in both JSON and human-readable formats

### Known Limitations
- LLaVA integration requires manual model download
- GPU recommended for optimal performance
- Some Instagram features may require authentication

## [Unreleased]

### Planned
- Fine-tuned model for cinematography analysis
- Batch processing for multiple videos
- Web UI for easier interaction
- Export to professional editing software formats
- Real-time analysis mode