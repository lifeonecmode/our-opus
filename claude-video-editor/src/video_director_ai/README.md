# Video Director AI

Professional cinematographic video analysis tool that provides director-level insights into any video from YouTube, TikTok, Instagram, or local files.

## Features

- **Comprehensive Cinematographic Analysis**
  - Shot type detection (extreme close-up to extreme wide)
  - Camera movement analysis (pan, tilt, zoom, dolly, handheld)
  - Color grading and LUT detection
  - Composition analysis (rule of thirds, symmetry, leading lines)
  - Depth of field estimation

- **Emotion & Content Analysis**
  - Facial emotion recognition
  - Emotional arc tracking
  - Scene boundary detection
  - Audio transcription with Whisper

- **Professional Insights**
  - Director-style shot descriptions
  - Cinematographic pattern recognition
  - Visual narrative structure analysis
  - Pacing and rhythm evaluation

- **Multi-Platform Support**
  - YouTube
  - TikTok
  - Instagram
  - Local video files

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video_director_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models (if not automatic):
```bash
# Whisper will download on first use
# For LLaVA support (optional), follow instructions at:
# https://github.com/haotian-liu/LLaVA
```

## Usage

### Basic Usage

Analyze a YouTube video:
```bash
python main.py https://www.youtube.com/watch?v=VIDEO_ID
```

Analyze a local video:
```bash
python main.py /path/to/video.mp4
```

### Advanced Options

```bash
# Specify output format
python main.py video.mp4 --format json

# Custom frame analysis interval (default: 1 second)
python main.py video.mp4 --interval 2

# Use specific Whisper model
python main.py video.mp4 --whisper-model large-v3

# Verbose output
python main.py video.mp4 --verbose
```

### Output Formats

- **JSON**: Complete structured data (`analysis.json`)
- **Markdown**: Human-readable report (`analysis_report.md`)
- **CSV**: Timeline data for further analysis (`timeline_analysis.csv`)
- **Visualization**: Comprehensive charts and graphs (`visualization.png`)

## Output Structure

```
output/
├── analysis_YYYYMMDD_HHMMSS/
│   ├── analysis.json           # Complete analysis data
│   ├── analysis_report.md      # Markdown report
│   ├── timeline_analysis.csv   # Timeline data
│   └── visualization.png       # Multi-panel visualization
└── clips/
    └── cinematographic_YYYYMMDD_HHMMSS/
        ├── clip_001_emotional_peak_surprise.mp4
        ├── clip_002_high_cinematography_score.mp4
        └── ...
```

## Analysis Components

### 1. Shot Type Detection
- Extreme Close-up (ECU)
- Close-up (CU)
- Medium Shot (MS)
- Wide Shot (WS)
- Extreme Wide Shot (EWS)

### 2. Camera Movement Analysis
- Static shots
- Pan (left/right)
- Tilt (up/down)
- Zoom (in/out)
- Dolly movements
- Handheld shake detection

### 3. Color Grading Analysis
- Color temperature (warm/cool/neutral)
- Common LUT detection
- Film stock emulation
- Popular color grading styles:
  - Teal & Orange (blockbuster)
  - Bleach Bypass (gritty)
  - Day-for-Night
  - Matrix Green
  - Mexico Filter

### 4. Emotional Analysis
- 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Emotional intensity tracking
- Peak moment identification
- Emotional arc classification

### 5. Cinematography Scoring
- Emotional impact
- Visual dynamics
- Aesthetic quality
- Overall cinematography score

## Example Output

```markdown
# Video Director Analysis Report

## Overall Assessment
The video follows a complex narrative structure with a rollercoaster emotional arc. 
The cinematographic style is characterized by intimate framing, dynamic camera work, 
and Teal_Orange color grading. The brisk pacing (average shot duration: 3.2s) 
creates a dynamic visual rhythm.

## Key Moments
1. 45.3s - emotional_peak
   Extreme Close-up (ECU): Face fills entire frame, showing intense emotion. 
   Camera pushing in creates psychological pressure.
   > "I can't do this anymore..."
```

## Requirements

- Python 3.8+
- OpenCV
- PyTorch (for Whisper and advanced features)
- FFmpeg (for video processing)
- 8GB+ RAM recommended
- GPU recommended for faster processing

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Ubuntu)

2. **Out of memory**
   - Use smaller Whisper model: `--whisper-model base`
   - Increase frame interval: `--interval 5`

3. **Download errors**
   - For Instagram, ensure you're logged in to Chrome
   - For private videos, authentication may be required

## Contributing

Contributions are welcome! Please check the CHANGELOG.md for recent updates.

## License

This project is licensed under the MIT License.