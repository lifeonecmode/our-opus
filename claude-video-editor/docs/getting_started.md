# Getting Started with Claude Video Editor

Welcome to Claude Video Editor - the AI-powered video editing tool that creates professional videos without After Effects. This guide will get you up and running in minutes.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- Claude API key from Anthropic

### 1. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official site](https://ffmpeg.org/download.html) and add to PATH.

### 2. Install Claude Video Editor

```bash
# Clone the repository
git clone https://github.com/yourusername/claude-video-editor.git
cd claude-video-editor

# Install with pip
pip install -e .
```

### 3. Set Up Claude API Key

Get your API key from [Anthropic Console](https://console.anthropic.com/):

```bash
export CLAUDE_API_KEY="your_api_key_here"
```

Or create a `.env` file in your project:
```
CLAUDE_API_KEY=your_api_key_here
```

## 🎬 Your First Edit

### Method 1: Quick Start (Recommended)

1. **Organize your footage:**
```
my_project/
├── input/
│   ├── main_video.mp4
│   ├── broll_1.mp4
│   └── broll_2.mp4
├── audio/
│   └── background_music.mp3
└── graphics/
    └── logo.png
```

2. **Let Claude edit:**
```bash
claude-edit my_project/ --style viral
```

3. **Get results:**
- `viral_edit.mp4` - Main edit
- `viral_edit_tiktok.mp4` - TikTok version
- `viral_edit_instagram.mp4` - Instagram version
- `viral_edit_youtube.mp4` - YouTube version

### Method 2: Python API

```python
import asyncio
from claude_video_editor import ClaudeDecisionEngine

async def create_edit():
    # Initialize Claude
    engine = ClaudeDecisionEngine(
        project_dir="my_project/",
        claude_api_key="your_key_here"
    )
    
    # Analyze and edit
    analysis = await engine.analyze_project()
    decisions = await engine.create_edit_decisions(style="viral")
    await engine.execute_edit(decisions, "output.mp4")

asyncio.run(create_edit())
```

## 🎯 Understanding Project Structure

### Input Directory
Place your raw footage in `input/` for automatic processing:
- Videos are automatically categorized (speaker, product, lifestyle)
- Claude analyzes each video for emotions, composition, and quality
- All videos become available for editing decisions

### Traditional Structure (Optional)
For more control, organize footage by purpose:
```
project/
├── footage/
│   ├── main-speaker.mp4      # Primary talking head
│   ├── product-hero.mp4      # Product glamour shots
│   ├── transformation/       # Before/after content
│   └── b-roll/              # Supporting footage
├── audio/
│   ├── voiceover.wav        # Narration
│   ├── music.mp3            # Background music
│   └── sfx/                 # Sound effects
└── graphics/
    ├── logos/               # Brand assets
    ├── overlays/            # Text graphics
    └── transitions/         # Custom transitions
```

## 🤖 How Claude Makes Decisions

Claude analyzes your footage and creates edits based on:

### 1. Emotional Analysis
- Detects emotional peaks and valleys
- Places effects at high-impact moments
- Speeds up low-energy sections

### 2. Composition Analysis
- Identifies best shots and framing
- Determines optimal cut points
- Applies appropriate effects

### 3. Narrative Structure
- Creates engaging story flow
- Builds tension and resolution
- Includes strong calls-to-action

### 4. Platform Optimization
- Adjusts pacing for each platform
- Applies platform-specific color grading
- Optimizes aspect ratios and compression

## 🎨 Editing Styles

### Viral Style
Perfect for social media engagement:
- Fast-paced cuts and speed ramps
- Punch zooms on emotional peaks
- Bold yellow text overlays
- Strong calls-to-action

### Cinematic Style
For professional, polished content:
- Letterbox aspect ratio
- Teal and orange color grading
- Smooth transitions and fades
- Minimal, elegant text

### Fast-Paced Style
High-energy content:
- Quick cuts every 2-3 seconds
- Dynamic transitions
- Constant movement and effects

## 🔧 Customization

### Custom Styles
Create your own editing style:
```python
custom_style = {
    "pacing": "medium",           # slow, medium, fast
    "effects_intensity": 0.8,     # 0.0 to 1.0
    "text_frequency": "high",     # low, medium, high
    "color_grade": "warm",        # cool, neutral, warm
    "transitions": "smooth"       # cut, smooth, dynamic
}

decisions = await engine.create_edit_decisions(style=custom_style)
```

### Claude Rules
Modify Claude's behavior:
```python
engine.editing_rules = {
    "viral": {
        "hook_duration": 3,        # Seconds for opening hook
        "peak_zoom_scale": 1.2,    # Zoom intensity
        "low_energy_speed": 1.5,   # Speed up multiplier
        "text_overlay_count": 8,   # Number of text overlays
    }
}
```

## 📊 Performance Tips

### Optimization
- Use 1080p source footage (4K will be downscaled)
- Keep clips under 2 minutes for faster processing
- Organize files before running analysis

### Expected Processing Times
- **Analysis**: ~30 seconds per minute of footage
- **Decisions**: ~5 seconds regardless of length
- **Rendering**: ~45 seconds per minute of output

### Hardware Acceleration
On Apple Silicon Macs, Metal Performance Shaders are automatically used for faster rendering.

## 🎬 Advanced Features

### Watch Mode
Automatically process new videos:
```bash
claude-watch my_project/ --auto-edit --style viral
```

### Batch Processing
Process multiple projects:
```bash
for project in project_*/; do
    claude-edit "$project" --style viral
done
```

### Custom Effects
Add your own effects to the pipeline:
```python
from claude_video_editor import VideoEditor

editor = VideoEditor()
editor.register_effect("my_effect", my_effect_function)
```

## 🆘 Troubleshooting

### Common Issues

**"FFmpeg not found"**
- Install FFmpeg and ensure it's in your PATH
- Test with: `ffmpeg -version`

**"Claude API key invalid"**
- Check your API key in Anthropic Console
- Ensure environment variable is set correctly

**"No videos found"**
- Check file formats (mp4, mov, avi supported)
- Ensure videos are in input/ or footage/ directories

**"Rendering failed"**
- Check available disk space
- Ensure source videos are not corrupted
- Try with smaller test videos first

### Getting Help

1. Check the [API Reference](api_reference.md)
2. Review [examples/](../examples/) for working code
3. Open an issue on GitHub with:
   - Error messages
   - Project structure
   - System information

## 🚀 Next Steps

Now that you're set up:

1. **Try the examples** - Run the demo projects to see Claude in action
2. **Experiment with styles** - Test different editing approaches
3. **Customize workflows** - Adapt the tool to your specific needs
4. **Share your results** - Show us what you create!

Ready to create your first AI-edited video? Let's go! 🎬