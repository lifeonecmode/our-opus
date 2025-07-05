# 🎬 AI Video Editor - Code-Based Video Editing System

An open-source, code-based video editor that uses AI to automatically edit videos without After Effects. Edit videos programmatically using React-style state management with Claude as your AI decision engine.

## 🚀 Overview

This system provides two powerful approaches to video editing:

1. **Code-Based Editing**: Edit videos using React-style components and state management
2. **Claude AI Automation**: Let Claude analyze your footage and make all editing decisions

**No After Effects required!** Everything runs with FFmpeg and pure code.

## ✨ Key Features

- 🤖 **Claude as Decision Engine** - AI analyzes footage and creates edits automatically
- 📊 **Automatic Footage Analysis** - Comprehensive analysis of all videos in your project
- 🎯 **React-Style Video Editing** - Edit videos like building React components
- 🎨 **Multiple Editing Styles** - Viral, cinematic, fast-paced, and more
- 📱 **Platform Optimization** - Auto-export for TikTok, Instagram, YouTube
- ⚡ **Hardware Acceleration** - Uses Metal Performance Shaders on Apple Silicon
- 🔥 **No After Effects** - Pure FFmpeg-based rendering

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/video-ai-editor.git
cd video-ai-editor

# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg (required)
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu
```

## 📁 Project Structure

Organize your project with this structure:

```
/your_project/
├── /input/                     # Drop videos here for automatic processing
│   ├── video1.mp4              # Videos to be analyzed automatically
│   ├── video2.mp4              # Claude will detect and process these
│   └── ...                     # Any video format supported by FFmpeg
│
├── /footage/
│   ├── main-speaker.mp4         # Primary talking head
│   ├── bottle-hero.mp4          # Product glamour shot
│   ├── transformation/          # Before/after content
│   │   ├── before_001.mp4
│   │   └── after_001.mp4
│   └── b-roll/                  # Supporting footage
│       ├── lifestyle_shots/
│       └── product_details/
│
├── /audio/
│   ├── voiceover.wav           # 48kHz, -12dB normalized
│   ├── music.mp3               # Background music
│   └── sfx/                    # Sound effects
│       ├── boom.wav
│       ├── swoosh.wav
│       └── ding.wav
│
├── /graphics/                  # Visual assets and overlays
│   ├── logos/                  # Brand logos and watermarks
│   ├── overlays/               # Text overlays and graphics
│   └── transitions/            # Custom transition elements
│
└── /analysis/                  # Auto-generated analyses
```

## 🎯 Quick Start

### Option 1: Automatic Input Processing (NEW!)

Drop videos into the `/input/` directory for automatic analysis:

```bash
# Process all videos in input directory
python analyze_all_footage.py /path/to/project --input

# Watch directory for new videos and process automatically
python input_processor.py /path/to/project --watch

# Process and create edits automatically
python input_processor.py /path/to/project --auto-edit --style viral
```

### Option 2: Let Claude Edit Your Project

```bash
# 1. Analyze all your footage
python analyze_all_footage.py /path/to/project --audio

# 2. Let Claude create the edit
python claude_decision_engine.py /path/to/project --style viral

# Output files:
# - claude_edit_viral.mp4
# - claude_edit_viral_tiktok.mp4
# - claude_edit_viral_instagram.mp4
# - claude_edit_viral_youtube.mp4
```

### Option 3: Code-Based Manual Editing

```python
from video_ai_editor.core.video_state_editor import VideoStateEditor

# Create editor
editor = VideoStateEditor()

# Add clips
clip_id = editor.addClip("main-speaker.mp4", start=0)

# Add text with animation
editor.addText(
    "WAIT FOR IT...",
    style={"fontSize": 72, "color": "#FFD700"},
    start=0.5,
    duration=2.5,
    animate={"scale": [0, 1.2, 1]}
)

# Add effects
editor.addEffect(clip_id, "zoom", {
    "from": 1.0,
    "to": 1.1,
    "duration": 0.5
})

# Render
await editor.render("output.mp4")
```

## 🤖 Claude AI Features

Claude automatically:
- 📊 Analyzes emotional peaks in footage
- ✂️ Decides optimal cut points
- 📝 Places text overlays at perfect moments
- 🎯 Adds zoom effects on high-emotion moments
- 💨 Speeds up low-energy sections
- 🔊 Syncs sound effects with visual moments
- 🎨 Applies platform-specific optimizations

### Claude's Edit Structure

1. **HOOK (0-3s)** - Attention-grabbing opening
2. **SOCIAL PROOF (3-8s)** - Trust building
3. **TRANSFORMATION (8-11s)** - Before/after reveal
4. **STORY/BENEFITS (11-50s)** - Main content with B-roll
5. **URGENCY (50-57s)** - Create FOMO
6. **CTA (57-60s)** - Clear call to action

## 📊 Footage Analysis

The system analyzes:
- **Emotions**: 7 emotion detection with intensity
- **Shot Types**: ECU, CU, MS, WS, EWS classification
- **Motion**: Camera movements and energy levels
- **Composition**: Rule of thirds, symmetry, balance
- **Quality**: Cinematography scoring
- **Scenes**: Automatic scene detection

```bash
# Analyze all footage in a project
python analyze_all_footage.py /project/path --audio

# Outputs:
# - analysis/complete_footage_analysis.json
# - analysis/footage_analysis_report.md
# - analysis/audio_library_analysis.json
```

## 🎨 Editing Styles

### Viral Style
- Punch zooms on emotional peaks
- Fast pacing with speed ramps
- Yellow trust-building text
- Strong CTAs

### Cinematic Style
- Letterbox aspect ratio
- Teal/orange color grading
- Smooth transitions
- Minimal text

### Fast-Paced Style
- Quick cuts every 2-3 seconds
- High energy transitions
- Dynamic effects

## 📥 Input Directory Workflow

The input directory provides a drop-and-process workflow for automatic video handling:

### Basic Usage

1. **Drop videos into `/input/` directory**
   ```bash
   cp your_video.mp4 /project/input/
   ```

2. **Process all videos**
   ```bash
   python analyze_all_footage.py /project --input
   ```

### Watch Mode (Automatic Processing)

Watch the input directory and process new videos automatically:

```bash
# Basic watch mode
python input_processor.py /project --watch

# With automatic editing
python input_processor.py /project --watch --auto-edit --style viral
```

### How It Works

1. **Automatic Categorization**: Videos are analyzed and categorized as:
   - `speaker` - Talking head/person videos
   - `product` - Product shots and close-ups
   - `lifestyle` - B-roll and scene footage
   - `unknown` - Requires manual review

2. **Smart Analysis**: Each video receives:
   - Emotion detection
   - Shot type classification
   - Quality scoring
   - Usage recommendations

3. **Claude Integration**: Categorized videos are automatically available for Claude's editing decisions

### Input Processing Features

- **Batch Processing**: Process multiple videos at once
- **Watch Mode**: Monitor directory for new videos
- **Auto-Categorization**: Smart content detection
- **Quality Assessment**: Automatic quality scoring
- **Usage Suggestions**: AI-powered recommendations
- **Processed Tracking**: Avoid reprocessing videos

### Example Workflow

```bash
# 1. Set up project
mkdir my_project
cd my_project

# 2. Start watching input directory
python input_processor.py . --watch --auto-edit

# 3. Drop videos into input/
# (Videos are automatically processed)

# 4. Claude creates edit using all available footage
python claude_decision_engine.py . --style viral
```

## 🔧 Advanced Usage

### Declarative API (Like JSX)

```python
from video_ai_editor.core.video_state_editor import Video

editor = Video.sequence(
    duration=60,
    fps=30,
    children=[
        Video.clip("input.mp4", start=0),
        Video.text(
            "VIRAL TEXT",
            style={"fontSize": 72, "color": "#FFD700"},
            start=0.5,
            duration=3,
            animate={"scale": [0, 1.2, 1]}
        ),
        Video.effect("zoom", params={"from": 1.0, "to": 1.1})
    ]
)

await editor.render("output.mp4")
```

### Custom Claude Rules

Modify Claude's behavior in `claude_decision_engine.py`:

```python
self.editing_rules = {
    "viral": {
        "hook_duration": 5,      # Longer hook
        "peak_zoom": 1.2,        # More dramatic zooms
        "low_energy_speed": 2.0, # Faster speed-ups
    }
}
```

## 📦 Available Effects

- **Zoom**: Scale in/out with easing
- **Fade**: Fade in/out transitions
- **Blur**: Gaussian blur
- **Shake**: Camera shake
- **Speed**: Time remapping
- **Color**: Brightness, contrast, saturation
- **Split Screen**: Side-by-side comparison
- **Vignette**: Darkened edges

## 🚀 Performance

- Analyzes 60s footage in ~30 seconds
- Creates edit decisions in ~5 seconds
- Renders final video in ~45 seconds
- **Total: Under 2 minutes per video**

## 📱 Platform Exports

Automatic optimization for each platform:

| Platform | Modifications |
|----------|--------------|
| TikTok | +10% saturation, heavy compression |
| Instagram | -5% contrast, medium compression |
| YouTube | Standard color, light compression |

## 🔌 API Reference

### VideoStateEditor

```python
editor = VideoStateEditor()
editor.addClip(src, start, duration, trim)
editor.addEffect(clip_id, effect_type, params, timestamp)
editor.addText(text, style, start, duration, animate)
editor.render(output_path)
```

### ClaudeDecisionEngine

```python
engine = ClaudeDecisionEngine(project_dir)
await engine.analyze_project()
await engine.create_edit_decision_list(style)
await engine.execute_edit(decisions, output)
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- New effect types
- Additional editing styles
- Better scene detection
- More platform profiles

## 📄 License

MIT License - Free for commercial and personal use

---

**No After Effects. No manual editing. Just code and AI.** 🚀

Built with ❤️ for content creators who want AI-powered editing superpowers.