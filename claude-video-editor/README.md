# 🎬 Claude Video Editor

> AI-powered video editing with Claude Code integration - No After Effects required!

Transform your raw footage into professional, viral-ready videos using Claude's AI decision-making and code-based editing. Drop videos, let Claude analyze and edit, get platform-optimized outputs.

## 🚀 What It Does

- **AI Analysis**: Claude analyzes your footage for emotions, timing, and composition
- **Automatic Editing**: Creates professional edits with zero manual work
- **Platform Optimization**: Exports for TikTok, Instagram, YouTube automatically
- **Code-Based**: Edit videos like writing React components
- **No After Effects**: Pure FFmpeg-based rendering

## ⚡ 30-Second Demo

```python
from claude_video_editor import ClaudeDecisionEngine, transcribe, load_model

# Initialize with your Claude API key
engine = ClaudeDecisionEngine(
    project_dir="./my_videos/",
    claude_api_key="your_key_here"
)

# Optional: Add transcription for speech-aware editing
model = load_model("base")
transcription = transcribe(model, "video.mp4")

# Let Claude edit with transcription context
decisions = engine.create_edit_decisions(
    style="viral",
    transcription=transcription
)

# Outputs created:
# - viral_edit.mp4
# - viral_edit_tiktok.mp4  
# - viral_edit_instagram.mp4
# - viral_edit_youtube.mp4
```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/claude-video-editor.git
cd claude-video-editor

# One-command setup
pip install -e .

# Install FFmpeg (required)
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu

# Set your Claude API key
export CLAUDE_API_KEY="your_key_here"
```

## 🎯 Quick Start

### 1. Organize Your Project

```
your_project/
├── input/              # Drop videos here
│   ├── main_video.mp4
│   ├── broll_1.mp4
│   └── broll_2.mp4
├── audio/              # Background music & SFX
└── graphics/           # Logos, overlays
```

### 2. Let Claude Edit

```bash
# Process and edit automatically
python -m claude_video_editor.quick_start your_project/ --style viral

# Or watch directory for new videos
python -m claude_video_editor.watch your_project/ --auto-edit
```

### 3. Get Results

- **Analysis**: Complete footage analysis in JSON
- **Edits**: Professional video edits
- **Variants**: Platform-specific versions
- **Timeline**: Edit decision list for review

## 🤖 Claude AI Features

Claude automatically handles:

- 📊 **Emotional Analysis**: Detects peaks and valleys in footage
- ✂️ **Smart Cutting**: Finds optimal cut points
- 🎯 **Effect Placement**: Adds zooms, fades, and transitions at perfect moments
- 📝 **Text Overlays**: Places engaging text at high-impact moments
- 🎨 **Color Grading**: Applies platform-specific color adjustments
- 🔊 **Audio Sync**: Syncs sound effects with visual beats
- 🎤 **Speech Analysis**: Uses Whisper transcription for speech-aware editing

## 🎨 Editing Styles

### Viral Style
- Punch zooms on emotional peaks
- Fast-paced cuts with speed ramps
- Yellow trust-building text overlays
- Strong calls-to-action

### Cinematic Style  
- Letterbox aspect ratio
- Teal/orange color grading
- Smooth transitions and fades
- Minimal, elegant text

### Fast-Paced Style
- Quick cuts every 2-3 seconds
- High-energy transitions
- Dynamic zoom effects

## 🔧 Code-Based Editing

Edit videos programmatically with React-style components:

```python
from claude_video_editor import VideoEditor

editor = VideoEditor()

# Add clips
main_clip = editor.add_clip("speaker.mp4", start=0, duration=10)

# Add effects
editor.add_effect(main_clip, "zoom", {
    "from": 1.0,
    "to": 1.15,
    "duration": 0.5,
    "easing": "ease_out"
})

# Add text overlay
editor.add_text(
    "WAIT FOR IT...",
    style={
        "fontSize": 72,
        "color": "#FFD700",
        "fontWeight": "bold",
        "stroke": "#000000",
        "strokeWidth": 2
    },
    start=0.5,
    duration=3,
    animate={"scale": [0, 1.2, 1]}
)

# Render
editor.render("output.mp4")
```

## 📱 Platform Optimization

Automatic optimization for each platform:

| Platform | Modifications |
|----------|---------------|
| TikTok | 9:16 aspect, +10% saturation, heavy compression |
| Instagram | 4:5 aspect, -5% contrast, medium compression |
| YouTube | 16:9 aspect, standard color, light compression |

## 🔌 API Reference

### ClaudeDecisionEngine
```python
engine = ClaudeDecisionEngine(
    project_dir="./project",
    claude_api_key="your_key",
    style="viral"
)

# Analyze project
analysis = await engine.analyze_project()

# Create edit decisions
decisions = await engine.create_edit_decisions()

# Execute edit
await engine.execute_edit(decisions, "output.mp4")
```

### VideoEditor
```python
editor = VideoEditor()

# Core methods
editor.add_clip(src, start, duration, trim)
editor.add_effect(clip_id, effect_type, params)
editor.add_text(text, style, start, duration, animate)
editor.add_audio(src, start, volume, fade_in, fade_out)
editor.render(output_path, platform="universal")
```

## 📊 Performance

- **Analysis**: 60s footage analyzed in ~30 seconds
- **Decisions**: Edit decisions created in ~5 seconds  
- **Rendering**: Final video rendered in ~45 seconds
- **Total**: Professional edit in under 2 minutes

## 🎬 Examples

Check out `examples/demo_projects/viral_video_example/` to see:
- How Claude analyzed the footage (`claude_decisions/`)
- The final edited video (`output/viral_edit.mp4`)
- Platform-specific variants

## 🤝 Contributing

Areas for improvement:
- Additional editing styles
- New effect types
- Better scene detection
- More platform profiles

## 📄 License

MIT License - Free for commercial and personal use

---

**No After Effects. No manual editing. Just Claude and code.** 🚀

Built for content creators who want AI-powered editing superpowers.