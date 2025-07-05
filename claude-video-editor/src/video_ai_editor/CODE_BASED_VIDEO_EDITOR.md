# 🎬 Code-Based Video Editor (No After Effects!)

## Overview

This is a **React-style video editor** that lets you edit videos programmatically using state management and components - no After Effects required! It works like Framer Motion but for video editing.

## 🚀 How It Works

Instead of using After Effects, we:
1. **Manage video edits as state** (like React components)
2. **Apply effects programmatically** through code
3. **Render directly with FFmpeg** to H.264 MP4

## 📝 Basic Usage

### State-Based Editing (Like React Hooks)

```python
from video_ai_editor.core.video_state_editor import VideoStateEditor

# Create editor instance
editor = VideoStateEditor()

# Add video clip
clip_id = editor.addClip("input.mp4", start=0)

# Add text overlay with animation
editor.addText(
    "VIRAL TEXT",
    style={
        "fontSize": 72,
        "color": "#FFD700",
        "position": {"x": 540, "y": 960}
    },
    start=0.5,
    duration=3,
    animate={
        "scale": [0, 1.2, 1],  # Scale animation
        "opacity": [0, 1],     # Fade in
        "duration": 0.5
    }
)

# Add zoom effect on emotional peak
editor.addEffect(clip_id, "zoom", {
    "from": 1.0,
    "to": 1.1,
    "duration": 0.5
}, timestamp=5.0)

# Render to MP4
await editor.render("output.mp4")
```

### Declarative API (Like JSX)

```python
from video_ai_editor.core.video_state_editor import Video

# Create video sequence declaratively
editor = Video.sequence(
    duration=60,
    fps=30,
    children=[
        Video.clip("input.mp4", start=0),
        
        Video.text(
            "WAIT FOR IT...",
            style={"fontSize": 72, "color": "#FFD700"},
            start=0.5,
            duration=2.5,
            animate={"scale": [0, 1.2, 1]}
        ),
        
        Video.effect(
            clip_id="clip_0",
            effect_type="zoom",
            params={"from": 1.0, "to": 1.1}
        )
    ]
)

await editor.render("viral_output.mp4")
```

## 🎯 Features

### Video Effects
- **Zoom**: Punch zoom on key moments
- **Fade**: Fade in/out transitions
- **Blur**: Gaussian blur
- **Shake**: Camera shake effect
- **Speed**: Speed ramping
- **Color**: Brightness, contrast, saturation

### Text Animations
- **Scale**: Grow/shrink animations
- **Opacity**: Fade effects
- **Position**: Movement animations
- **Shake**: Text shake
- **Box**: Background box

### Analysis-Driven Edits
```python
# Apply edits based on video analysis
editor.applyAnalysis(analysis_data, style="viral")

# Automatically adds:
# - Zoom on emotional peaks
# - Speed up low energy sections
# - Text overlays at key moments
# - Platform-specific optimizations
```

## 🔧 Complete Example

```python
async def create_viral_video(video_path: str):
    # Initialize editor
    editor = VideoStateEditor()
    
    # Add main clip
    clip_id = editor.addClip(video_path)
    
    # Hook section (0-3s)
    editor.addText(
        "WAIT FOR IT...",
        style={
            "fontSize": 72,
            "color": "#FFD700",
            "position": {"x": 540, "y": 960}
        },
        start=0.5,
        duration=2.5,
        animate={
            "scale": [0, 1.2, 1],
            "opacity": [0, 1]
        }
    )
    
    # Find emotional peaks and add zoom
    for peak in emotional_peaks:
        editor.addEffect(clip_id, "zoom", {
            "from": 1.0,
            "to": 1.1,
            "duration": 0.5
        }, timestamp=peak['timestamp'])
    
    # Call to action (57-60s)
    editor.addText(
        "TAP THE LINK NOW",
        style={
            "fontSize": 72,
            "color": "#FFD700",
            "box": True
        },
        start=57,
        duration=3,
        animate={
            "scale": [0, 1.3, 1],
            "shake": {"x": 5, "y": 5}
        }
    )
    
    # Render with platform exports
    await editor.render("output.mp4")
    # Automatically creates:
    # - output.mp4 (master)
    # - output_tiktok.mp4 (+10% saturation)
    # - output_instagram.mp4 (-5% contrast)
    # - output_youtube.mp4 (standard)
```

## 🛠️ How FFmpeg Rendering Works

The system converts your state/components into FFmpeg filter graphs:

```python
# Your code:
editor.addEffect(clip_id, "zoom", {"from": 1.0, "to": 1.1})

# Becomes FFmpeg filter:
scale=w=iw*1.0+iw*(1.1-1.0)*t/0.5:h=ih*1.0+ih*(1.1-1.0)*t/0.5

# Text overlay:
drawtext=text='WAIT FOR IT':fontsize=72:fontcolor=yellow:x=(w-text_w)/2:y=960
```

## 📦 Installation

```bash
# Install FFmpeg (required)
brew install ffmpeg  # macOS

# Install Python dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Basic viral edit
python examples/viral_edit_example.py input.mp4

# With pre-computed analysis
python examples/viral_edit_example.py input.mp4 --analysis analysis.json

# Using declarative API
python examples/viral_edit_example.py input.mp4 --declarative
```

## 🎨 State Management

The editor uses React-like state management:

```python
# State updates trigger re-renders
editor.setState({
    "duration": 60,
    "fps": 30,
    "resolution": (1080, 1920)
})

# Component state
clips = editor.useState([])
clips.append(new_clip)

# Side effects
editor.useEffect(lambda: print("State changed"), [clips])
```

## 🔥 Advantages

1. **No After Effects Required** - Pure code-based editing
2. **State Management** - Edit videos like React components
3. **Programmatic Control** - Full control through code
4. **Analysis Integration** - Automatic edits from AI analysis
5. **Platform Optimization** - Auto-export for TikTok/Instagram/YouTube
6. **Fast Rendering** - Direct FFmpeg encoding
7. **Hardware Acceleration** - Uses VideoToolbox on macOS

## 📊 Workflow Integration

```python
# 1. Analyze video
analyzer = InstagramFrameAnalyzer(video_path)
analysis = analyzer.analyze_video()

# 2. Create editor
editor = VideoStateEditor()

# 3. Apply automatic edits
editor.createViralVideo(video_path, analysis)

# 4. Render
await editor.render("viral_output.mp4")
```

## 🎯 Use Cases

- **Viral Content**: Auto-generate viral edits from any video
- **Social Media**: Platform-optimized exports
- **Batch Processing**: Edit multiple videos programmatically
- **A/B Testing**: Generate variations with different styles
- **Template System**: Create reusable edit templates

---

**No After Effects. No GUI. Just code.** 🚀

Edit videos with the same mental model as building React apps!