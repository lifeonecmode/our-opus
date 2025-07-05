# 🎬 Darwin AI Video Editor - Implementation Guide

## Overview

Darwin is an open-source AI-powered video editor for macOS that automatically edits videos based on frame-by-frame analysis data. It uses Adobe After Effects as the editing engine, controlled headlessly through ExtendScript.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video Input    │────▶│ Frame Analysis   │────▶│ Decision Engine │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Rendered Video  │◀────│ After Effects    │◀────│ Edit Commands   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 🚀 How It Works

### 1. **Video Analysis Pipeline**
The system analyzes videos frame-by-frame, extracting:
- **Emotions**: 7 emotion detection with intensity
- **Shot Types**: ECU, CU, MS, WS, EWS
- **Motion**: Camera movements and intensity
- **Composition**: Rule of thirds, symmetry
- **Color**: Grading styles and temperature
- **Cinematography Score**: Overall quality metric

### 2. **Decision Engine**
Converts analysis data into editing decisions:
```python
# Example: Emotional peak → Zoom emphasis
if emotion_intensity > 0.8:
    add_edit("punch_zoom", timestamp, scale=110)

# Low energy → Speed ramp
if cinematography_score < 0.3:
    add_edit("speed_change", timestamp, speed=1.5)
```

### 3. **After Effects Automation**
Uses ExtendScript to control AE headlessly:
```javascript
// Create project
var project = app.newProject();

// Import video
var video = project.importFile(new ImportOptions(File(videoPath)));

// Apply edits
layer.property("Scale").setValueAtTime(5.0, [110, 110]);
```

### 4. **Platform Optimization**
Automatic adjustments for each platform:
- **TikTok**: +10% saturation, heavy compression
- **Instagram**: -5% contrast, medium compression  
- **YouTube**: Standard color, light compression

## 📦 Key Components

### `darwin.py` - Main Orchestrator
```python
# Edit a video with viral style for TikTok
darwin = DarwinVideoEditor()
output = await darwin.edit_video(
    "input.mp4",
    style="viral",
    platform="tiktok"
)
```

### `ae_controller.py` - After Effects Control
```python
# Create and render AE project
controller = AfterEffectsController()
project = controller.create_project_from_analysis(video, analysis)
controller.render_project(project, "output.mp4")
```

### `decision_engine.py` - AI Editing Logic
```python
# Generate edit decisions from analysis
engine = EditDecisionEngine()
decisions = engine.generate_decisions(
    analysis_data,
    style="viral",
    platform="tiktok"
)
```

### `project_manager.py` - File Organization
```python
# Create organized project structure
manager = ViralProjectManager()
project = manager.create_project("video.mp4")
manager.prepare_for_platform(project, "tiktok")
```

## 🎨 Editing Styles

### Viral Style
- Punch zooms on emotional peaks
- Speed ramping for pacing
- Yellow text overlays for trust
- Pattern interrupts for attention
- Strong CTA at end

### Cinematic Style
- Letterbox aspect ratio
- Teal/orange color grading
- Slow motion on dramatic moments
- Smooth transitions
- Minimal text overlays

### Documentary Style
- Lower thirds for context
- Ken Burns on still shots
- Neutral color grading
- Longer takes
- Informational overlays

### Fast-Paced Style
- Quick cuts every 2-3 seconds
- 1.2x speed overall
- High energy transitions
- Dynamic camera shakes
- Beat-synced editing

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
cd video_ai_editor
pip install -r requirements.txt
```

### 2. Setup After Effects
```bash
python setup_ae.py
```
This will:
- Find AE installation
- Install automation scripts
- Configure preferences
- Create CLI helper

### 3. Configure in After Effects
1. Open After Effects
2. File → Scripts → Run Script File
3. Run `Darwin_Configure.jsx`
4. Test with `Darwin_Test.jsx`

### 4. Test Darwin
```bash
python darwin.py --test
```

## 📊 Usage Examples

### Basic Edit
```bash
python darwin.py edit video.mp4 --style viral --platform tiktok
```

### With Pre-computed Analysis
```bash
python darwin.py edit video.mp4 --analysis analysis.json --style cinematic
```

### Batch Processing
```bash
python darwin.py batch /videos/folder --style fast_paced --platform instagram
```

## 🎯 Automatic Editing Features

### Based on Emotions
- **Happy moments** → Warm color boost
- **Surprise peaks** → Zoom emphasis  
- **Sad sections** → Slower pacing
- **Energetic** → Quick cuts

### Based on Shot Types
- **Close-ups** → Face enhancement
- **Wide shots** → Establish context
- **Medium shots** → Standard pacing
- **Extreme close** → Detail emphasis

### Based on Motion
- **High motion** → Match cuts
- **Camera pans** → Smooth transitions
- **Static shots** → Ken Burns effect
- **Shaky footage** → Stabilization

### Based on Cinematography Score
- **High score** → Minimal edits
- **Low score** → Heavy processing
- **Medium score** → Balanced approach

## 🚦 Workflow Integration

### 1. Download Instagram Video
```bash
python analyze_instagram_workflow.py https://instagram.com/reel/xxx
```

### 2. Analyze Frames
```bash
python instagram_frame_analyzer.py video.mp4 --interval 0.1
```

### 3. Edit with Darwin
```bash
python darwin.py edit video.mp4 --analysis analysis.json --style viral
```

### 4. Export for Platform
Output automatically optimized for target platform with proper encoding settings.

## 🔮 Future Enhancements

1. **Real-time Preview**: Live preview during editing
2. **Custom Styles**: User-defined editing styles
3. **Music Sync**: Beat detection and sync
4. **AI Voice**: Automatic voiceover generation
5. **Multi-cam**: Support for multiple angles
6. **3D Effects**: Integration with C4D
7. **Cloud Rendering**: Distributed processing

## 🤝 Contributing

Darwin is open-source! Areas for contribution:
- New editing styles
- Effect presets
- Platform profiles
- Analysis improvements
- Performance optimization

## 📝 Technical Notes

### Performance Optimization
- Uses Metal Performance Shaders on Apple Silicon
- Parallel processing for analysis
- Cached preview renders
- Efficient ExtendScript execution

### Headless Operation
- No UI required for editing
- Command-line rendering via aerender
- Batch processing support
- Remote operation capable

### Extensibility
- Plugin architecture for custom effects
- Style templates in JSON
- Platform profiles configurable
- Analysis interpreters pluggable

---

**Darwin brings AI-powered editing to everyone.** The system automatically creates engaging, platform-optimized videos by understanding content at a cinematographic level and applying professional editing techniques.