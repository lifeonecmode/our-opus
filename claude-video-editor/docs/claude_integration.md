# Claude Integration Guide

This guide explains how Claude Video Editor integrates with Anthropic's Claude API to make intelligent video editing decisions.

## 🤖 How Claude Analyzes Videos

Claude doesn't just randomly cut videos - it uses sophisticated analysis to make intelligent editing decisions based on content, emotion, and best practices.

### Analysis Pipeline

1. **Frame Extraction**: Key frames are extracted from videos
2. **Visual Analysis**: Claude analyzes composition, lighting, and subjects
3. **Emotional Mapping**: Identifies emotional peaks and valleys
4. **Narrative Structure**: Maps content to proven viral structures
5. **Edit Decisions**: Creates precise timeline of cuts, effects, and overlays

## 🔧 API Integration

### Setting Up Claude

```python
from claude_video_editor import ClaudeDecisionEngine

# Initialize with API key
engine = ClaudeDecisionEngine(
    project_dir="./my_project",
    claude_api_key="your_api_key_here",
    model="claude-3-5-sonnet-20241022"  # Optional: specify model
)
```

### Environment Variables

```bash
# Required
export CLAUDE_API_KEY="your_api_key_here"

# Optional
export CLAUDE_MODEL="claude-3-5-sonnet-20241022"
export CLAUDE_MAX_TOKENS=4000
export CLAUDE_TEMPERATURE=0.3
```

### Configuration File

Create `claude_config.json`:
```json
{
  "api_key": "your_api_key_here",
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 4000,
  "temperature": 0.3,
  "analysis_depth": "detailed",
  "editing_style": "viral"
}
```

## 📊 Claude's Analysis Process

### 1. Project Structure Analysis

Claude first understands your project:
```python
analysis = await engine.analyze_project()
```

**What Claude looks for:**
- Available footage and categorization
- Audio assets and quality
- Graphics and branding elements
- Project goals and target audience

### 2. Content Analysis

For each video, Claude analyzes:

**Visual Elements:**
- Shot composition and framing
- Lighting and color quality
- Subject positioning and movement
- Background and environment

**Emotional Content:**
- Facial expressions and body language
- Energy levels and pacing
- Emotional arc and transitions
- Engagement potential

**Technical Quality:**
- Resolution and clarity
- Audio quality and sync
- Stability and camera movement
- Overall production value

### 3. Narrative Mapping

Claude maps content to proven structures:

**Viral Video Structure:**
```
0-3s:    Hook (attention-grabbing opening)
3-8s:    Social Proof (trust building)
8-11s:   Transformation (before/after reveal)
11-50s:  Story/Benefits (main content)
50-57s:  Urgency (FOMO creation)
57-60s:  Call to Action (clear next steps)
```

**Cinematic Structure:**
```
0-10s:   Establishing shot (set the scene)
10-20s:  Character introduction
20-40s:  Conflict/tension building
40-50s:  Climax/resolution
50-60s:  Denouement/conclusion
```

## 🎬 Edit Decision Making

### Timeline Creation

Claude creates detailed edit decisions:

```python
decisions = await engine.create_edit_decisions(style="viral")
```

**Decision Types:**
- **Cuts**: Precise in/out points based on content flow
- **Effects**: Zooms, fades, and transitions at optimal moments
- **Text**: Overlay placement and timing
- **Audio**: Music sync and sound effect placement
- **Color**: Grading decisions for mood and platform

### Effect Placement Logic

**Zoom Effects:**
- Applied at emotional peaks (detected via facial analysis)
- Intensity based on content importance (1.1x to 1.3x)
- Duration optimized for platform (0.3s-0.8s)

**Text Overlays:**
- Placed at high-attention moments
- Style matches content mood
- Timing ensures readability

**Speed Ramps:**
- Applied to low-energy sections
- Intensity based on content density
- Smooth transitions maintain flow

## 🎨 Style Customization

### Predefined Styles

```python
# Viral style for social media
await engine.create_edit_decisions(style="viral")

# Cinematic style for professional content
await engine.create_edit_decisions(style="cinematic")

# Fast-paced style for high-energy content
await engine.create_edit_decisions(style="fast-paced")
```

### Custom Style Configuration

```python
custom_style = {
    "pacing": {
        "cuts_per_minute": 15,
        "average_shot_length": 4.0,
        "speed_variance": 0.3
    },
    "effects": {
        "zoom_intensity": 1.15,
        "zoom_frequency": "high",
        "transition_style": "smooth"
    },
    "text": {
        "overlay_count": 6,
        "font_size": 72,
        "color": "#FFD700",
        "animation": "scale"
    },
    "audio": {
        "music_volume": 0.3,
        "sfx_enabled": True,
        "voice_enhancement": True
    }
}

decisions = await engine.create_edit_decisions(style=custom_style)
```

## 🧠 Advanced Claude Features

### Contextual Understanding

Claude considers broader context:

**Brand Consistency:**
- Maintains consistent visual style
- Applies brand colors and fonts
- Includes appropriate branding elements

**Audience Targeting:**
- Adjusts pacing for target demographic
- Selects appropriate music and effects
- Optimizes engagement strategies

**Platform Optimization:**
- Modifies editing for TikTok vs YouTube
- Adjusts aspect ratios and compression
- Optimizes for platform algorithms

### Adaptive Learning

Claude learns from your preferences:

```python
# Provide feedback on edits
engine.rate_edit(edit_id="viral_001", rating=4.5, notes="Great pacing, reduce zoom intensity")

# Claude adapts future decisions
engine.update_preferences(user_feedback)
```

## 🔍 Debugging Claude Decisions

### Verbose Mode

Enable detailed logging to understand Claude's reasoning:

```python
engine = ClaudeDecisionEngine(
    project_dir="./project",
    claude_api_key="your_key",
    verbose=True
)
```

### Decision Explanation

Get explanations for edit decisions:

```python
decisions = await engine.create_edit_decisions(style="viral", explain=True)

for decision in decisions['clips']:
    print(f"Cut at {decision['timestamp']}: {decision['reasoning']}")
```

### Analysis Export

Export Claude's analysis for review:

```python
# Export detailed analysis
analysis = await engine.analyze_project()
engine.export_analysis(analysis, "project_analysis.json")

# Export edit decisions
decisions = await engine.create_edit_decisions(style="viral")
engine.export_decisions(decisions, "edit_decisions.json")
```

## 🚀 Performance Optimization

### Efficient API Usage

**Batch Processing:**
```python
# Process multiple videos in one request
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
analysis = await engine.batch_analyze(videos)
```

**Caching:**
```python
# Enable analysis caching
engine.enable_cache(cache_dir="./cache")

# Analysis results are cached and reused
analysis = await engine.analyze_project()  # Cached after first run
```

### Rate Limiting

Handle API rate limits gracefully:

```python
engine = ClaudeDecisionEngine(
    project_dir="./project",
    claude_api_key="your_key",
    rate_limit={"requests_per_minute": 50}
)
```

## 🔒 Security and Privacy

### API Key Security

**Best Practices:**
- Never commit API keys to version control
- Use environment variables or secure config files
- Rotate keys regularly
- Limit API key permissions

**Key Management:**
```python
# Use environment variables
import os
api_key = os.getenv("CLAUDE_API_KEY")

# Or use secure config
from claude_video_editor.security import SecureConfig
config = SecureConfig.load_encrypted("config.enc")
```

### Data Privacy

**Local Processing:**
- Video analysis happens locally
- Only metadata sent to Claude API
- No raw video data transmitted

**Data Retention:**
- Analysis results stored locally
- No data retained by Claude API
- User controls all processing artifacts

## 🛠️ Troubleshooting

### Common Issues

**"Claude API timeout"**
- Reduce analysis complexity
- Use smaller video segments
- Check network connectivity

**"Invalid API key"**
- Verify key in Anthropic Console
- Check environment variable setup
- Ensure key has required permissions

**"Rate limit exceeded"**
- Implement request spacing
- Use batch processing
- Consider upgrading API plan

### Advanced Debugging

**Enable Debug Mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

engine = ClaudeDecisionEngine(
    project_dir="./project",
    claude_api_key="your_key",
    debug=True
)
```

**Custom Error Handling:**
```python
try:
    decisions = await engine.create_edit_decisions(style="viral")
except ClaudeAPIError as e:
    print(f"Claude API error: {e}")
except VideoAnalysisError as e:
    print(f"Analysis error: {e}")
```

## 🎯 Best Practices

### Project Organization

**Structure for Success:**
```
project/
├── input/              # Raw footage
├── analysis/           # Claude analysis results
├── decisions/          # Edit decision logs
├── output/             # Final videos
└── config/             # Claude configuration
```

### Workflow Optimization

1. **Start Simple**: Begin with basic viral style
2. **Iterate**: Refine based on results
3. **Customize**: Adapt to your specific needs
4. **Scale**: Apply learnings to multiple projects

### Quality Assurance

**Review Process:**
1. Check Claude's analysis results
2. Verify edit decisions make sense
3. Test render with small segments
4. Review final output quality
5. Gather feedback and iterate

This integration allows you to leverage Claude's intelligence while maintaining full control over your video editing process. The AI handles the complex analysis and decision-making, while you focus on creativity and results.