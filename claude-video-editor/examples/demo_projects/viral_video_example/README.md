# Viral Video Example - How Claude Made This Edit

This example demonstrates how Claude analyzed raw footage and created a professional viral-style video edit.

## 📁 Project Structure

```
viral_video_example/
├── input/                          # Original footage
├── output/                         # Final edited videos
│   └── viral_edit.mp4             # Claude's final edit
├── claude_decisions/              # Claude's AI analysis
│   ├── scene_analysis.json       # Detailed scene-by-scene analysis
│   ├── timeline.jsonl            # Edit timeline decisions
│   └── narrative.txt             # Generated narrative structure
└── README.md                     # This file
```

## 🤖 Claude's Analysis Process

### 1. Scene Analysis (`scene_analysis.json`)
Claude analyzed the footage and identified:
- **Emotional peaks and valleys**
- **Shot composition and framing**
- **Motion and energy levels**
- **Optimal cut points**
- **Text overlay opportunities**

### 2. Timeline Creation (`timeline.jsonl`)
Based on the analysis, Claude created a detailed timeline with:
- **Precise cut points** down to the frame
- **Effect timing** for zooms, fades, and transitions
- **Text overlay placement** at emotional peaks
- **Speed ramp decisions** for pacing

### 3. Narrative Structure (`narrative.txt`)
Claude generated a narrative structure following viral video best practices:
- **Hook** (0-3s): Attention-grabbing opening
- **Social Proof** (3-8s): Trust-building content
- **Transformation** (8-11s): Before/after reveal
- **Story/Benefits** (11-50s): Main content with B-roll
- **Urgency** (50-57s): FOMO creation
- **Call to Action** (57-60s): Clear next steps

## 🎬 Final Output

The final video demonstrates:
- **Viral-style editing** with punch zooms and fast pacing
- **Strategic text overlays** placed at emotional peaks
- **Professional color grading** optimized for social media
- **Platform-specific variants** for TikTok, Instagram, and YouTube

## 🔍 Key Insights

**What made this edit successful:**
1. **Data-driven decisions**: Every cut based on emotional analysis
2. **Precise timing**: Effects synchronized to content peaks
3. **Viral structure**: Following proven engagement patterns
4. **Professional polish**: Color grading and effects enhance the story

**Claude's editing decisions:**
- Identified 12 emotional peaks for zoom effects
- Created 8 strategic text overlays
- Applied 15 precise cuts for optimal pacing
- Generated platform-specific color adjustments

## 🚀 Try It Yourself

To recreate this edit:

```bash
# Run the example
python ../quick_start.py

# Or use the command line
claude-edit viral_video_example/ --style viral
```

This example shows how Claude can transform raw footage into professional, viral-ready content with zero manual editing required.