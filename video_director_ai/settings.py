import os
from pathlib import Path
from typing import List, Dict

class Config:
    # Dynamic paths based on project root
    PROJECT_ROOT = Path(__file__).parent
    OUTPUT_DIR = PROJECT_ROOT / "output"
    CLIPS_DIR = OUTPUT_DIR / "clips"
    ANALYSIS_DIR = OUTPUT_DIR / "analysis"
    FRAMES_DIR = OUTPUT_DIR / "frames"
    
    # Create directories if they don't exist
    for dir_path in [OUTPUT_DIR, CLIPS_DIR, ANALYSIS_DIR, FRAMES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
    LLAVA_MODEL_PATH = os.getenv("LLAVA_MODEL", "liuhaotian/llava-v1.6-34b")
    
    # Analysis settings
    FRAME_INTERVAL = 1  # seconds
    SCENE_THRESHOLD = 30.0  # PySceneDetect threshold
    EMOTION_THRESHOLD = 0.3  # Minimum emotion score to consider
    MOTION_THRESHOLD = 2.0  # Optical flow threshold for movement detection
    
    # Supported video formats
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    # Emotion categories
    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Shot types
    SHOT_TYPES = ["extreme_close", "close", "medium", "wide", "extreme_wide"]
    
    # Camera movements
    CAMERA_MOVEMENTS = ["static", "pan_left", "pan_right", "tilt_up", "tilt_down", 
                       "zoom_in", "zoom_out", "dolly_forward", "dolly_backward", "handheld"]
    
    # Color grading signatures
    COLOR_GRADING_SIGNATURES = {
        "Teal_Orange": {
            "description": "Hollywood blockbuster style",
            "characteristics": {"warm_highlights": True, "cool_shadows": True}
        },
        "Bleach_Bypass": {
            "description": "Gritty, desaturated look",
            "characteristics": {"low_saturation": True, "high_contrast": True}
        },
        "Day_for_Night": {
            "description": "Blue-shifted, underexposed",
            "characteristics": {"blue_tint": True, "underexposed": True}
        },
        "Matrix_Green": {
            "description": "Digital dystopian look",
            "characteristics": {"green_tint": True, "high_contrast": True}
        },
        "Mexico_Filter": {
            "description": "Yellow-orange desert look",
            "characteristics": {"yellow_tint": True, "high_saturation": True}
        },
        "Wes_Anderson": {
            "description": "Pastel symmetric palette",
            "characteristics": {"pastel_colors": True, "high_saturation": True}
        },
        "Fincher": {
            "description": "Dark, desaturated with crushed shadows",
            "characteristics": {"low_saturation": True, "crushed_blacks": True}
        }
    }
    
    # Cinematography prompts for LLMs
    DIRECTOR_PROMPTS = {
        "composition": "Analyze the visual composition: rule of thirds, leading lines, symmetry, framing, depth",
        "lighting": "Describe lighting setup: key light direction, fill ratio, shadows, contrast, mood, practical vs artificial",
        "camera": "Identify camera technique: angle (high/low/dutch), lens choice (wide/normal/telephoto), movement style",
        "color": "Analyze color grading: palette, temperature (warm/cool), saturation levels, color contrast",
        "mise_en_scene": "Describe mise-en-scène: set design, props, costumes, staging, visual storytelling elements",
        "emotion": "How do the visual elements contribute to the emotional tone of the scene?",
        "style": "What directorial style or influence is evident? Reference similar films or directors."
    }
    
    # Film stock emulations
    FILM_STOCKS = {
        "Kodak_Vision3_250D": {"grain": 0.3, "saturation": 0.8, "contrast": 1.1},
        "Kodak_Vision3_500T": {"grain": 0.5, "saturation": 0.7, "contrast": 1.0},
        "Fuji_Eterna_250D": {"grain": 0.2, "saturation": 0.9, "contrast": 0.9},
        "Kodak_2393": {"grain": 0.6, "saturation": 0.6, "contrast": 1.3},
        "Cinestill_800T": {"grain": 0.7, "saturation": 0.8, "contrast": 1.1, "halation": True}
    }
    
    # Output formats
    OUTPUT_FORMATS = {
        "json": "Structured JSON with all analysis data",
        "markdown": "Human-readable markdown report",
        "csv": "Timeline data for further analysis",
        "video": "Annotated video with analysis overlay"
    }
    
    # Processing settings
    MAX_CONCURRENT_FRAMES = 10  # For async processing
    BATCH_SIZE = 32  # For ML model inference
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = OUTPUT_DIR / "video_director.log"