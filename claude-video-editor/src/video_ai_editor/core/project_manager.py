#!/usr/bin/env python3
"""
Project Manager - Handles viral video production project structure
Creates and manages the organized file system for each project
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ViralProjectManager:
    """
    Manages the project structure for viral video production
    Implements the directory structure specified by the user
    """
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Project structure template
        self.project_structure = {
            "footage": {
                "transformation": ["before", "after"],
                "b-roll": ["lifestyle_shots", "product_details"]
            },
            "graphics": {
                "screenshots": [],
                "overlays": [],
                "animations": []
            },
            "audio": {
                "sfx": []
            },
            "cache": {
                "preview_renders": [],
                "gpu_cache": [],
                "neural_analysis": []
            },
            "exports": {
                "platform-specific": []
            }
        }
        
        # Asset templates for viral content
        self.asset_templates = {
            "overlays": [
                "yellow_highlight.aep",
                "arrow_bounce.aep", 
                "countdown_timer.aep"
            ],
            "animations": [
                "boom_explosion.aep",
                "text_shake.aep"
            ],
            "sfx": [
                "boom.wav",
                "swoosh.wav",
                "ding.wav"
            ]
        }
    
    def create_project(self, video_path: str, project_name: Optional[str] = None) -> Path:
        """
        Create a new project with complete directory structure
        
        Args:
            video_path: Path to main video file
            project_name: Optional project name (auto-generated if not provided)
            
        Returns:
            Path to project root
        """
        if not project_name:
            project_name = f"viral_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        project_root = self.workspace_root / project_name
        
        logger.info(f"Creating project: {project_name}")
        
        # Create directory structure
        self._create_directories(project_root)
        
        # Copy main video to footage
        video_dest = project_root / "footage" / "main-speaker.mp4"
        shutil.copy2(video_path, video_dest)
        
        # Create project manifest
        manifest = {
            "project_name": project_name,
            "created": datetime.now().isoformat(),
            "source_video": str(video_path),
            "status": "initialized",
            "structure_version": "1.0"
        }
        
        manifest_path = project_root / "project.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Initialize asset templates
        self._initialize_templates(project_root)
        
        logger.info(f"Project created at: {project_root}")
        return project_root
    
    def _create_directories(self, project_root: Path):
        """Create the complete directory structure"""
        
        def create_nested(base: Path, structure: Dict):
            for key, value in structure.items():
                path = base / key
                path.mkdir(parents=True, exist_ok=True)
                
                if isinstance(value, dict):
                    create_nested(path, value)
                elif isinstance(value, list):
                    for subdir in value:
                        (path / subdir).mkdir(parents=True, exist_ok=True)
        
        create_nested(project_root, self.project_structure)
    
    def _initialize_templates(self, project_root: Path):
        """Initialize template files and assets"""
        
        # Create placeholder After Effects templates
        for template in self.asset_templates["overlays"]:
            template_path = project_root / "graphics" / "overlays" / template
            self._create_ae_template(template_path, template.replace('.aep', ''))
        
        for template in self.asset_templates["animations"]:
            template_path = project_root / "graphics" / "animations" / template
            self._create_ae_template(template_path, template.replace('.aep', ''))
        
        # Create placeholder audio files
        for sfx in self.asset_templates["sfx"]:
            sfx_path = project_root / "audio" / "sfx" / sfx
            # Create empty audio file (in real implementation, copy from library)
            sfx_path.touch()
    
    def _create_ae_template(self, path: Path, template_type: str):
        """Create After Effects template file"""
        # In a real implementation, this would create actual .aep files
        # For now, create a JSON descriptor
        descriptor = {
            "template_type": template_type,
            "version": "1.0",
            "settings": self._get_template_settings(template_type)
        }
        
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(descriptor, f, indent=2)
    
    def _get_template_settings(self, template_type: str) -> Dict:
        """Get template-specific settings"""
        settings = {
            "yellow_highlight": {
                "color": [1.0, 0.843, 0.0],  # Trust yellow
                "duration": 2.0,
                "animation": "fade_in_bounce"
            },
            "arrow_bounce": {
                "style": "cta_arrow",
                "bounce_height": 20,
                "frequency": 2
            },
            "countdown_timer": {
                "start_value": 24,
                "end_value": 0,
                "style": "digital",
                "urgency_color": [1.0, 0.0, 0.0]
            },
            "boom_explosion": {
                "particle_count": 500,
                "spread": 360,
                "colors": ["orange", "yellow", "white"]
            },
            "text_shake": {
                "intensity": 10,
                "frequency": 30,
                "decay": 0.8
            }
        }
        
        return settings.get(template_type, {})
    
    def import_assets(self, project_root: Path, assets: Dict[str, List[str]]):
        """
        Import additional assets into project
        
        Args:
            project_root: Project root directory
            assets: Dict mapping asset types to file paths
        """
        logger.info(f"Importing assets to: {project_root}")
        
        for asset_type, file_paths in assets.items():
            if asset_type == "b_roll":
                dest_dir = project_root / "footage" / "b-roll"
            elif asset_type == "screenshots":
                dest_dir = project_root / "graphics" / "screenshots"
            elif asset_type == "transformation":
                dest_dir = project_root / "footage" / "transformation"
            else:
                logger.warning(f"Unknown asset type: {asset_type}")
                continue
            
            for file_path in file_paths:
                src = Path(file_path)
                if src.exists():
                    dest = dest_dir / src.name
                    shutil.copy2(src, dest)
                    logger.info(f"Imported: {src.name} -> {dest_dir}")
    
    def prepare_for_platform(self, project_root: Path, platform: str) -> Dict[str, Any]:
        """
        Prepare project for specific platform export
        
        Args:
            project_root: Project root directory
            platform: Target platform (tiktok, instagram, youtube)
            
        Returns:
            Platform-specific settings
        """
        platform_settings = {
            "tiktok": {
                "resolution": [1080, 1920],
                "fps": 30,
                "max_duration": 60,
                "color_boost": {
                    "saturation": 1.1,
                    "contrast": 1.05
                },
                "audio": {
                    "normalize": -14,  # LUFS
                    "compression": "heavy"
                }
            },
            "instagram": {
                "resolution": [1080, 1920],
                "fps": 30,
                "max_duration": 90,
                "color_boost": {
                    "saturation": 0.95,
                    "contrast": 0.98
                },
                "audio": {
                    "normalize": -14,
                    "compression": "medium"
                }
            },
            "youtube": {
                "resolution": [1080, 1920],
                "fps": 30,
                "max_duration": 60,
                "color_boost": {
                    "saturation": 1.0,
                    "contrast": 1.0
                },
                "audio": {
                    "normalize": -14,
                    "compression": "light"
                }
            }
        }
        
        settings = platform_settings.get(platform, platform_settings["tiktok"])
        
        # Create platform-specific export directory
        export_dir = project_root / "exports" / "platform-specific" / platform
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save platform settings
        settings_path = export_dir / "platform_settings.json"
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        return settings
    
    def optimize_cache(self, project_root: Path):
        """Optimize cache for M4 chip performance"""
        cache_dir = project_root / "cache"
        
        # Create Metal shader cache directory
        metal_cache = cache_dir / "gpu_cache" / "metal_shaders"
        metal_cache.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration for optimal M4 performance
        cache_config = {
            "gpu_acceleration": True,
            "metal_performance_shaders": True,
            "cache_size_mb": 2048,
            "preview_quality": "half",  # Use half precision for previews
            "neural_engine": True,
            "parallel_renders": 4  # M4 can handle multiple streams
        }
        
        config_path = cache_dir / "cache_config.json"
        with open(config_path, 'w') as f:
            json.dump(cache_config, f, indent=2)
    
    def create_render_queue(self, project_root: Path, 
                          edits: List[Dict], 
                          output_settings: Dict) -> Path:
        """
        Create render queue for batch processing
        
        Args:
            project_root: Project root directory
            edits: List of edit decisions
            output_settings: Render settings
            
        Returns:
            Path to render queue file
        """
        queue = {
            "project": str(project_root),
            "created": datetime.now().isoformat(),
            "render_settings": output_settings,
            "compositions": []
        }
        
        # Group edits by type for efficient rendering
        edit_groups = {}
        for edit in edits:
            edit_type = edit.get('type', 'unknown')
            if edit_type not in edit_groups:
                edit_groups[edit_type] = []
            edit_groups[edit_type].append(edit)
        
        # Create render items
        for edit_type, type_edits in edit_groups.items():
            comp_item = {
                "name": f"comp_{edit_type}",
                "edits": type_edits,
                "priority": max(e.get('priority', 1) for e in type_edits)
            }
            queue["compositions"].append(comp_item)
        
        # Sort by priority
        queue["compositions"].sort(key=lambda x: x['priority'], reverse=True)
        
        # Save queue
        queue_path = project_root / "cache" / "render_queue.json"
        with open(queue_path, 'w') as f:
            json.dump(queue, f, indent=2)
        
        return queue_path
    
    def cleanup_project(self, project_root: Path, keep_exports: bool = True):
        """
        Clean up project files
        
        Args:
            project_root: Project root directory
            keep_exports: Whether to keep exported files
        """
        logger.info(f"Cleaning up project: {project_root}")
        
        # Clear cache
        cache_dir = project_root / "cache"
        if cache_dir.exists():
            for cache_subdir in ["preview_renders", "gpu_cache", "neural_analysis"]:
                subdir_path = cache_dir / cache_subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir()
        
        # Remove temporary files
        for temp_file in project_root.rglob("*.tmp"):
            temp_file.unlink()
        
        for ae_lock in project_root.rglob("*.aep.lock"):
            ae_lock.unlink()
        
        # Update manifest
        manifest_path = project_root / "project.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            manifest["cleaned"] = datetime.now().isoformat()
            manifest["status"] = "cleaned"
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        
        logger.info("Project cleanup complete")
    
    def archive_project(self, project_root: Path, archive_dir: Optional[Path] = None) -> Path:
        """
        Archive completed project
        
        Args:
            project_root: Project root directory
            archive_dir: Archive destination (default: workspace/archive)
            
        Returns:
            Path to archive file
        """
        if not archive_dir:
            archive_dir = self.workspace_root / "archive"
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        project_name = project_root.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{project_name}_{timestamp}"
        
        # Create archive
        archive_path = shutil.make_archive(
            str(archive_dir / archive_name),
            'zip',
            project_root.parent,
            project_root.name
        )
        
        logger.info(f"Project archived to: {archive_path}")
        return Path(archive_path)


# Example usage
if __name__ == "__main__":
    # Create project manager
    manager = ViralProjectManager(Path.home() / "viral_video_production")
    
    # Create new project
    project = manager.create_project("/path/to/video.mp4", "test_project")
    
    # Import additional assets
    manager.import_assets(project, {
        "screenshots": ["/path/to/celebrity1.png", "/path/to/celebrity2.png"],
        "b_roll": ["/path/to/lifestyle1.mp4", "/path/to/product1.mp4"]
    })
    
    # Prepare for TikTok
    settings = manager.prepare_for_platform(project, "tiktok")
    print(f"Platform settings: {settings}")
    
    # Optimize cache for M4
    manager.optimize_cache(project)
    
    # Clean up when done
    manager.cleanup_project(project, keep_exports=True)