#!/usr/bin/env python3
"""
Analyze All Footage - Comprehensive analysis of all videos in project
Scans footage directory and generates detailed analysis for each video
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import logging
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from instagram_frame_analyzer import InstagramFrameAnalyzer
from nuclear_frame_extractor import NuclearFrameExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectAnalyzer:
    """Analyzes all footage in a project directory"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.footage_dir = self.project_dir / "footage"
        self.input_dir = self.project_dir / "input"
        self.analysis_dir = self.project_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
    async def analyze_all_footage(self, use_nuclear: bool = False) -> Dict[str, Any]:
        """Analyze all video files in the project with optional nuclear extraction"""
        logger.info("🔍 Analyzing all footage in project...")
        if use_nuclear:
            logger.info("🚀 Nuclear frame extraction enabled!")
        
        all_analysis = {
            "project": str(self.project_dir),
            "footage": {},
            "summary": {
                "total_videos": 0,
                "total_duration": 0,
                "categories": {},
                "nuclear_extractions": 0
            }
        }
        
        # Analyze main speaker
        main_speaker = self.footage_dir / "main-speaker.mp4"
        if main_speaker.exists():
            logger.info(f"Analyzing main speaker: {main_speaker}")
            analysis = await self._analyze_video(main_speaker, "main_speaker", use_nuclear=use_nuclear)
            all_analysis["footage"]["main_speaker"] = analysis
            all_analysis["summary"]["total_videos"] += 1
            all_analysis["summary"]["total_duration"] += analysis["metadata"]["duration"]
            if analysis.get("nuclear_extraction", {}).get("success"):
                all_analysis["summary"]["nuclear_extractions"] += 1
        
        # Analyze bottle hero shot
        bottle_hero = self.footage_dir / "bottle-hero.mp4"
        if bottle_hero.exists():
            logger.info(f"Analyzing bottle hero: {bottle_hero}")
            analysis = await self._analyze_video(bottle_hero, "bottle_hero", use_nuclear=use_nuclear)
            all_analysis["footage"]["bottle_hero"] = analysis
            all_analysis["summary"]["total_videos"] += 1
            all_analysis["summary"]["total_duration"] += analysis["metadata"]["duration"]
            if analysis.get("nuclear_extraction", {}).get("success"):
                all_analysis["summary"]["nuclear_extractions"] += 1
        
        # Analyze transformation videos
        transform_dir = self.footage_dir / "transformation"
        if transform_dir.exists():
            all_analysis["footage"]["transformation"] = {}
            
            for video in transform_dir.glob("*.mp4"):
                category = "before" if "before" in video.name else "after"
                logger.info(f"Analyzing transformation {category}: {video}")
                
                analysis = await self._analyze_video(video, f"transformation_{category}", use_nuclear=use_nuclear)
                all_analysis["footage"]["transformation"][category] = analysis
                all_analysis["summary"]["total_videos"] += 1
                all_analysis["summary"]["total_duration"] += analysis["metadata"]["duration"]
                if analysis.get("nuclear_extraction", {}).get("success"):
                    all_analysis["summary"]["nuclear_extractions"] += 1
        
        # Analyze B-roll
        b_roll_dir = self.footage_dir / "b-roll"
        if b_roll_dir.exists():
            all_analysis["footage"]["b_roll"] = {}
            
            for category_dir in b_roll_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    all_analysis["footage"]["b_roll"][category] = []
                    
                    for video in category_dir.glob("*.mp4"):
                        logger.info(f"Analyzing B-roll {category}: {video}")
                        
                        analysis = await self._analyze_video(
                            video, 
                            f"b_roll_{category}_{video.stem}",
                            use_nuclear=use_nuclear
                        )
                        all_analysis["footage"]["b_roll"][category].append({
                            "filename": video.name,
                            "analysis": analysis
                        })
                        all_analysis["summary"]["total_videos"] += 1
                        all_analysis["summary"]["total_duration"] += analysis["metadata"]["duration"]
                        if analysis.get("nuclear_extraction", {}).get("success"):
                            all_analysis["summary"]["nuclear_extractions"] += 1
        
        # Generate insights
        all_analysis["insights"] = self._generate_insights(all_analysis)
        
        # Save complete analysis
        output_path = self.analysis_dir / "complete_footage_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(all_analysis, f, indent=2)
        
        logger.info(f"✅ Complete analysis saved to: {output_path}")
        
        # Generate summary report
        self._generate_summary_report(all_analysis)
        
        return all_analysis
    
    async def _analyze_video(self, video_path: Path, output_name: str, use_nuclear: bool = False) -> Dict[str, Any]:
        """Analyze a single video with optional nuclear frame extraction"""
        output_dir = self.analysis_dir / output_name
        
        if use_nuclear:
            # Use Nuclear Frame Extractor for frame-perfect extraction
            logger.info(f"Using Nuclear Frame Extraction for {video_path.name}")
            nuclear_dir = output_dir / "nuclear_frames"
            
            try:
                extractor = NuclearFrameExtractor(str(video_path), str(nuclear_dir))
                nuclear_result = extractor.extract(cleanup=True)
                
                # Load the nuclear extraction data
                if nuclear_result['success']:
                    with open(nuclear_result['outputs']['json'], 'r') as f:
                        nuclear_data = json.load(f)
                    
                    # Enhance with standard analysis
                    analyzer = InstagramFrameAnalyzer(str(video_path), str(output_dir))
                    analyzer.analyze_video(interval=0.5, detect_scenes=True)
                    analyzer.save_outputs()
                    
                    # Merge nuclear and standard analysis
                    analysis_files = list(output_dir.glob("analysis_*.json"))
                    if analysis_files:
                        with open(analysis_files[0], 'r') as f:
                            standard_analysis = json.load(f)
                        
                        # Combine analyses
                        return {
                            **standard_analysis,
                            "nuclear_extraction": {
                                "success": True,
                                "frames_extracted": nuclear_result['frames_extracted'],
                                "frame_paths": nuclear_data['frames'],
                                "validation": nuclear_data['validation']
                            }
                        }
                
            except Exception as e:
                logger.error(f"Nuclear extraction failed: {e}. Falling back to standard analysis.")
        
        # Standard analysis
        analyzer = InstagramFrameAnalyzer(str(video_path), str(output_dir))
        
        # Analyze with appropriate interval based on video type
        if "main" in output_name or "speaker" in output_name:
            interval = 0.5  # More detailed for main content
        elif "b_roll" in output_name:
            interval = 1.0  # Less detailed for B-roll
        else:
            interval = 0.5
        
        analyzer.analyze_video(interval=interval, detect_scenes=True)
        analyzer.save_outputs()
        
        # Load the analysis
        analysis_files = list(output_dir.glob("analysis_*.json"))
        if analysis_files:
            with open(analysis_files[0], 'r') as f:
                analysis_data = json.load(f)
                if use_nuclear:
                    analysis_data["nuclear_extraction"] = {"success": False, "reason": "Fallback to standard"}
                return analysis_data
        else:
            return {"error": "Analysis failed"}
    
    def _generate_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from all footage analysis"""
        insights = {
            "content_availability": {},
            "emotional_peaks": [],
            "quality_scores": {},
            "recommendations": []
        }
        
        # Check content availability
        insights["content_availability"] = {
            "has_main_speaker": "main_speaker" in analysis["footage"],
            "has_transformation": "transformation" in analysis["footage"],
            "has_b_roll": "b_roll" in analysis["footage"],
            "has_product_shots": "bottle_hero" in analysis["footage"]
        }
        
        # Find emotional peaks across all footage
        all_peaks = []
        
        def extract_peaks(footage_data, source):
            if isinstance(footage_data, dict) and "frame_data" in footage_data:
                frames = footage_data.get("frame_data", [])
                for frame in frames:
                    emotion = frame.get("emotional_tone", {})
                    if emotion.get("confidence", 0) > 0.8:
                        all_peaks.append({
                            "source": source,
                            "timestamp": frame["timestamp"],
                            "emotion": emotion.get("dominant"),
                            "confidence": emotion.get("confidence")
                        })
        
        # Extract from main speaker
        if "main_speaker" in analysis["footage"]:
            extract_peaks(analysis["footage"]["main_speaker"], "main_speaker")
        
        # Sort by confidence
        all_peaks.sort(key=lambda x: x["confidence"], reverse=True)
        insights["emotional_peaks"] = all_peaks[:10]  # Top 10 peaks
        
        # Calculate average quality scores
        quality_scores = []
        
        def extract_quality(footage_data):
            if isinstance(footage_data, dict) and "summary" in footage_data:
                score = footage_data["summary"].get("cinematography", {}).get("avg_score", 0)
                quality_scores.append(score)
        
        for key, value in analysis["footage"].items():
            if isinstance(value, dict):
                extract_quality(value)
        
        if quality_scores:
            insights["quality_scores"] = {
                "average": sum(quality_scores) / len(quality_scores),
                "highest": max(quality_scores),
                "lowest": min(quality_scores)
            }
        
        # Generate recommendations
        if insights["content_availability"]["has_transformation"]:
            insights["recommendations"].append(
                "Use transformation shots at video midpoint for maximum impact"
            )
        
        if len(insights["emotional_peaks"]) > 5:
            insights["recommendations"].append(
                "Multiple emotional peaks detected - space them evenly for consistent engagement"
            )
        
        if insights["quality_scores"].get("average", 0) < 0.7:
            insights["recommendations"].append(
                "Consider color grading to improve visual quality"
            )
        
        return insights
    
    def _generate_summary_report(self, analysis: Dict[str, Any]):
        """Generate human-readable summary report"""
        report_path = self.analysis_dir / "footage_analysis_report.md"
        
        report = f"""# Footage Analysis Report

## Project Summary
- **Total Videos**: {analysis['summary']['total_videos']}
- **Total Duration**: {analysis['summary']['total_duration']:.1f} seconds
- **Project Path**: {analysis['project']}

## Available Content

### Main Speaker
"""
        
        if "main_speaker" in analysis["footage"]:
            main = analysis["footage"]["main_speaker"]
            report += f"""✅ Available
- Duration: {main['metadata']['duration']:.1f}s
- Resolution: {main['metadata']['width']}x{main['metadata']['height']}
- Quality Score: {main.get('summary', {}).get('cinematography', {}).get('avg_score', 0):.0%}
"""
        else:
            report += "❌ Not found\n"
        
        report += "\n### Transformation Shots\n"
        if "transformation" in analysis["footage"]:
            trans = analysis["footage"]["transformation"]
            if "before" in trans:
                report += f"✅ Before shot available ({trans['before']['metadata']['duration']:.1f}s)\n"
            if "after" in trans:
                report += f"✅ After shot available ({trans['after']['metadata']['duration']:.1f}s)\n"
        else:
            report += "❌ Not found\n"
        
        report += "\n### B-Roll Content\n"
        if "b_roll" in analysis["footage"]:
            for category, videos in analysis["footage"]["b_roll"].items():
                report += f"- **{category}**: {len(videos)} videos\n"
        else:
            report += "❌ No B-roll found\n"
        
        # Add insights
        insights = analysis.get("insights", {})
        
        report += "\n## Key Insights\n\n"
        
        report += "### Emotional Peaks\n"
        for i, peak in enumerate(insights.get("emotional_peaks", [])[:5], 1):
            report += f"{i}. {peak['source']} at {peak['timestamp']:.1f}s - "
            report += f"{peak['emotion']} ({peak['confidence']:.0%})\n"
        
        report += "\n### Quality Analysis\n"
        quality = insights.get("quality_scores", {})
        if quality:
            report += f"- Average Quality: {quality.get('average', 0):.0%}\n"
            report += f"- Highest: {quality.get('highest', 0):.0%}\n"
            report += f"- Lowest: {quality.get('lowest', 0):.0%}\n"
        
        report += "\n### Recommendations\n"
        for rec in insights.get("recommendations", []):
            report += f"- {rec}\n"
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Summary report saved to: {report_path}")
    
    async def analyze_audio_library(self) -> Dict[str, Any]:
        """Analyze audio files in the project"""
        logger.info("🎵 Analyzing audio library...")
        
        audio_dir = self.project_dir / "audio"
        audio_analysis = {
            "voiceover": None,
            "music": None,
            "sound_effects": []
        }
        
        if not audio_dir.exists():
            logger.warning("Audio directory not found")
            return audio_analysis
        
        # Analyze voiceover
        voiceover = audio_dir / "voiceover.wav"
        if voiceover.exists():
            audio_analysis["voiceover"] = {
                "path": str(voiceover),
                "format": "wav",
                "quality": "48kHz, -12dB normalized",
                "duration": self._get_audio_duration(voiceover)
            }
        
        # Analyze music
        music = audio_dir / "music.mp3"
        if music.exists():
            audio_analysis["music"] = {
                "path": str(music),
                "format": "mp3",
                "bpm": 120,
                "purpose": "2Hz resonance",
                "duration": self._get_audio_duration(music)
            }
        
        # Analyze sound effects
        sfx_dir = audio_dir / "sfx"
        if sfx_dir.exists():
            for sfx in sfx_dir.glob("*.wav"):
                audio_analysis["sound_effects"].append({
                    "name": sfx.stem,
                    "path": str(sfx),
                    "type": self._categorize_sfx(sfx.stem),
                    "suggested_use": self._suggest_sfx_timing(sfx.stem)
                })
        
        # Save audio analysis
        audio_output = self.analysis_dir / "audio_library_analysis.json"
        with open(audio_output, 'w') as f:
            json.dump(audio_analysis, f, indent=2)
        
        logger.info(f"✅ Audio analysis saved to: {audio_output}")
        
        return audio_analysis
    
    def _get_audio_duration(self, path: Path) -> float:
        """Get audio file duration"""
        import subprocess
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _categorize_sfx(self, name: str) -> str:
        """Categorize sound effect"""
        name_lower = name.lower()
        if "boom" in name_lower:
            return "impact"
        elif "swoosh" in name_lower:
            return "transition"
        elif "ding" in name_lower:
            return "success"
        elif "whoosh" in name_lower:
            return "movement"
        else:
            return "accent"
    
    def _suggest_sfx_timing(self, name: str) -> List[str]:
        """Suggest when to use sound effect"""
        suggestions = {
            "boom": ["transformation_reveal", "climax_moment", "cta"],
            "swoosh": ["scene_transition", "text_appear", "wipe"],
            "ding": ["benefit_appear", "success_moment", "completion"]
        }
        return suggestions.get(name, ["accent_moment"])
    
    async def analyze_input_directory(self, use_nuclear: bool = False) -> Dict[str, Any]:
        """Analyze all videos in the input directory with optional nuclear extraction"""
        logger.info("📥 Analyzing videos in input directory...")
        if use_nuclear:
            logger.info("🚀 Nuclear frame extraction enabled for input videos!")
        
        input_analysis = {
            "input_videos": [],
            "total_videos": 0,
            "total_duration": 0,
            "nuclear_extractions": 0,
            "categorized": {
                "speaker": [],
                "product": [],
                "lifestyle": [],
                "unknown": []
            }
        }
        
        if not self.input_dir.exists():
            logger.warning("Input directory not found. Creating it...")
            self.input_dir.mkdir(exist_ok=True)
            return input_analysis
        
        # Process all video files in input directory
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(ext))
        
        for video_file in video_files:
            logger.info(f"Analyzing input video: {video_file.name}")
            
            # Analyze video
            analysis = await self._analyze_video(
                video_file, 
                f"input_{video_file.stem}",
                use_nuclear=use_nuclear
            )
            
            # Categorize video based on content
            category = self._categorize_input_video(analysis, video_file.name)
            
            video_info = {
                "filename": video_file.name,
                "path": str(video_file),
                "category": category,
                "analysis": analysis,
                "suggested_use": self._suggest_video_use(category, analysis)
            }
            
            input_analysis["input_videos"].append(video_info)
            input_analysis["categorized"][category].append(video_file.name)
            input_analysis["total_videos"] += 1
            input_analysis["total_duration"] += analysis.get("metadata", {}).get("duration", 0)
            if analysis.get("nuclear_extraction", {}).get("success"):
                input_analysis["nuclear_extractions"] += 1
        
        # Save input analysis
        output_path = self.analysis_dir / "input_videos_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(input_analysis, f, indent=2)
        
        logger.info(f"✅ Input directory analysis saved to: {output_path}")
        
        # Generate input report
        self._generate_input_report(input_analysis)
        
        return input_analysis
    
    def _categorize_input_video(self, analysis: Dict[str, Any], filename: str) -> str:
        """Categorize a video based on its content and filename"""
        filename_lower = filename.lower()
        
        # Check filename hints
        if any(word in filename_lower for word in ["speaker", "talk", "person", "face"]):
            return "speaker"
        elif any(word in filename_lower for word in ["product", "bottle", "item", "hero"]):
            return "product"
        elif any(word in filename_lower for word in ["lifestyle", "scene", "broll", "b-roll"]):
            return "lifestyle"
        
        # Check content analysis
        if "frame_data" in analysis:
            # Look for faces to identify speaker videos
            has_faces = any(
                frame.get("people_detected", 0) > 0 
                for frame in analysis.get("frame_data", [])
            )
            if has_faces:
                return "speaker"
        
        # Check shot types
        if "summary" in analysis:
            shot_types = analysis["summary"].get("shot_distribution", {})
            if shot_types.get("CU", 0) > 0.5:  # Mostly close-ups
                return "product"
            elif shot_types.get("WS", 0) > 0.3:  # Wide shots
                return "lifestyle"
        
        return "unknown"
    
    def _suggest_video_use(self, category: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest how to use a video based on its category and analysis"""
        suggestions = {
            "speaker": {
                "primary_use": "main_narrative",
                "recommended_position": "base_layer",
                "editing_notes": "Use as primary footage with cutaways"
            },
            "product": {
                "primary_use": "hero_shots",
                "recommended_position": "emphasis_moments",
                "editing_notes": "Use for product reveals and features"
            },
            "lifestyle": {
                "primary_use": "b_roll",
                "recommended_position": "supporting_content",
                "editing_notes": "Use to illustrate benefits and context"
            },
            "unknown": {
                "primary_use": "review_needed",
                "recommended_position": "manual_review",
                "editing_notes": "Review content to determine best use"
            }
        }
        
        suggestion = suggestions[category].copy()
        
        # Add quality-based recommendations
        if "summary" in analysis:
            quality = analysis["summary"].get("cinematography", {}).get("avg_score", 0)
            if quality > 0.8:
                suggestion["quality_note"] = "High quality - suitable for hero moments"
            elif quality < 0.5:
                suggestion["quality_note"] = "Lower quality - consider as quick cuts only"
        
        return suggestion
    
    def _generate_input_report(self, analysis: Dict[str, Any]):
        """Generate a report for input directory analysis"""
        report_path = self.analysis_dir / "input_videos_report.md"
        
        report = f"""# Input Directory Analysis Report

## Summary
- **Total Videos Found**: {analysis['total_videos']}
- **Total Duration**: {analysis['total_duration']:.1f} seconds
- **Categories**:
  - Speaker Videos: {len(analysis['categorized']['speaker'])}
  - Product Videos: {len(analysis['categorized']['product'])}
  - Lifestyle Videos: {len(analysis['categorized']['lifestyle'])}
  - Unknown/Review Needed: {len(analysis['categorized']['unknown'])}

## Video Details

"""
        
        for video in analysis['input_videos']:
            report += f"### {video['filename']}\n"
            report += f"- **Category**: {video['category']}\n"
            report += f"- **Duration**: {video['analysis'].get('metadata', {}).get('duration', 0):.1f}s\n"
            report += f"- **Suggested Use**: {video['suggested_use']['primary_use']}\n"
            
            if 'quality_note' in video['suggested_use']:
                report += f"- **Quality**: {video['suggested_use']['quality_note']}\n"
            
            report += f"- **Notes**: {video['suggested_use']['editing_notes']}\n\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        
        if len(analysis['categorized']['speaker']) > 0:
            report += "- ✅ Speaker footage available for main narrative\n"
        else:
            report += "- ⚠️ No speaker footage detected - consider adding talking head content\n"
        
        if len(analysis['categorized']['product']) > 0:
            report += "- ✅ Product shots available for hero moments\n"
        else:
            report += "- ⚠️ No product footage detected - consider adding product shots\n"
        
        if len(analysis['categorized']['lifestyle']) > 0:
            report += "- ✅ B-roll footage available for context\n"
        
        if len(analysis['categorized']['unknown']) > 0:
            report += f"- ⚠️ {len(analysis['categorized']['unknown'])} videos need manual review\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Input videos report saved to: {report_path}")


async def main():
    """Main function to analyze project"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all footage in project")
    parser.add_argument("project", nargs="?", default=".",
                       help="Project directory path")
    parser.add_argument("--audio", action="store_true",
                       help="Also analyze audio library")
    parser.add_argument("--input", action="store_true",
                       help="Analyze videos in input directory")
    parser.add_argument("--nuclear", action="store_true",
                       help="Use nuclear frame extraction for 1800 frames @ 30fps")
    
    args = parser.parse_args()
    
    analyzer = ProjectAnalyzer(args.project)
    
    # Analyze input directory if requested
    if args.input:
        print("📥 Analyzing input directory...")
        if args.nuclear:
            print("🚀 Nuclear frame extraction enabled!")
        input_analysis = await analyzer.analyze_input_directory(use_nuclear=args.nuclear)
        
        print(f"\n✅ Analyzed {input_analysis['total_videos']} videos in input directory")
        print(f"📊 Total duration: {input_analysis['total_duration']:.1f} seconds")
        if args.nuclear:
            print(f"🎯 Nuclear extractions: {input_analysis['nuclear_extractions']} videos")
        print(f"📁 Categories:")
        for category, videos in input_analysis['categorized'].items():
            if videos:
                print(f"  - {category}: {len(videos)} videos")
    else:
        # Analyze standard footage directory
        print("🎬 Starting comprehensive footage analysis...")
        if args.nuclear:
            print("🚀 Nuclear frame extraction enabled!")
        footage_analysis = await analyzer.analyze_all_footage(use_nuclear=args.nuclear)
        
        print(f"\n✅ Analyzed {footage_analysis['summary']['total_videos']} videos")
        print(f"📊 Total duration: {footage_analysis['summary']['total_duration']:.1f} seconds")
        if args.nuclear:
            print(f"🎯 Nuclear extractions: {footage_analysis['summary']['nuclear_extractions']} videos")
        
        # Analyze audio if requested
        if args.audio:
            print("\n🎵 Analyzing audio library...")
            audio_analysis = await analyzer.analyze_audio_library()
            print(f"✅ Found {len(audio_analysis['sound_effects'])} sound effects")
    
    print("\n📁 Analysis complete! Check the 'analysis' directory for:")
    if args.input:
        print("  - input_videos_analysis.json")
        print("  - input_videos_report.md")
    else:
        print("  - complete_footage_analysis.json")
        print("  - footage_analysis_report.md")
        if args.audio:
            print("  - audio_library_analysis.json")
    print("  - Individual video analyses in subdirectories")


if __name__ == "__main__":
    asyncio.run(main())