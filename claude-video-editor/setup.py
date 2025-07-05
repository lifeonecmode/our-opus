#!/usr/bin/env python3
"""
Claude Video Editor - AI-powered video editing with Claude Code integration
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="claude-video-editor",
    version="1.0.0",
    description="AI-powered video editing with Claude Code integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/claude-video-editor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Non-Linear Editor",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.3.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "ffmpeg-python>=0.2.0",
        "asyncio",
        "aiofiles",
        "watchdog>=2.0.0",
        "pydantic>=1.8.0",
        "rich>=10.0.0",
        "typer>=0.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "jsonlines>=2.0.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
            "mkdocs-mermaid2-plugin>=0.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "claude-video-editor=claude_video_editor.cli:main",
            "claude-edit=claude_video_editor.quick_start:main",
            "claude-watch=claude_video_editor.watch:main",
        ],
    },
    include_package_data=True,
    package_data={
        "claude_video_editor": [
            "assets/*",
            "templates/*",
            "examples/*",
        ],
    },
    zip_safe=False,
    keywords="video editing, AI, Claude, automation, content creation, FFmpeg",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/claude-video-editor/issues",
        "Source": "https://github.com/yourusername/claude-video-editor",
        "Documentation": "https://claude-video-editor.readthedocs.io/",
    },
)