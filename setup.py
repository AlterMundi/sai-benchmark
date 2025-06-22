#!/usr/bin/env python3
"""
Setup script for SAI-Benchmark framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="sai-benchmark",
    version="0.1.0",
    description="Unified multi-dimensional vision assessment framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAI-Benchmark Team",
    author_email="",
    url="https://github.com/your-org/sai-benchmark",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "datasets",
        "ollama", 
        "Pillow",
        "pandas",
        "scikit-learn",
        "tqdm",
        "requests",
        "transformers>=4.37.0",
        "torch>=2.0.0",
        "accelerate",
        "qwen-vl-utils",
        "pyyaml",
        "numpy",
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "pytest-asyncio>=0.21.0",
            "pytest-timeout>=2.1.0",
            "pytest-xdist>=3.3.0",
            "hypothesis>=6.82.0",
            "faker>=19.2.0",
        ],
        "dev": [
            "black",
            "ruff",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "sai-benchmark=evaluate:main",
            "sai-suite=run_suite:main",
            "sai-matrix=run_matrix:main",
        ],
    },
)