#!/usr/bin/env python3
"""
Setup script for golden_params package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read version from __init__.py
def get_version():
    version_file = os.path.join("golden_params", "__init__.py")
    if not os.path.exists(version_file):
        return "0.1.0"

    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                # Extract version string, handling both single and double quotes
                version_line = line.strip()
                if "=" in version_line:
                    version_str = version_line.split("=", 1)[1].strip()
                    # Remove quotes
                    version_str = version_str.strip('\'"')
                    return version_str
    return "0.1.0"

setup(
    name="golden_params",
    version=get_version(),
    description="Golden Parameters Package for neural network sparsity and parameter importance analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/golden_params",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "examples": [
            "datasets>=2.0.0",
            "accelerate>=0.12.0",
        ],
    },
    keywords=[
        "neural networks",
        "sparsity",
        "parameter importance",
        "machine learning",
        "deep learning",
        "model compression",
        "golden parameters",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/golden_params/issues",
        "Source": "https://github.com/yourusername/golden_params",
        "Documentation": "https://golden-params.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)