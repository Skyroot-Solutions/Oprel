"""
Setup script for Oprel SDK
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version
version = {}
version_path = Path(__file__).parent / "oprel" / "version.py"
exec(version_path.read_text(), version)

setup(
    name="oprel",
    version=version["__version__"],
    author=version["__author__"],
    author_email=version["__email__"],
    
    description="Local-first AI runtime - The SQLite of LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ragultv/oprel-SDK",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "huggingface-hub>=0.20.0",
        "psutil>=5.9.0",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "local": ["torch>=2.1.0"],
        "cuda": ["torch>=2.1.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oprel=oprel.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="llm ai local inference privacy gguf llama",
    project_urls={
        "Documentation": "https://docs.oprel.dev",
        "Source": "https://github.com/ragultv/oprel-SDK",
        "Bug Reports": "https://github.com/ragultv/oprel-SDK/issues",
    },
)