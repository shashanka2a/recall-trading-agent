"""
Setup script for Recall Trading Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="recall-trading-agent",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A sophisticated AI-powered trading agent for Recall Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/recall-trading-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "plotly>=5.15.0",
            "seaborn>=0.12.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "scipy>=1.11.0",
            "ta-lib>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "recall-agent=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.json", "*.yaml"],
        "docs": ["*.md"],
    },
)