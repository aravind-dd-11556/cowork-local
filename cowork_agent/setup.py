"""Setup script for cowork_agent package."""

from setuptools import setup, find_packages

setup(
    name="cowork-agent",
    version="0.1.0",
    description="A Cowork-like AI agent framework with configurable LLM providers",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.25"],
        "web": [
            "trafilatura>=1.6",
            "markdownify>=0.11",
            "beautifulsoup4>=4.12",
            "lxml>=4.9",
        ],
        "all": [
            "openai>=1.0",
            "anthropic>=0.25",
            "trafilatura>=1.6",
            "markdownify>=0.11",
            "beautifulsoup4>=4.12",
            "lxml>=4.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "cowork-agent=cowork_agent.main:main",
        ],
    },
    package_data={
        "cowork_agent": ["config/default_config.yaml"],
    },
)
