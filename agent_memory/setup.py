from setuptools import setup, find_packages

setup(
    name="agent-memory",
    version="1.0.0",
    description="Standalone AI Agent Memory Module — short/mid/long-term + cross-session",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],   # zero external dependencies
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.20.0",
        ],
        "embeddings": [
            "sentence-transformers>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
