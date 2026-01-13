from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memobrain",
    version="0.1.0",
    author="Tommy Chien",
    description="A reasoning graph-based memory system for LLM agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qhjqhj00/MemoBrain/memobrain",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ],
    package_dir={"memobrain": "src"},
    packages=["memobrain"],
)

