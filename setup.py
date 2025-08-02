from setuptools import setup, find_packages

setup(
    name="brain-agent",
    version="0.1.0",
    description="Neural-inspired knowledge retrieval system built on heaven-base",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "heaven-framework>=0.1.0",
    ],
    author="HEAVEN Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)