from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="happenings",
    version="0.0.1",
    author="Tim Nonet",
    author_email="tim.nonet@gmail.com",
    description="A utility package for creating complex event based time-series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TNonet/happenings",
    project_urls={
        "Bug Tracker": "https://github.com/TNonet/happenings/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="happenings"),
    python_requires=">=3.7",
    extras_require={"test": [
        'hypothesis',
        'pytest',
    ]
    },
    install_requires=[
        'numpy>=1.19.0',
        'scipy',
        'pandas>=0.25',
        'python-dateutil'
    ]
)
