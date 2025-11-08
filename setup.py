from setuptools import setup, find_packages

setup(
    name="tomo-detect",
    version="0.1.7",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'tomo-detect=tomo_detect.cli:main',
        ],
    },
    author="Rajath Kumar",
    author_email="1rajathkumar@gmail.com",
    description="A CLI tool for detecting flagellar motor coordinates in tomography data",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords="Bacterial Flagellar Motor, tomography, motor detection, deep learning, CNN",
    url="https://github.com/Rajathk6/tomo-detect-cli",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",

    # Include your JSON files
    include_package_data=True,
    package_data={
        "tomo_detect": ["*.json"],
    },
)
