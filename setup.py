from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deeg",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.5.4",
        "mne>=0.22.0",
        "pathlib>=1.0.1",
        "torch>=1.7.0"
    ],
    author="Jiaqi Li, Zhihao Zhao, Yiming Li, Cong Zhang, Chen Chen",
    author_email="lijiaqi199609@sina.com",
    description="package for EEG data analysis and deep learning",
    license="MIT",
    url="https://github.com/LiJiaqi96/DEEG",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
