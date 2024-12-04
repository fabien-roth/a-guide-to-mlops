from setuptools import setup, find_packages

setup(
    name="a_guide_to_mlops",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    author="Fabien Roth",
    description="A guide to MLOps for managing ML projects, focusing on quantization and deployment.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fabien-roth/a-guide-to-mlops",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
