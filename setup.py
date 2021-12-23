import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="i2mc",
    version="0.0.1",
    description="Library for performing the I2MC algorithm (Noise-robust fixation classification).",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dcnieho/I2MC_Python",
    author="",
    author_email="",
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[""],
)