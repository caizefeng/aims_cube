import codecs
import os

import setuptools


# Helper Functions

def get_required_packages():
    """Retrieve the list of required packages from requirements.txt."""
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


def read(rel_path):
    """Read the content of a file at a relative path."""
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    """Extract the version string from a Python file."""
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# Setup Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='aims_cube',
    version=get_version("aims_cube/__init__.py"),
    description='Cube file toolkit for FHI-aims',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Zefeng Cai',
    url='https://github.com/caizefeng/aims_cube',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=setuptools.find_packages(),
    install_requires=get_required_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'cube2parchg=aims_cube.cli.cube2parchg_cli:main'
        ],
    },
)