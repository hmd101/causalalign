# causalalign/setup.py
from setuptools import find_packages, setup

setup(
    name="causalalign",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        # Core
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        # Plotting
        "matplotlib",
        "seaborn",
        "tueplots",
        # Testing
        "pytest",
        # Jupyter
        "jupyter",
        "openai",
        "google-generativeai",
        "anthropic",
    ],
)
