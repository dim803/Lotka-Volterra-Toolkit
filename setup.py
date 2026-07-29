from setuptools import setup, find_packages

setup(
    name="lotka-volterra-toolkit",
    version="0.1.0",
    description="A general-purpose toolkit for fitting, simulating, "
                 "and evaluating Lotka-Volterra predator-prey models "
                 "against real time series data.",
    packages=find_packages(exclude=["examples", "examples.*"]),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "pandas>=1.3",
        "statsmodels>=0.13",
        "matplotlib>=3.4",
    ],
    python_requires=">=3.8",
)
