from setuptools import find_packages, setup


setup(
    name="poker44-bot-detector",
    version="1.0.0",
    description="LightGBM baseline miner for Poker44 Subnet 126 bot detection.",
    packages=find_packages(include=["src", "src.*", "miners", "miners.*", "neurons", "neurons.*"]),
    install_requires=[
        "requests>=2.31.0",
        "PyYAML>=6.0.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.5.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.9",
)
