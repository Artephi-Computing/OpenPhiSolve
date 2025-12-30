from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="phisolve",
    version="0.3.0",
    description="OpenPhiSolve: An Open-Source Python library for solving MIQP",
    author="Artephi Team",
    author_email="pickspeng@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
)
