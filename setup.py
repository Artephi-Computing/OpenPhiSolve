from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="phisolve",
    version="0.2.0",
    description="TODO",
    author="TODO",
    author_email="TODO",
    packages=find_packages(),
    install_requires=install_requires,
)
