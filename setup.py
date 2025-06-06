from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="model_editing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
)