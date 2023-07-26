from setuptools import setup, find_packages
import setuptools

setup(
    name="rt1plotpy",
    version="0.3.1",
    author='Kenji Ueda',
    author_email='kentokamak@gmail.com',
    url="https://github.com/UedaKenji/rt1_advanced_plot",
    install_requires=["dxfgrabber",'opencv-python'],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
)