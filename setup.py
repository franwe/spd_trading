from setuptools import setup, find_packages

setup(
    name="spd_trading",
    version="0.0.1",
    url="https://github.com/franwe/spd_trading",
    author="Franziska Wehrmann",
    author_email="franziska.wehrmann@gmail.com",
    description="Description of my package",
    packages=find_packages(),
    install_requires=["numpy >= 1.11.1"],
)
