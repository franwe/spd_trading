from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="spd_trading",
    version="0.1.0",
    url="https://github.com/franwe/spd_trading",
    project_urls={
        "Documentation": "https://spd_trading.readthedocs.io/en/latest/",
        "Code": "https://github.com/franwe/spd_trading",
    },
    author="franwe",
    author_email="franziska.wehrmann@gmail.com",
    description="Estimates the Risk Neutral Density and Historical Density of an underlying and suggests trading intervals based on the Pricing Kernel.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=["numpy", "pandas", "scipy", "arch", "localpoly", "sklearn", "matplotlib", "statsmodels", "scipy"],
)
