from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="spd_trading",
    version="0.1.3",
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
    python_requires=">=3.9",
    install_requires=["numpy==1.21.0", 
                      "pandas==1.3.0", 
                      "scipy==1.7.0", 
                      "arch==4.19", 
                      "localpoly==0.1.4", 
                      "sklearn==0.0", 
                      "matplotlib==3.4.2", 
                      "statsmodels==0.12.2", 
                      "scipy==1.7.0"],
)
