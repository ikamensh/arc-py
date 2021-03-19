import re
from pathlib import Path

from setuptools import setup

_DIR = Path(__file__).parent


with (_DIR / "arc" / "version.py").open() as f:
    version = re.search('__version__ = "(.*?)"', f.read()).group(1)


def get_requirements():
    with (_DIR / "requirements.txt").open() as f:
        return f.read()


setup(
    name="arc-py",
    packages=["arc"],
    package_dir={"arc": "arc"},
    version=version,
    description="Utilities for working with ARC Challenge.",
    author="Ilya Kamenshchikov",
    keywords=["utility", "ARC"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    license="MIT",
    long_description=(_DIR / "README.md").read_text().strip(),
    long_description_content_type="text/markdown",
    install_requires=get_requirements(),
    python_requires=">=3.6",
    url="https://github.com/ikamensh/arc-py",
    include_package_data=True,
)
