from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR / "requirements" / "common.txt") as f:
    common_required = f.read().splitlines()

with open(THISDIR / "requirements" / "dev.txt") as f:
    dev_required = f.read().splitlines()

with open(THISDIR / "README.md", encoding="utf-8") as f:
    long_description = f.read()

main_ns = {}
with open(THISDIR / "gcsl" / "__version__.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="gcsl",
    author="Christoph Heindl",
    description="Implementation of goal-conditioned supervised learning (GCSL).",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=main_ns["__version__"],
    packages=find_packages(".", include="gcsl*"),
    install_requires=common_required,
    zip_safe=False,
    extras_require={
        "dev": dev_required,
    },
)