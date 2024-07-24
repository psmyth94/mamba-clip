from setuptools import find_packages, setup

setup(
    name="isic",
    version="0.0.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="International Skin Imaging Collaboration (ISIC) 2024 Challenge",
    author_email="psmyth1994@gmail.com",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.8.0,<3.12.0",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
