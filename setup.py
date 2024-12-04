from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MISA_XGModel",  # Replace with your project name
    version="0.2.0",  # Semantic versioning (Major.Minor.Patch)
    author="Mateo Cardona Serrano",  # Replace with your name
    author_email="mcardonaserrano@berkeley.edu",  # Replace with your email
    description="A Python library for predicting electron density using XGBoost",  # Short description
    long_description=long_description,  # Use the content of README.md
    long_description_content_type="text/markdown",  # Specify the content type
    url="https://github.com/mcardonaserrano/MISA_XGModel",  # Replace with your GitHub URL
    packages=find_packages(),  # Automatically find all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version required
    install_requires=[
        "xgboost",
        "numpy",
        "xarray",
        "scikit-learn",
        "netcdf4",
        "pandas"
    ],
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)