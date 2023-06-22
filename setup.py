from setuptools import find_packages, setup

import versioneer

setup(
    name="pykhiops",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url="https://www.khiops.com",
    description="Python API for the Khiops AutoML suite",
    maintainer="Felipe Olmos",
    maintainer_email="92923444+felipe-olmos-orange@users.noreply.github.com",
    license_files=["LICENSE.md"],
    entry_points={"console_scripts": ["convert-pk10=pykhiops.tools:convert_pk10_main"]},
    packages=find_packages(exclude=["tests"]),
    package_data={
        "pykhiops.core": [
            "scenarios/**/*._kh*",
            "doc/samples/samples*.py",
            "doc/samples/samples*.ipynb",
        ]
    },
    install_requires=[
        "pandas>=0.25.0",
        "scikit-learn>=0.21",
    ],
    extras_require={"s3": ["boto3>=1.17.39"], "gcs": ["google-cloud-storage>=1.37.0"]},
)
