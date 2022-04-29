from setuptools import find_packages, setup

import versioneer

setup(
    name="pykhiops",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url="https://www.khiops.com",
    description="Python library for the Khiops AutoML suite",
    maintainer="Felipe Olmos",
    maintainer_email="92923444+felipe-olmos-orange@users.noreply.github.com",
    license_files=["LICENSE.md"],
    entry_points={
        "console_scripts": [
            "convert-pk10=pykhiops.tools:convert_pk10_entry_point",
            "pk-status=pykhiops.tools:pk_status_entry_point",
        ]
    },
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "pandas>=0.25.0",
        "scikit-learn>=0.21",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    extras_require={
        "s3": ["boto3>=1.17.39"],
        "gcs": ["google-cloud-storage>=1.37.0"],
    },
)
