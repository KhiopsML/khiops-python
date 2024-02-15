######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Python package builder and installer driver
"""

from setuptools import find_packages, setup

import versioneer

if __name__ == "__main__":
    setup(
        name="khiops",
        version=versioneer.get_version(),
        url="https://khiops.org",
        description="Python library for the Khiops AutoML suite",
        maintainer="The Khiops Team",
        maintainer_email="khiops.team@orange.com",
        license_files=["LICENSE.md"],
        entry_points={
            "console_scripts": [
                "kh-status=khiops.tools:kh_status_entry_point",
                "kh-samples=khiops.tools:kh_samples_entry_point",
                "kh-download-datasets=khiops.tools:kh_download_datasets_entry_point",
                "pk-status=khiops.tools:pk_status_entry_point",  # deprecated
            ]
        },
        packages=find_packages(
            include=["khiops", "khiops.*", "pykhiops", "pykhiops.*"]
        ),
        include_package_data=True,
        python_requires=">=3.8",
        install_requires=[
            "pandas>=0.25.3",
            "scikit-learn>=0.22.2",
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
        cmdclass=versioneer.get_cmdclass(),
        extras_require={
            "s3": ["boto3>=1.17.39"],
            "gcs": ["google-cloud-storage>=1.37.0"],
        },
    )
