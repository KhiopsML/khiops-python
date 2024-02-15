#! /usr/bin/python
"""Transforms the samples.py script to a notebook or reST page"""

import argparse
import inspect
import json
import os
import sys
import textwrap


def create_boilerplate_code(script_name):
    if script_name == "samples":
        boilerplate_code = [
            "import os\n",
            "from math import sqrt\n",
            "from os import path\n",
            "\n",
            "from khiops import core as kh\n",
            "\n",
        ]
    elif script_name == "samples_sklearn":
        boilerplate_code = [
            "import os\n",
            "import pickle\n",
            "from os import path\n",
            "\n",
            "import pandas as pd\n",
            "from sklearn import metrics\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.experimental import enable_hist_gradient_boosting\n",
            "from sklearn.ensemble import HistGradientBoostingClassifier\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.preprocessing import OneHotEncoder\n",
            "\n",
            "from khiops import core as kh\n",
            "from khiops.sklearn import (\n",
            "    KhiopsClassifier,\n",
            "    KhiopsCoclustering,\n",
            "    KhiopsEncoder,\n",
            "    KhiopsRegressor,\n",
            ")\n",
        ]
    else:
        raise ValueError(f"Invalid samples script name '{script_name}'")
    return boilerplate_code


def create_header_cells(script_name):
    """Creates the header cells for the notebook"""
    boilerplate_code = create_boilerplate_code(script_name)

    # Create the boilerplate cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Khiops Python {script_name}\n",
                f"This is a notebook containing the code in the `{script_name}.py` script\n"
                "of the Khiops Python library.\n\n"
                "Make sure you have already installed the latest version of ",
                "[Khiops](https://khiops.org) before using this this notebook",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"collapsed": True},
            "outputs": [],
            "source": boilerplate_code,
        },
    ]
    return cells


def create_sample_cell(sample_method):
    """Creates a code cell and an execution cell for the specified method"""

    # Create the cell source as a list of lines
    sample_method_source = inspect.getsource(sample_method)
    sample_source_list = [line + "\n" for line in sample_method_source.split("\n")]
    sample_source_list += ["#Run sample\n", sample_method.__name__ + "()"]

    sample_execution_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sample_source_list,
    }

    return sample_execution_cell


def create_rest_page_header(script_name):
    boilerplate_code = "".join(create_boilerplate_code(script_name))
    indented_boilerplate_code = textwrap.indent(boilerplate_code, "    ")
    subtitle = "The code snippets on this page demonstrate the basic use of the "
    if script_name == "samples":
        title = "Samples core"
        subtitle += ":py:mod:`khiops.core` module."
    else:
        title = "Samples sklearn"
        subtitle += ":py:mod:`khiops.sklearn` module."
    return (
        ":orphan:\n"
        "\n"
        f".. currentmodule:: {script_name}\n"
        "\n"
        f"{title}\n"
        f"{'=' * len(title)}\n"
        "\n"
        f"{subtitle}\n"
        "\n"
        "Script and Jupyter notebook\n"
        "---------------------------\n"
        "The samples in this page are also available as:\n"
        "\n"
        f"- :download:`Python script <../../khiops/samples/{script_name}.py>`\n"
        f"- :download:`Jupyter notebook <../../khiops/samples/{script_name}.ipynb>`\n"
        "\n"
        "Setup\n"
        "-----\n"
        "First make sure you have installed the sample datasets. In a configured\n"
        "conda shell (ex. *Anaconda Prompt* in Windows) execute:\n"
        "\n"
        ".. code-block:: shell\n"
        "\n"
        "    kh-download-datasets\n"
        "\n"
        "If that doesn't work open a python console and execute:\n"
        "\n"
        ".. code-block:: python\n"
        "\n"
        "    from khiops.tools import download_datasets\n"
        "    download_datasets()\n"
        "\n"
        "Before copying any code snippet make sure to precede it with following\n"
        "preamble:\n"
        "\n"
        ".. code-block:: python\n"
        "\n"
        f"{indented_boilerplate_code}"
        "\n"
        "Samples\n"
        "-------\n"
    )


def remove_docstring(source):
    docstring_open = source.find('"""')
    if docstring_open == -1:
        source_without_docstring = sample_source
    else:
        docstring_close = source[docstring_open + 3 :].find('"""')
        source_without_docstring = source[docstring_open + 3 + docstring_close + 4 :]
    return source_without_docstring


def create_rest_page_section(sample_function):
    code = f"def {sample_function.__name__}():\n" + remove_docstring(
        inspect.getsource(sample_function)
    )
    indented_code = textwrap.indent(code, "    ")
    return (
        f".. autofunction:: {sample_function.__name__}\n"
        ".. code-block:: python\n"
        "\n"
        f"{indented_code}"
    )


def main(args):
    """Main method"""
    # Obtain the script name
    if args.sklearn:
        script_name = "samples_sklearn"
    else:
        script_name = "samples"

    # Sanity check
    script_path = os.path.join(args.samples_dir, f"{script_name}.py")
    if os.path.abspath(script_path) == os.path.abspath(args.output_path):
        print("error: input and output paths are the same")
        sys.exit(1)

    # Add khiops root to the python path
    khiops_root_path = os.path.dirname(
        os.path.dirname(os.path.realpath(args.samples_dir))
    )
    sys.path.append(khiops_root_path)

    # Import samples as a module
    try:
        if args.sklearn:
            from khiops.samples import samples_sklearn as samples
        else:
            from khiops.samples import samples as samples
    except ImportError as error:
        print(f"Could not import samples script {script_name}.py:")
        print(error)
        sys.exit(1)

    # Case of a Jupyter notebook: Create cells and then dump to the file
    if args.format == "ipynb":
        notebook_objects = {}
        notebook_objects["cells"] = create_header_cells(script_name)
        for sample_method in samples.exported_samples:
            notebook_objects["cells"].append(create_sample_cell(sample_method))
        notebook_objects["metadata"] = {}
        notebook_objects["nbformat"] = 4
        notebook_objects["nbformat_minor"] = 2

        with open(args.output_path, "w") as notebook:
            json.dump(notebook_objects, notebook, indent=1)
    # Case of a reST page: Print the header and sections to the file
    else:
        with open(args.output_path, "w") as rest_page:
            print(create_rest_page_header(script_name), file=rest_page)
            for sample_method in samples.exported_samples:
                print(create_rest_page_section(sample_method), file=rest_page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python convert_samples.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Transforms the samples.py script to a notebook or reST page",
    )
    parser.add_argument("samples_dir", metavar="PYFILE", help="samples scripts dir")
    parser.add_argument("output_path", metavar="OUTFILE", help="output file")
    parser.add_argument("--sklearn", action="store_true", default=False)
    parser.add_argument(
        "-f", "--format", type=str, choices=["ipynb", "rst"], default="ipynb"
    )
    main(parser.parse_args())
