#! /usr/bin/python
"""Transforms the samples.py script to a notebook or reST page"""

import argparse
import inspect
import json
import os
import sys
import textwrap

import black


def create_header_cells(script_name):
    """Creates the header cells for the notebook"""
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
    ]
    return cells


def create_sample_cells(sample_method):
    """Creates a code cell and an execution cell for the specified method"""

    # Create the code block
    code, docstring = split_docstring(inspect.getsource(sample_method))
    code = textwrap.dedent(code)
    code = black.format_str(code, mode=black.Mode())

    # Create the cell source as a list of lines
    code_list = [line + "\n" for line in code.rstrip().split("\n")]
    code_list[-1] = code_list[-1].rstrip()

    sample_execution_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"### `{sample_method.__name__}()`\n\n", f"{docstring}\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_list,
        },
    ]

    return sample_execution_cells


def create_rest_page_header(script_name):
    subtitle = "The code snippets on this page demonstrate the basic use of the "
    if script_name == "samples":
        title = "Samples core"
        subtitle += ":py:mod:`khiops.core` module."
    else:
        title = "Samples sklearn"
        subtitle += ":py:mod:`khiops.sklearn <khiops.sklearn.estimators>` module."
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
        "\n"
        "Samples\n"
        "-------\n"
    )


def split_docstring(source):
    docstring_open_quote = source.find('"""')
    if docstring_open_quote == -1:
        source_without_docstring = sample_source
        docstring = ""
    else:
        docstring_close_quote = (
            docstring_open_quote + 3 + source[docstring_open_quote + 3 :].find('"""')
        )
        source_without_docstring = source[docstring_close_quote + 4 :]
        docstring = source[docstring_open_quote + 3 : docstring_close_quote]
    return source_without_docstring, docstring


def create_rest_page_section(sample_function):
    code, _ = split_docstring(inspect.getsource(sample_function))
    code = textwrap.dedent(code)
    code = black.format_str(code, mode=black.Mode())
    code = textwrap.indent(code, "    ")
    code = code.rstrip()
    return (
        f".. autofunction:: {sample_function.__name__}\n"
        ".. code-block:: python\n"
        "\n"
        f"{code}"
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
    print(f"Converting to format '{args.format}' samples script at {script_path}")
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
            notebook_objects["cells"].extend(create_sample_cells(sample_method))
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
