"""Converts the Jupyter notebooks of the Khiops Python tutorial to reST"""
import argparse
import glob
import os
import sys

import nbformat
from jupyter_client import KernelManager
from nbconvert import NotebookExporter, RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import notebooknode as nbnode


def main(args):
    print(args)
    # Check tutorial directory
    if not os.path.isdir(args.tutorial_dir):
        print(f"Invalid tutorials directory: {args.tutorial_dir}")
        sys.exit(1)

    # Save and change the current directory to that of the notebooks
    initial_working_dir = os.getcwd()
    os.chdir(args.tutorial_dir)

    # Create the reST tutorials directory
    rest_tutorial_dir = os.path.join(initial_working_dir, "tutorials")
    os.makedirs(rest_tutorial_dir, exist_ok=True)

    # Collect the notebooks filenames and paths
    notebook_paths = sorted(glob.glob("*.ipynb"))
    notebook_names = [
        os.path.splitext(os.path.basename(path))[0] for path in notebook_paths
    ]

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(args.output_dir)

    # Execute each notebook and convert it to reST if specified
    if args.execute_notebooks:
        # Set up one kernel for all executions
        kernel_manager = KernelManager(kernel_name="python3")
        kernel_manager.start_kernel()
        preprocessor = ExecutePreprocessor(km=kernel_manager)

        for notebook_path, notebook_name in zip(notebook_paths, notebook_names):
            print(f"Processing {notebook_path}")
            with open(notebook_path, encoding="utf8") as notebook_file:
                notebook = nbformat.read(notebook_file, 4)
                notebook_exporter = NotebookExporter()
                rst_exporter = RSTExporter()
                export_setups = [(rst_exporter, "rst"), (notebook_exporter, "ipynb")]

                for exporter, file_ext in export_setups:
                    # Add a setup cell for rest
                    # - Disables the html output when displaying dataframes
                    # - Adds "../.." to the sys path
                    setup_source = "import sys\n" 'sys.path.append("../..")\n'
                    if file_ext == "rst":
                        setup_source += (
                            "import pandas as pd\n"
                            'pd.set_option("display.notebook_repr_html", False)\n'
                        )

                    setup_cell_dict = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": setup_source,
                    }
                    setup_cell = nbnode.from_dict(setup_cell_dict)
                    notebook.cells.insert(0, setup_cell)

                    # Execute the notebook to obtain the output cells
                    preprocessor.preprocess(notebook, {})

                    # Eliminate the setup cell
                    notebook.cells.pop(0)

                    # Execute the notebook and write output
                    output_file_path = os.path.join(
                        abs_output_dir, f"{notebook_name}.{file_ext}"
                    )
                    print(f"Writing file {output_file_path}")
                    with open(output_file_path, "w", encoding="utf8") as output_file:
                        body, _ = exporter.from_notebook_node(notebook)
                        if file_ext == "rst":
                            output_file.write(":orphan:\n\n")
                        output_file.write(body)
        kernel_manager.shutdown_kernel(now=True)

    # Restore the initial current directory
    os.chdir(initial_working_dir)

    # Define the message creator local function
    def _tutorials_message(module_name):
        return (
            "These "
            f":download:`Jupyter notebook tutorials <{module_name}_tutorials.zip>` "
            f"cover the basic usage of the ``{module_name}`` Khiops sub-module. The "
            "solution notebooks are "
            f":download:`available here <{module_name}_tutorials_solutions.zip>` "
            "or you can browse them in this page:\n\n"
        )

    # Write the tutorial page
    sklearn_tutorials = [name for name in notebook_names if name.startswith("Sklearn")]
    core_tutorials = [name for name in notebook_names if name.startswith("Core")]
    tutorials_file_path = os.path.join(rest_tutorial_dir, "index.rst")
    with open(tutorials_file_path, "w", encoding="utf8") as tutorials_file:
        tutorials_file.write("Tutorials\n")
        tutorials_file.write("=========\n")
        tutorials_file.write("\n")
        tutorials_file.write("Sklearn\n")
        tutorials_file.write("-------\n")
        tutorials_file.write(_tutorials_message("sklearn"))
        for name in sklearn_tutorials:
            tutorials_file.write(f"- :doc:`{name}`\n")
        tutorials_file.write("\n")
        tutorials_file.write("Core\n")
        tutorials_file.write("----\n")
        tutorials_file.write(_tutorials_message("core"))
        for name in core_tutorials:
            tutorials_file.write(f"- :doc:`{name}`\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python convert_tutorial.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Converts the tutorial notebooks to a reST page",
    )
    parser.add_argument(
        "tutorial_dir",
        metavar="DIR",
        help="Location of the khiops-tutorial directory",
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTDIR",
        help="Location of the output directory",
    )
    parser.add_argument(
        "-e",
        "--execute-notebooks",
        action="store_true",
        help="Executes the notebooks (takes time)",
    )
    main(parser.parse_args())
