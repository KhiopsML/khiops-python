######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Converts the Jupyter notebooks of the Khiops Python tutorial to Markdown"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbconvert import MarkdownExporter, NotebookExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from nbformat import notebooknode as nbnode


def main(args):
    # Check tutorial directory
    if not os.path.isdir(args.tutorial_dir):
        print(f"Invalid tutorials directory: {args.tutorial_dir}")
        sys.exit(1)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(args.output_dir)

    # Store the repository root for the kernel's sys.path (must be computed
    # before the chdir)
    repository_root = str(Path(__file__).resolve().parents[2])

    # Save and change the current directory to that of the notebooks
    initial_working_dir = os.getcwd()
    os.chdir(args.tutorial_dir)

    # Collect the notebooks filenames and paths
    notebook_paths = sorted(glob.glob("*.ipynb"))
    notebook_names = [
        os.path.splitext(os.path.basename(path))[0] for path in notebook_paths
    ]

    # Execute each notebook and convert it to Markdown if specified
    if args.execute_notebooks:
        # Set up one kernel for all executions
        kernel_manager = KernelManager(kernel_name="python3")

        # Do not print errors
        kernel_manager.start_kernel(stderr=subprocess.DEVNULL)
        preprocessor = ExecutePreprocessor(km=kernel_manager)

        for notebook_path, notebook_name in zip(notebook_paths, notebook_names):
            print(f"Processing {notebook_path}")
            with open(notebook_path, encoding="utf8") as notebook_file:
                notebook = nbformat.read(notebook_file, 4)

                # Add setup cell (sys.path + disable HTML dataframes)
                setup_source = (
                    "import sys\n"
                    f'sys.path.append("{repository_root}")\n'
                    "import pandas as pd\n"
                    'pd.set_option("display.notebook_repr_html", False)\n'
                )
                setup_cell = nbnode.from_dict(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": setup_source,
                    }
                )
                notebook.cells.insert(0, setup_cell)

                # Execute the notebook once
                try:
                    preprocessor.preprocess(notebook, {}, km=kernel_manager)
                except CellExecutionError:
                    print(f"WARNING: '{notebook_path}' had execution" f" errors")

                # Remove the setup cell from the executed notebook
                notebook.cells.pop(0)

                # Export as Markdown (with text-only dataframe outputs)
                md_path = os.path.join(abs_output_dir, f"{notebook_name}.md")
                print(f"Writing file {md_path}")
                with open(md_path, "w", encoding="utf8") as output_file:
                    body, _ = MarkdownExporter().from_notebook_node(notebook)
                    output_file.write(body)

                # Export as .ipynb (same executed outputs)
                nb_path = os.path.join(abs_output_dir, f"{notebook_name}.ipynb")
                print(f"Writing file {nb_path}")
                with open(nb_path, "w", encoding="utf8") as output_file:
                    body, _ = NotebookExporter().from_notebook_node(notebook)
                    output_file.write(body)

        kernel_manager.shutdown_kernel(now=True)

    # Restore the initial current directory
    os.chdir(initial_working_dir)

    # Define the message creator local function
    def _tutorials_message(module_name):
        return (
            f"These [Jupyter notebook tutorials]({module_name}_tutorials.zip) "
            f"cover the basic usage of the `{module_name}` Khiops sub-module. The "
            "solution notebooks are "
            f"[available here]({module_name}_tutorials_solutions.zip) "
            "or you can browse them in this page:\n\n"
        )

    # Write the tutorial page
    sklearn_tutorials = [name for name in notebook_names if name.startswith("Sklearn")]
    core_tutorials = [name for name in notebook_names if name.startswith("Core")]
    tutorials_file_path = os.path.join(abs_output_dir, "index.md")
    with open(tutorials_file_path, "w", encoding="utf8") as tutorials_file:
        tutorials_file.write("# Tutorials\n")
        tutorials_file.write("\n")
        tutorials_file.write("## Sklearn\n")
        tutorials_file.write("\n")
        tutorials_file.write(_tutorials_message("sklearn"))
        for name in sklearn_tutorials:
            tutorials_file.write(f"- [{name}]({name}.md)\n")
        tutorials_file.write("\n")
        tutorials_file.write("## Core\n")
        tutorials_file.write("\n")
        tutorials_file.write(_tutorials_message("core"))
        for name in core_tutorials:
            tutorials_file.write(f"- [{name}]({name}.md)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python convert_tutorial.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Converts the tutorial notebooks to a Markdown page",
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
