#!/usr/bin/env bash
set -euo pipefail

# Add the khiops directory to the Python path
if [[ -z "${PYTHONPATH+x}" ]]
then
  export PYTHONPATH=".."
else
  export PYTHONPATH="$PYTHONPATH:.."
fi

# Check command existence
command_requirements="tar git make python zip"
for command_name in $command_requirements
do
  if ! command -v "$command_name" &> /dev/null
  then
    echo "Command '$command_name' not found in path, aborting"
    exit 1
  fi
done

echo "Obtaining khiops-python-tutorial"
khiops_python_tutorial_repo="git@github.com:KhiopsML/khiops-python-tutorial.git"
khiops_python_tutorial_repo_branch="main"
git clone --depth 1 --branch="$khiops_python_tutorial_repo_branch" \
    "$khiops_python_tutorial_repo" \
    && rm -rf ./khiops-python-tutorial/.git

echo "Creating coursework"
tutorials_dir="./tutorials"
mkdir -p tutorials_dir
tutorials_dir="$(realpath $tutorials_dir)"
cd khiops-python-tutorial
zip "$tutorials_dir/core_tutorials_solutions.zip" "Core Basics*.ipynb" helper_functions.py data/*/*
zip "$tutorials_dir/sklearn_tutorials_solutions.zip" "Sklearn Basics*.ipynb" helper_functions.py data/*/*
python create-coursework.py
cd coursework
zip "$tutorials_dir/core_tutorials.zip" "Core Basics*.ipynb" helper_functions.py data/*/*
zip "$tutorials_dir/sklearn_tutorials.zip" "Sklearn Basics*.ipynb" helper_functions.py data/*/*
cd ../..

echo "Creating reST tutorial pages"
python convert_tutorials.py -g khiops-python-tutorial

echo "Executing Sphinx"
make html
