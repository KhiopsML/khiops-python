# !/usr/bin/env bash
set -e

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

echo "Obtaining pykhiops-tutorial"
pykhiops_tutorial_repo="git@github.com:KhiopsML/khiops-python-tutorial"
pykhiops_tutorial_repo_branch="main"
git archive --prefix=pykhiops-tutorial/ --format=tar \
   --remote="$pykhiops_tutorial_repo" "$pykhiops_tutorial_repo_branch" |\
  tar -xf -

echo "Creating coursework"
tutorials_dir="./tutorials"
mkdir -p tutorials_dir
tutorials_dir="$(realpath $tutorials_dir)"
cd pykhiops-tutorial
zip "$tutorials_dir/core_tutorials_solutions.zip" "Core Basics*.ipynb" helper_functions.py data/*/*
zip "$tutorials_dir/sklearn_tutorials_solutions.zip" "Sklearn Basics*.ipynb" helper_functions.py data/*/*
python create-coursework.py
cd coursework
zip "$tutorials_dir/core_tutorials.zip" "Core Basics*.ipynb" helper_functions.py data/*/*
zip "$tutorials_dir/sklearn_tutorials.zip" "Sklearn Basics*.ipynb" helper_functions.py data/*/*
cd ../..

echo "Creating reST tutorial pages"
python convert_tutorials.py -g pykhiops-tutorial

echo "Executing Sphinx"
make html
