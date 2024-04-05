# khiops-python Coding Guidelines

## Quick Start
```bash
# Make sure you are a member of the pyKhiops GitHub project and that you have set up the SSH keys
# for Git

# Clone the repo
git clone ssh://github.com/khiopsml/khiops-python

# Set up the commit message template
git config commit.template .git_commit_template

# Install pre-commit and the pre-commit scripts
pip install -U pre-commit
pre-commit install

# To run all tests
python -m unittest

# To run all test suites in a particular test file, for instance:
python -m unittest tests/test_core.py

# To run a particular test suite, for instance:
python -m unittest tests.test_sklearn.KhiopsSklearnParameterPassingTests
```

## Coding Style

### Language
`khiops-python` is coded in English (comments included).

### Style Enforcement: Tools & Git Hooks
The coding style of pyKhiops is almost PEP8 compliant. To enforce these guidelines we use three
tools:
- [Black](https://github.com/psf/black)
- [isort](https://pycqa.github.io/isort/)
- [Pylint](https://www.pylint.org/)

We strongly recommend to automate the execution of these tools with [pre-commit] which setups Git
hooks that run before each `git commit`. To create the hook scripts in your local repository copy
execute at the root of the repo:
```bash
pip install pre-commit
pre-commit install
```
The `pre-commit` configuration is located at [./.pre-commit-config.yaml](./.pre-commit-config.yaml).
If either `black` or `isort` fail the commit will be aborted and you must execute a `git add` on the
files corrected by these tools and then re-commit. As for `pylint`, it is configured with a score
threshold so that only code with serious errors will abort the commit.

[pre-commit]: https://pre-commit.com

### Manually executing the tools
It is useful to manually run Black as the layout of the formatted code sometimes suggests
improvements. Note that Black does not solve all issues (eg. PEP8 variable names, docstring line
length), so you should check the issues reported by `pylint` and address all *errors* (code E).

You can address other less serious issues reported by a full run of Pylint only if you have the
time. The important point **is not to be a slave** of the linter.

#### A Note on Line Length

Black will shorten the lines to 88 chars, **except for long literal strings**. Since we enforce the
line-length with `pylint` you must fix this by hand or it will pass neither the pre-commit nor the
CI/CD workflows. To quickly find the lines to fix execute:

```
pylint --disable=all --enable=line-too-long
```

### Paragraph-Oriented Programming Style
We encourage a programming style that increases readability at the cost of some verbosity. The idea
is to create paragraphs of code, which consist in a header and body:
- The body is the code to be executed.
- The header is a comment describing what the code does at a high level of abstraction.

Each paragraph is separated by a single empty line, except when it starts an indented code snippet
(notably `if` expressions) or when it follows a docstring that has the same indentation level.

Most of the code should go into paragraphs; exceptions are:
- return statements
- loop variable assignments (usually on `while` loops)
- very short and obvious methods: the docstring suffices as header

Additionally, the number of paragraphs should be kept as small as possible, because technically
**commenting every line** conforms to this style.

The advantages of this approach are:
- It allows to quickly skim through the code while understanding the big picture
- It forces to code coherent blocks of code
- It forces to understand what you are coding

and the disadvantages:
- Comments need maintenance, especially when refactoring
- Sometimes single lines of obvious code must be commented

#### Example
```python
def value_count(values):
    """Prints the counts of each unique value in an array"""
    # Initialize the counts dictionary
    counts = {}

    # Count the unique occurrences in values
    for value in values:
        # If the value count exists: update it
        if value in counts:
            counts[value] += 1

        # Otherwise create a new count of 1
        else:
           counts[value] = 1

    # Print the counts
    for value, count in counts.items():
      print(f"{value}: {count}")
```
Note that the first paragraph could be fused into the second. These kinds of decisions are rather
subjective and code review may help to resolve them.

### Writing Documentation
See the documentation practices and tools [here](./doc/README.md).

## Git & GitHub
### Main Guiding Principles
> The commit history must be the cleanest possible

> The `dev` branch must pass all tests

### Commits
#### Commit Message
The commit message title should answer the question *What do these changes do?*
Respect the following format when writing it:
- use English in imperative form (eg. "Add new feature", "Fix old bug")
- start with upper-case
- respect the git commit message length if possible
- put no periods at the end and avoid any other punctuation
- do not use Markdown

The optional commit description should answer the questions *How are these changes made?* and *Why
are these changes made?*. Use preferably bullet points to write this part. The description can
eventually answer more thoroughly *What do these changes do?* but try create commits that are
"atomic" enough so that their title suffices to answer this question.

#### Rewriting History
Rewriting commit history is only allowed in feature branches and `dev`. Rewriting the history in
`dev` is very limited: it should be only done to serious problems in the history. It is forbidden to
rebase before the last merge commit.

A usual application of history rewriting is to eliminate commits named "Fix previous commit". In
this case use `rebase` to rewrite the history as follows:
```bash
git rebase -i <REF-TO-FIX> # Usually REF-TO-FIX is either "develop", "HEAD~2" or a specific hash
```
to interactively move and squash fix commits with the `fixup` or `squash` operator. Usually this
operation will make your feature branch diverge from `origin` (GitHub repo), so a `git push --force`
is necessary to update it.

More generally, rewriting allows to obtain a commit history that is the cleanest possible. See [Git
Rewriting History](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) for more details.

**Warning:** Do not rebase on a commit before the branching point of your feature branch as it may
screw up your local repository by erasing merge commits.

### Branches and Development Workflow
We use feature branches that branch/merge from/to `dev`. The `main` branch is only used for final
releases.

The rules for the branches are:
- Feature branches: You can commit and rewrite the branch at will. It should pass the CI/CD workflows
  before merging to `dev`. Note that only short tests are run in feature branches.
- `dev` branch: You should avoid directly commit and rewrite this branch (and only maintainers
  can do it). This branch may not succeed all the CI/CD workflows since short and long tests are run.
  You may release beta versions from this branch.
- `main` branch: Only merges from `dev` are accepted. All CI/CD workflows must succeed (this
  includes the long tests).

As for the use of feature branches, GitHub makes it easy to implement the following workflow:
- Create an issue for the particular feature/fix/etc.
- Create a branch for the issue by clicking the "Create a branch" link in the issue's right column.
- Develop, commit changes locally. Don't hesitate to rewrite the branch history if it is clearer.
  This will create a feature branch from `dev` with a name starting with the issue number followed
  by the issue title.
- Once you have some commits (`WIP`s or otherwise), you may push your branch and create a pull
  request (PR) for the issue.
- pull `dev` and rebase on it if there are new changes (see [below](#rebasing-on-develop))
- Push your feature branch to GitHub (if necessary, use `push --force` to rewrite the feature branch
  history) and ensure that it passes the CI/CD jobs.
- Ask the PR be reviewed by another team member:
  - The reviewer creates comment threads with the issues.
  - Discuss the issues and/or make fixes to your code to address them; push the new code to GitHub.
  - The reviewer closes the thread once the issue is settled.
- Merge once all review threads are closed and the reviewer LGTM-ed the PR.

#### Branch Naming
Use the automatic branch names that GitHub creates from an issue. Name issues succinctly so the
branches have short names. If you name branches by yourself try informative and succinct.

### Pull Requests
This project's GitHub repository is configured to create merge commits for PR and to remove the
feature branches by default after merging. This strategy has the following pros and cons:

_Pros:_
- The merges of feature branches are tracked in the history.
- The repository branch set is small and clean with no *stale* ones.

_Cons:_
- Extra commits in history.
- Non-linear `dev` and `main` history.

### Rebasing on `dev`
It may happen that the `dev` branch was updated while you are developing your feature branch.
To avoid extra merge commits do the following to update your local copy of your feature branch:
```bash
# on my-feat-branch
git stash           # only when you have non-committed changes
git switch dev
git pull
git switch my-feat-branch
git rebase dev
# fix any conflicts you may have
git stash pop       # only when you have non-committed changes
```

## Dependencies
### Package dependencies
We should strive to minimize external package dependencies to minimize installation problems. The
current dependency policy is:
- `khiops.core` should only depend on python built-in modules.
- `khiops.sklearn` should only depend on python built-in modules and the following mainstream
data-science packages:
  - [Scikit-learn](https://scikit-learn.org/stable/)
  - [Pandas](https://pandas.pydata.org/)

Note that these 4 packages already have a sizable number of dependencies. We should discuss
thoroughly the pros and cons of any new external package dependency before adding it.

### Development/Build dependencies
For development dependencies (eg. `black`, `isort`, `sphinx`, `wrapt`, `furo`) we can be more
carefree while still trying to not add too many dependencies.

## Versioning
We follow a non-standard `MAJOR.MINOR.PATCH.INCREMENT[PRE_RELEASE]` versioning convention. The
first three numbers `MAJOR.MINOR.PATCH` are the latest Khiops version that is compatible with the
package. The number `INCREMENT` indicates the evolution of `khiops-python` followed by an optional
`[PRE_RELEASE]` version for alpha, beta and release candidate releases (eg. `b2`).

## Releases

## Pre-releases
When tagging a revision the CI will create the packages and upload them to the `khiops-dev` channel.
Prefer to augment the pre-release revision number to re-create a tag because the CI overwrites
packages with the same tag in the `khiops-dev` channel. Do not forget to clean any temporary
pre-releases from `khiops-dev` and the releases GitHub page.


## Public Releases
Checklist:
- Release issue and its related PR
  - Update the API Docs if necessary
  - Update `CHANGELOG.md`
  - Update the default `khiops-core` version in [.github/workflows/conda.yml]
- Git manipulations
  - Update your local repo and save your work:
    - `git stash # if necessary`
    - `git fetch --tags --prune --prune-tags`
    - `git switch dev`
    - `git pull`
    - `git switch main`
    - `git pull`
  - Merge the `dev` branch into `main`
    - `git switch main`
    - `git merge dev`
  - Tag the merge commit with the release version (see Versioning above)
    - `git switch main`
    - `git tag 10.3.0.1 # Just an example`
  - Make `dev` point to the merge commit just created in `main`
    - This is necessary to include the merge commit into master to calculate intermediary versions
      with Versioneer.
    - Steps:
      - `git switch dev`
      - `git reset --hard main`
      - `git push dev` (you need to remove the protections of `dev` for this step)
- Workflows
  - Execute the `Conda Package` workflow specifying:
    - The release tag
    - `khiops` as the release channel
  - Execute the `API Docs` workflow specifying "Deploy GH Pages".

To make a public release, you must execute the `Conda Package` CI workflow manually on a tag and
specify the `khiops` anaconda channel for upload. These uploads do not overwrite any packages in
this channel, so you must correct any mistake manually.

### Git Manipulations upon a Major Release
The following is the check list to be done upon a major release:
