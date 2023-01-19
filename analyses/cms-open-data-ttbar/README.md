This notebook (`coffea.ipynb`) is paired to the file `coffea.py` via Jupytext (https://jupytext.readthedocs.io/en/latest/). Using `git diff` with this file instead of the `.ipynb` file is much simpler, as you don't have to deal with notebook metadata or output images. However, in order for the notebook output to be preserved, the notebook still needs to be version controlled. It is ideal to run `git diff` with the option `-- . ':(exclude)*.ipynb'`, so that `.ipynb` files are ignored. The `.py` file can be run as a Python script.

There is a pre-commit hook that will ensure that the files are synced via Jupytext. If you are using this for the first time, you must run
```
pre-commit install
```
in the GitHub repository. After this, you should be able to commit with the Jupytext pre-commit hook! You may have to commit twice, as the pre-commit hook may make changes to the notebook file.
