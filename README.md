# Digital Communications Notebooks

This repository contains notes for teaching the fundamentals of modern Digital Communication Systems.

## Quick Start

### Use these notebooks online without having to install anything
Use Google Colab by following the link below. You will need a Google account to use Colab and notebooks will be stored on your Google Drive.

* <a href="https://colab.research.google.com/github/bepepa/digital_comms/blob/main/index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Just look at some notebooks, without executing any code

* <a href="https://nbviewer.jupyter.org/github/bepepa/digital_comms/blob/main/index.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a>

* [github.com's notebook viewer](https://github.com/bepepa/digital_comms/blob/main/index.ipynb) also works but it's not ideal: it's slower, the math equations are not always displayed correctly, and large notebooks sometimes fail to open.

### Install this project on your own machine

Start by installing [Anaconda](https://www.anaconda.com/distribution/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), and [git](https://git-scm.com/downloads).

Next, clone this project by opening a terminal and typing the following commands (do not type the first `$` signs on each line, they just indicate that these are terminal commands):

```shell
    git clone https://github.com/bepepa/digital_comms.git
    cd digital_comms
```

Next, run the following commands:

```shell
    conda env create -f environment.yml
    python -m ipykernel install --user --name=python3
```

Finally, start Jupyter:

```shell
    jupyter notebook
```

## FAQ