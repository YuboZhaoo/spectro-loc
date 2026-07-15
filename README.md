## Overview

This repository contains the code for the paper "SpectroLoc: Cryptographic
Operation Localization via Spectrogram Projection and Time-Series Analysis".

## Dataset

The dataset required to run the code can be downloaded from:
https://drive.google.com/file/d/17iAaZP0QoogScAILAO1w7NdIjDXl4Ocl/

Extract the downloaded file and place the `dataset/` directory in the project
root before running the experiments.

## Requirements

### Required Software

Python 3.9 is recommended. The artifact was tested with Python 3.9.23.

### Python Packages

Required packages:

- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- claspy >= 0.1.1
- jupyter >= 1.0.0
- pandas >= 1.3.0

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The core implementation is packaged through `pyproject.toml`. For local
development, reuse from another notebook, or running the notebooks in this
artifact, install the repository in editable mode after installing the
dependencies:

```bash
pip install -e .
```

The notebooks import the reusable code as `spectroloc.*`, so this editable
install step is required unless `src/` is otherwise added to `PYTHONPATH`.

## Usage

The repository includes four primary notebooks corresponding to the main
experimental results of the paper:

- `1.grid_search.ipynb`: runs the analyst-guided seeding mode on software and
  hardware AES traces under different projection settings. It corresponds to
  the projection robustness and parameter grid-search study in Section 4.2.

- `2.auto_search.ipynb`: runs the automatic repetitive-CO localization
  pipeline on AES traces, including the main localization experiments,
  additive-noise robustness evaluation, and target-length sensitivity analysis.
  It corresponds to Section 4.3.

- `3.segment.ipynb`: performs unsupervised change-point segmentation on ECDSA
  traces, evaluates the detected boundaries against trigger-derived ground
  truth, and visualizes the segmentation results. It corresponds to
  Section 4.4.1.

- `4.disint.ipynb`: performs motif-based localization on mixed AES/SHA traces,
  including interleaved-CO localization, projection-window sensitivity, and
  additive-noise robustness evaluation. It corresponds to Section 4.4.2.

The notebooks keep dataset paths and paper-specific experiment setup. The
`spectroloc` package contains the reusable core methods:

- `spectroloc.projection`: time-domain and STFT projection helpers.
- `spectroloc.self_temp_analyst`: analyst-guided projection, template extraction, and
  single-combination evaluation helpers.
- `spectroloc.self_temp_auto`: automatic repetitive-operation localization.
- `spectroloc.cpd`: change point segmentation,
  evaluation, and plotting helpers.
- `spectroloc.motif`: motif localization, evaluation, and plotting
  helpers for mixed traces.
- `spectroloc.config`: named dataset configuration classes used by the notebooks.

## Directory Structure

```text
spectro-loc/
|-- 1.grid_search.ipynb
|-- 2.auto_search.ipynb
|-- 3.segment.ipynb
|-- 4.disint.ipynb
|-- pyproject.toml
|-- requirements.txt
|-- dataset/
|   |-- chameleon_lowpass/
|   |-- em/
|   |-- semi-loc/
|   `-- trace-copilot/
`-- src/
    `-- spectroloc/
        |-- __init__.py
        |-- config.py
        |-- cpd.py
        |-- motif.py
        |-- projection.py
        |-- self_temp_analyst.py
        `-- self_temp_auto.py
```

The `dataset/` directory is not included in the GitHub repository and should be
added manually after downloading and extracting the dataset.

## License

This project is released under the [MIT License](LICENSE).
