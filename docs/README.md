# TBsim documentation

This folder contains the source for the TBsim documentation site, built with [Quarto](https://quarto.org).

## Tutorials

Tutorials are available as Quarto notebooks (`.qmd`) in the `tutorials` subfolder.

## Building the docs

To build the site locally:

1. Install [Quarto](https://quarto.org/docs/get-started/) (version 1.5+).

2. Install TBsim and the documentation dependencies:
   ```bash
   pip install -e .[dev]
   pip install -r docs/requirements.txt
   ```

3. Add the interlinks extension (first time only), from the `docs` folder:
   ```bash
   quarto add machow/quartodoc --no-prompt
   ```

4. Build the site (from the `docs` folder):
   ```bash
   ./render        # full build -> _site/
   ./preview       # live-reloading local preview
   ```

The compiled site is written to `docs/_site`. The API reference is generated
from docstrings by [quartodoc](https://machow.github.io/quartodoc/) as part of
the pre-render step in `quarto_utils.py`.

## Publishing

`./publish` renders the site and pushes it to the `gh-pages` branch. This
normally runs automatically via GitHub Actions on tagged releases (see
`.github/workflows/publish_docs.yaml`).
