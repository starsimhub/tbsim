# TBsim documentation

The TBsim documentation site is built with [Great Docs](https://posit-dev.github.io/great-docs/), which generates a [Quarto](https://quarto.org) site from a single configuration file plus the narrative content.

## Where things live

Because Great Docs discovers content relative to the repository root, the source pages live at the top level rather than under `docs/`:

| Path | Contents |
|------|----------|
| `great-docs.yml` | All site configuration: navigation, theming, authors, and the API reference structure |
| `user_guide/` | User guide pages (`.qmd`), ordered by their numeric filename prefixes |
| `tutorials/` | Runnable tutorial notebooks (`.qmd`), ordered by their numeric filename prefixes |
| `docs/assets/` | Logo and favicon |
| `docs/references/` | Background reference material (papers, schematics); not part of the rendered site |
| `README.md` | Rendered as the site homepage |
| `CHANGELOG.md` | Rendered as the "What's new" page, via the `user_guide/_changelog.md` symlink |
| `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE` | Auto-detected and rendered as their own pages |

The numeric prefixes (`01-`, `02-`, …) control sidebar order only; Great Docs strips them from the output filenames, so `user_guide/01-installation.qmd` is served as `user-guide/installation.html`. Link between pages without the prefix.

The API reference is generated from docstrings. Which symbols appear, and how they are grouped, is set by the `reference:` section of `great-docs.yml`. To see what is available to add there:

```bash
great-docs scan
```

## Building the docs

1. Install [Quarto](https://quarto.org/docs/get-started/).

2. Install TBsim and the documentation dependencies:
   ```bash
   pip install -e .[dev]
   ```

3. Build the site (from this folder, or run `great-docs` directly from the repository root):
   ```bash
   ./render        # full build -> great-docs/_site/
   ./preview       # live-reloading local preview
   ./clean_all     # remove the build directory and execution cache
   ```

The `great-docs/` directory is ephemeral: it is regenerated on every build and is gitignored, along with the `_freeze/` execution cache. Never edit anything inside it — including its `_quarto.yml`, which Great Docs rewrites each time. All configuration belongs in `great-docs.yml`.

Executed code cells are cached in `_freeze/` and only re-run when a page's source changes. If a dependency update changes outputs without changing the page, refresh it explicitly:

```bash
great-docs freeze tutorials/01-tuberculosis_sim.qmd   # re-execute one page
great-docs freeze --info                              # show what is cached
```

To check for broken links across the site and source docstrings:

```bash
great-docs check-links
```

## Publishing

`.github/workflows/publish_docs.yaml` builds the site on every push to `main` and on pull requests, and deploys to GitHub Pages from `main`. The published site is <https://starsim.org/tbsim/>.
