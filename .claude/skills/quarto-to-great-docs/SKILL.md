---
name: quarto-to-great-docs
description: Migrate a Python project's documentation from a hand-maintained Quarto + quartodoc site to Great Docs.
---

# Quarto → Great Docs migration skill

Migrates a documentation site from hand-maintained Quarto (`docs/_quarto.yml` + quartodoc + the `interlinks` filter) to [Great Docs](https://posit-dev.github.io/great-docs/), which generates the whole Quarto project from a single `great-docs.yml`.

## When to use

Invoke directly (`/quarto-to-great-docs`), or when the user asks to migrate/convert docs to Great Docs. Applies to any of the Starsim-family repos (starsim, stisim, fpsim, hpvsim, tbsim, …), which share the same Quarto layout.

## Source of truth

Use <https://posit-dev.github.io/great-docs/> — **not** another repo's `great-docs/` directory, which may be a stock `init` that was never finished. The fastest way to load the docs is `curl -sL https://posit-dev.github.io/great-docs/llms-full.txt`, then grep for `^# ` to get the page list and read the relevant sections. Confirm behaviour against the installed version's `great_docs/config.py` (`DEFAULT_CONFIG` lists every key) and `great_docs/core.py`, since the published docs may describe a newer version.

## Instructions

1. **Survey the existing site.** Read `docs/_quarto.yml` (navbar, sidebar, quartodoc sections, theme, interlinks), the pre/post-render scripts (e.g. `docs/quarto_utils.py`), the build scripts (`render`/`preview`/`publish`/`clean_all`), the docs CI workflow, and `pyproject.toml`. Record the built page list (`find docs/_site -name '*.html' | sort`) as a coverage baseline.

2. **Grep the content for Quarto features that won't survive**: `[](\`sym\`)` interlinks, `{{< var >}}`, `{{< include >}}`, cross-page relative links, and any custom CSS classes. Often the content uses almost none of these, and the migration is mostly a copy — check rather than assume.

3. **Write the plan and present it for approval before changing anything.** Include the known losses in step 10.

4. **Create `great-docs.yml` at the repo root.** Do not run `great-docs init` on a repo that already has a config. Key settings:
   - `module`, `display_name`, `parser` (usually `google` for Starsim repos), `dynamic: true`, `jupyter: python3`, `freeze: auto`
   - `repo:` and `site_url:` — `site_url` is **required** for subdirectory deployments (e.g. `https://starsim.org/tbsim/`); check with `gh api repos/<owner>/<repo>/pages`. Without it, assets 404.
   - `logo:`/`favicon:` — auto-detection only finds `logo.{svg,png}` or `assets/logo.*`, so repos with names like `tbsim-logo.png` must configure them explicitly. Light/dark variants are swapped client-side via a meta tag; both `<img>` tags showing the light file in the HTML source is expected, not a bug.
   - `inline_methods: true` to keep methods on the class page, which is closest to quartodoc's module pages. The default (`5`) splits large classes into one page per method and floods the sidebar.
   - `authors:` and `funding:` — these generate the footer. There is **no** config for custom footer text: Great Docs only writes a footer if `_quarto.yml` lacks one, and it regenerates that file every build. `funding:` is the closest equivalent to a copyright/sponsor line.

5. **Translate the API reference from modules to symbols.** quartodoc's `sections.contents` lists *modules* (`tb`, `interventions.bcg`); Great Docs' `reference` lists *symbols*. Run `great-docs scan` to enumerate everything, then map each old module to its `__all__` (grep `^__all__` across the package). Keep the old section titles and descriptions. Use dotted names (`compartmental.TB_ODE`, `interventions.drug_params`) for anything not re-exported at the top level. Flag any public module that had no reference page — adding it is usually the right call, but say so.

6. **Move narrative content to the repo root**: `docs/user_guide/` → `user_guide/`, `docs/tutorials/` → `tutorials/` (use `git mv`). This matters: `user_guide:` is special-cased and always outputs to `user-guide/`, but a generic section with `dir: docs/tutorials` **preserves the whole path**, giving ugly `/docs/tutorials/…` URLs. Then:
   - Convert `index.md` → `index.qmd` and replace the leading `# Heading` with a frontmatter `title:`.
   - Add numeric prefixes (`01-`, `02-`) to control sidebar order. They are stripped from output filenames, so cross-page links must omit them (`installation.qmd`, not `01-installation.qmd`).
   - Prefer a **flat** user guide over `guide-section:` frontmatter grouping: with `guide-section`, the root `index.qmd` has no section and gets appended to the *bottom* of the sidebar. Flat keeps it first.
   - Do not use the explicit `user_guide:` list form unless you need it — it preserves numeric prefixes in URLs.
   - Fix relative links for the new layout, remembering `user_guide/` → `user-guide/` in the output.

7. **Handle the changelog.** Great Docs generates a changelog from *GitHub Releases*, which in these repos usually lag far behind `CHANGELOG.md`. Prefer `changelog: {enabled: false}` plus a user guide page that inlines the real file:
   - `user_guide/NN-whatsnew.qmd` with a `title:` and `{{< include _changelog.md >}}`
   - `user_guide/_changelog.md` as a **symlink** to `../CHANGELOG.md` (underscore-prefixed so Quarto doesn't render it as its own page)

   A `pre_render` script cannot supply an include target: Quarto expands includes while building its project file list, which happens *before* pre-render hooks run.

8. **Rename community files to uppercase.** Detection is case-sensitive on exact `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `ROADMAP.md`, `CITATION.cff` in the root or `.github/`. Update any references, and delete the `docs/*.md` symlinks the old sidebar needed.

9. **Rewrite the build tooling.** Delete `_quarto.yml`, the pre/post-render script, `_variables.yml`, custom `.scss`/`.css`, the `index.md`/`whatsnew.md` symlinks, and `docs/requirements.txt`. Point `render`/`preview`/`clean_all` at `great-docs build|preview` with a `cd "$(dirname "$0")/.."` first — **`great-docs` must run from the project root**. Drop `publish` if Pages deploys via a workflow (`build_type: workflow`). Swap `quartodoc`/`sphobjinv` for `great-docs` in `[dev]`, add `Repository` and `Documentation` to `[project.urls]`, and gitignore `great-docs/` and `_freeze/` (now at the root, not `docs/`).

10. **Tell the user what is lost.** These are not recoverable by trying harder:
    - **Custom stylesheets** — no SCSS hook; theming is `site.theme`, `navbar_color`, `accent_color`, `include_in_header` only.
    - **External interlinks** — the linking system resolves only the package's own symbols, so intersphinx-style links to NumPy/pandas/Starsim docs stop working.
    - **Outbound `objects.inv`** — Great Docs emits `objects.json` but no Sphinx inventory, so other projects can no longer intersphinx *into* this site. Port the old generator as a post-render script only if the user wants it.
    - **Custom footer text and navbar tools** (email/Slack/site icons) — Great Docs owns both; the navbar right side is the GitHub widget.
    - The navbar logo tooltip reports the latest **GitHub Release**, which will look wrong if releases lag the package version.

11. **Build and expect notebook failures.** `great-docs build` re-executes everything, whereas the old site was probably rendering from a stale `docs/_freeze/`. Genuine, pre-existing bugs in the tutorials will surface. Diagnose each one, confirm it is pre-existing (look for a cached `html.json` in the old `_freeze/`), and fix it — but say clearly in the summary that you touched model code and why. Run the test suite afterwards.

12. **Verify.** Compare the page list against the baseline; confirm executed figures exist (`find great-docs/_site -path '*figure-html*' -name '*.png'`); check the navbar order, sidebar order, homepage hero, and footer; and run `great-docs check-links`. Note that `navbar_after` matches the *literal* navbar text — setting `reference.title` breaks the default "insert before Reference" placement, so anchor custom sections to an item that exists. Absolute GitHub URLs are the fix for relative links in `README.md`/`CONTRIBUTING.md` that must work both on GitHub and on the site.

13. **Update the docs about the docs**: `docs/README.md`, the build-commands section of `CLAUDE.md`, and a `CHANGELOG.md` entry covering the migration, the content move, the API reference restructuring, and anything dropped.
