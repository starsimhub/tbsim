# TBSim Documentation

This directory contains the documentation for TBSim, built using [Quarto](https://quarto.org/), a modern documentation system that provides better theming, navigation, and user experience compared to traditional Sphinx documentation.

## Overview

The documentation has been migrated from Sphinx to Quarto to provide:

- **Better theme support** with automatic light/dark mode switching
- **Modern styling** with custom CSS that prevents organization theme overrides
- **Improved navigation** and search functionality
- **Better mobile responsiveness**
- **Enhanced code highlighting** and syntax support
- **Callout boxes** for notes, warnings, and tips

## Structure

```
docs/
├── _quarto.yml              # Quarto configuration
├── index.qmd                # Main landing page
├── tutorials.qmd            # Tutorials index
├── user_guide.qmd           # User guide index
├── api/                     # API documentation
├── tutorials/               # Tutorial notebooks
├── user_guide/              # User guide content
├── assets/                  # Static assets
│   ├── styles.css           # Main CSS file
│   ├── styles-light.scss    # Light theme styles
│   └── styles-dark.scss     # Dark theme styles
└── requirements-quarto.txt  # Python dependencies
```

## Building the Documentation

### Prerequisites

1. Install Quarto: https://quarto.org/docs/get-started/
2. Install Python dependencies:
   ```bash
   pip install -r docs/requirements-quarto.txt
   ```

### Local Development

1. Navigate to the docs directory:
   ```bash
   cd docs
   ```

2. Start the Quarto preview server:
   ```bash
   quarto preview
   ```

3. Open your browser to the URL shown (typically http://localhost:4848)

### Building for Production

```bash
cd docs
quarto render --to html
```

The built documentation will be in `docs/_site/`.

## Theme System

The documentation uses a custom theme system with:

- **Light theme**: Clean, modern design with good contrast
- **Dark theme**: Easy on the eyes for low-light environments
- **Automatic switching**: Respects user's system preference
- **Manual override**: Users can manually switch themes

### Theme Features

- **CSS Variables**: Consistent theming across all components
- **Responsive Design**: Works well on all screen sizes
- **Accessibility**: High contrast ratios and readable fonts
- **Organization Override Protection**: Uses `!important` declarations to prevent organization themes from overriding the custom styling

## Content Guidelines

### Writing Documentation

1. **Use Markdown**: All content should be written in Markdown (`.qmd` files)
2. **Include YAML headers**: Each file should have appropriate YAML metadata
3. **Use callouts**: Highlight important information with callout boxes
4. **Include code examples**: Provide working code examples where possible
5. **Cross-reference**: Link between related sections

### Callout Types

- `.callout-note`: General information
- `.callout-tip`: Helpful tips and best practices
- `.callout-warning`: Important warnings
- `.callout-caution`: Critical information

Example:
```markdown
::: {.callout-note}
## Note
This is a note callout.
:::

::: {.callout-tip}
## Tip
This is a tip callout.
:::
```

## Migration from Sphinx

The documentation has been migrated from Sphinx to Quarto. Key changes:

- **File extensions**: `.rst` → `.qmd`
- **Configuration**: `conf.py` → `_quarto.yml`
- **Build system**: `make html` → `quarto render`
- **Output directory**: `_build/html/` → `_site/`

### Converting Existing Content

To convert existing Sphinx content to Quarto:

1. **RST to Markdown**: Convert `.rst` files to `.qmd` files
2. **Update links**: Change internal links to use Markdown syntax
3. **Update includes**: Replace Sphinx directives with Quarto equivalents
4. **Test rendering**: Ensure all content renders correctly

## Deployment

The documentation is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the main branch. The workflow:

1. Installs Quarto and dependencies
2. Builds the documentation
3. Deploys to GitHub Pages
4. Provides preview builds for pull requests

## Troubleshooting

### Common Issues

1. **Theme not applying**: Check that CSS files are properly linked in `_quarto.yml`
2. **Build failures**: Ensure all dependencies are installed
3. **Missing content**: Verify file paths in `_quarto.yml` sidebar configuration

### Getting Help

- [Quarto Documentation](https://quarto.org/docs/)
- [GitHub Issues](https://github.com/your-org/tbsim/issues)
- [Quarto Community](https://github.com/quarto-dev/quarto/discussions)

## Contributing

When contributing to the documentation:

1. Follow the existing style and structure
2. Test your changes locally before submitting
3. Update the table of contents in `_quarto.yml` if adding new pages
4. Ensure all links work correctly
5. Add appropriate callouts for important information
