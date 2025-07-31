# MkDocs Documentation Setup

This project has been converted from Sphinx to MkDocs for improved documentation management and user experience.

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install -r docs/requirements-mkdocs.txt
```

### Building the Documentation

To build the documentation:

```bash
mkdocs build
```

The built documentation will be available in the `site/` directory.

### Serving the Documentation Locally

To serve the documentation locally for development:

```bash
mkdocs serve
```

This will start a local server (usually at http://127.0.0.1:8000) where you can view the documentation in your browser.

## Configuration

The MkDocs configuration is in `mkdocs.yml`. Key features include:

- **Material Theme**: Modern, responsive theme with dark/light mode toggle
- **Auto-generated API docs**: Using mkdocstrings for Python API documentation
- **Jupyter notebook support**: Direct integration with Jupyter notebooks
- **Search functionality**: Full-text search across all documentation
- **Navigation**: Hierarchical navigation with expandable sections

## Structure

```
docs/
├── index.md                    # Home page
├── api/                        # API documentation
│   ├── overview.md             # API overview
│   ├── tbsim.md               # Main module docs
│   ├── tbsim.tb.md            # TB simulation docs
│   └── ...                    # Other module docs
├── tutorials/                  # Tutorials
│   ├── example_tutorial.md     # Basic tutorial
│   ├── tuberculosis_sim.md     # TB simulation tutorial
│   └── ...                    # Other tutorials
├── references/                 # Reference materials
│   ├── index.md               # References overview
│   └── ...                    # PDFs and other files
├── css/                       # Custom CSS
└── js/                        # Custom JavaScript
```

## Key Features

### Auto-generated API Documentation

API documentation is automatically generated from Python docstrings using mkdocstrings. Each module has its own page with:

- Function and class documentation
- Parameter descriptions
- Return value information
- Source code links

### Jupyter Notebook Integration

Jupyter notebooks are automatically converted to documentation pages with:

- Code execution (optional)
- Output display
- Interactive plots
- Syntax highlighting

### Custom Styling

Custom CSS and JavaScript files are included for:

- Theme switching functionality
- Custom styling
- Enhanced user experience

## Migration from Sphinx

The following changes were made during the migration:

1. **Configuration**: Replaced `conf.py` with `mkdocs.yml`
2. **File format**: Converted RST files to Markdown
3. **Navigation**: Restructured navigation using MkDocs format
4. **Static files**: Moved static files to appropriate directories
5. **API docs**: Replaced Sphinx autodoc with mkdocstrings

## Development

### Adding New Pages

1. Create a new Markdown file in the appropriate directory
2. Add the page to the navigation in `mkdocs.yml`
3. Use the `::: module_name` syntax for auto-generated API docs

### Customizing the Theme

Edit the `theme:` section in `mkdocs.yml` to customize:

- Color schemes
- Navigation features
- Search behavior
- Typography

### Adding Plugins

MkDocs supports various plugins for additional functionality. Add them to the `plugins:` section in `mkdocs.yml`.

## Deployment

The documentation can be deployed to various platforms:

- **GitHub Pages**: Use `mkdocs gh-deploy`
- **Netlify**: Connect your repository to Netlify
- **Read the Docs**: Configure for MkDocs support

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the tbsim package is installed and accessible
2. **Missing modules**: Check that all referenced modules exist
3. **Build errors**: Verify all Markdown files are properly formatted

### Getting Help

- Check the [MkDocs documentation](https://www.mkdocs.org/)
- Review the [Material theme documentation](https://squidfunk.github.io/mkdocs-material/)
- Consult the [mkdocstrings documentation](https://mkdocstrings.github.io/) 