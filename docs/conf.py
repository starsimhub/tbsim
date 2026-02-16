import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'TBsim'
copyright = '2024, IDM, Burnet Institute, and Collaborators'
author = 'Daniel Klein, Minerva Enriquez, Stewart Chang, Deven Gokhale, Mike Famulare'
release = '0.5.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
    'sphinx_copybutton',
    'sphinx_design',
    'nbsphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']
autodoc_mock_imports = ["starsim", "sciris", "matplotlib", "pandas", "numpy", "rdata"]

# nbsphinx configuration
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_timeout = 600
nbsphinx_requirejs_path = ''
nbsphinx_widgets_path = ''
nbsphinx_thumbnails_path = ''
nbsphinx_use_rtd_theme = False

# Furo theme configuration
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1e40af",
        "color-admonition-background": "#f0f9ff",
        "color-admonition-title-background": "#0ea5e9",
        "color-highlighted-background": "#fef3c7",
        "color-sidebar-background": "#f8fafc",
        "color-sidebar-background-border": "#e2e8f0",
        "color-sidebar-item-background": "#ffffff",
        "color-sidebar-item-background--hover": "#f1f5f9",
        "color-sidebar-item-background--current": "#dbeafe",
        "color-sidebar-item-text": "#475569",
        "color-sidebar-item-text--current": "#1e40af",
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-search-border": "#e2e8f0",
        "color-sidebar-search-icon": "#64748b",
        "color-sidebar-brand-text": "#1e293b",
        "color-content-foreground": "#1e293b",
        "color-content-background": "#ffffff",
        "color-content-border": "#e2e8f0",
        "color-announcement-background": "#fef3c7",
        "color-announcement-text": "#92400e",
        "color-toc-background": "#f8fafc",
        "color-toc-border": "#e2e8f0",
        "color-toc-title-text": "#1e293b",
        "color-toc-item-text": "#475569",
        "color-toc-item-text--hover": "#1e40af",
        "color-toc-item-text--active": "#1e40af",
        "color-link": "#2563eb",
        "color-link--hover": "#1d4ed8",
        "color-link-underline": "#cbd5e1",
        "color-link-underline--hover": "#94a3b8",
        "color-background-item": "#f8fafc",
        "color-background-item--hover": "#f1f5f9",
        "color-inline-code-background": "#f1f5f9",
        "color-inline-code": "#dc2626",
        "color-card-background": "#ffffff",
        "color-card-border": "#e2e8f0",
        "color-card-margin": "#f8fafc",
        "color-card-shadow": "rgba(0, 0, 0, 0.1)",
        "color-card-shadow-hover": "rgba(0, 0, 0, 0.15)",
        "color-card-stripe": "#f1f5f9",
        "color-tabs-background": "#f8fafc",
        "color-tabs-background--active": "#ffffff",
        "color-tabs-label": "#64748b",
        "color-tabs-label--active": "#1e293b",
        "color-tabs-label--hover": "#475569",
        "color-tabs-underline": "#e2e8f0",
        "color-tabs-underline--active": "#2563eb",
        "color-tabs-underline--hover": "#94a3b8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
        "color-admonition-background": "#1e3a8a",
        "color-admonition-title-background": "#1d4ed8",
        "color-highlighted-background": "#451a03",
        "color-sidebar-background": "#0f172a",
        "color-sidebar-background-border": "#1e293b",
        "color-sidebar-item-background": "#1e293b",
        "color-sidebar-item-background--hover": "#334155",
        "color-sidebar-item-background--current": "#1e40af",
        "color-sidebar-item-text": "#cbd5e1",
        "color-sidebar-item-text--current": "#93c5fd",
        "color-sidebar-search-background": "#1e293b",
        "color-sidebar-search-border": "#334155",
        "color-sidebar-search-icon": "#94a3b8",
        "color-sidebar-brand-text": "#f1f5f9",
        "color-content-foreground": "#f1f5f9",
        "color-content-background": "#0f172a",
        "color-content-border": "#1e293b",
        "color-announcement-background": "#451a03",
        "color-announcement-text": "#fbbf24",
        "color-toc-background": "#0f172a",
        "color-toc-border": "#1e293b",
        "color-toc-title-text": "#f1f5f9",
        "color-toc-item-text": "#cbd5e1",
        "color-toc-item-text--hover": "#93c5fd",
        "color-toc-item-text--active": "#93c5fd",
        "color-link": "#60a5fa",
        "color-link--hover": "#93c5fd",
        "color-link-underline": "#475569",
        "color-link-underline--hover": "#64748b",
        "color-background-item": "#1e293b",
        "color-background-item--hover": "#334155",
        "color-inline-code-background": "#1e293b",
        "color-inline-code": "#f87171",
        "color-card-background": "#1e293b",
        "color-card-border": "#334155",
        "color-card-margin": "#0f172a",
        "color-card-shadow": "rgba(0, 0, 0, 0.3)",
        "color-card-shadow-hover": "rgba(0, 0, 0, 0.4)",
        "color-card-stripe": "#334155",
        "color-tabs-background": "#0f172a",
        "color-tabs-background--active": "#1e293b",
        "color-tabs-label": "#94a3b8",
        "color-tabs-label--active": "#f1f5f9",
        "color-tabs-label--hover": "#cbd5e1",
        "color-tabs-underline": "#334155",
        "color-tabs-underline--active": "#60a5fa",
        "color-tabs-underline--hover": "#64748b",
    },
    "sidebar_hide_name": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/starsimhub/tbsim/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/starsimhub/tbsim",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Additional HTML context
html_context = {
    'display_github': True,
    'github_user': 'starsimhub',
    'github_repo': 'tbsim',
    'github_version': 'main',
    'conf_py_path': '/docs/',
    'source_suffix': '.rst',
}

# Custom CSS and JS files (Furo has built-in dark mode)
html_css_files = [
    'custom.css',
]

html_js_files = [
    'custom.js',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_warnings = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_include_summary_with_docstring = False
napoleon_custom_sections = None
napoleon_use_keyword = True
napoleon_attr_annotations = True

# Autosummary: generate API stubs from code at build time
autosummary_generate = True

# Additional autodoc settings
autodoc_ignore_module_all = True
autodoc_preserve_defaults = True