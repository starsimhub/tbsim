import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'tbsim'
copyright = '2024, Your Name'
author = 'Your Name'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autodoc_mock_imports = ["starsim"]

# Theme configuration
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'includehidden': True,
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    'canonical_url': '',
    'analytics_id': '',
}

# Additional HTML context
html_context = {
    'display_github': False,
    'display_gitlab': False,
    'display_bitbucket': False,
}

# Custom CSS and JS files
html_css_files = [
    'theme-switch.css',
]

html_js_files = [
    'theme-switch.js',
] 