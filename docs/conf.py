import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'TBsim'
copyright = '2024, IDM, Burnet Institute, and Collaborators'
author = 'Daniel Klein, Minerva Enriquez, Stewart Chang, Deven Gokhale, Mike Famulare'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'nbsphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autodoc_mock_imports = ["starsim"]

# nbsphinx configuration
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_timeout = 600
nbsphinx_requirejs_path = ''
nbsphinx_widgets_path = ''
nbsphinx_thumbnails_path = ''
nbsphinx_use_rtd_theme = True

# Theme configuration
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'includehidden': True,
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'canonical_url': '',
    'analytics_id': '',
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

# Custom CSS and JS files
html_css_files = [
    'theme-switch.css',
]

html_js_files = [
    'theme-switch.js',
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