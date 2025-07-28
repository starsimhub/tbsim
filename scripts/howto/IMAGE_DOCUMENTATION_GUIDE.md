# Image Documentation Guide for TB Simulation

This guide explains how to add images to docstrings in the TB simulation project.

## Overview

Yes, you can render pictures as part of docstrings! The workflow is:

1. **Generate the images first** using a script
2. **Save them to `docs/_static/`** directory
3. **Reference them in docstrings** using Sphinx image directives
4. **Build the documentation** to see the results

## Complete Workflow

### Step 1: Generate Images

Use the provided script to generate example images:

```bash
cd /path/to/tbdocs
python scripts/howto/generate_doc_images.py
```

This creates:
- `sankey_diagram_example.png`
- `histogram_kde_example.png`
- `network_graph_example.png`
- `reinfection_analysis_example.png`
- `interactive_bar_example.png`

### Step 2: Use Images in Docstrings

Add images to your docstrings using Sphinx image directives:

```python
def my_function():
    """
    This function creates amazing visualizations.
    
    .. image:: _static/sankey_diagram_example.png
        :width: 600px
        :alt: Sankey diagram showing TB state transitions
        :align: center
    
    The diagram above shows how agents flow between TB states.
    """
    pass
```

### Step 3: Build Documentation

```bash
cd docs
make html
```

## Image Options

### Sphinx Image Directive Options

```rst
.. image:: _static/filename.png
    :width: 600px          # Width of the image
    :height: 400px         # Height (optional)
    :alt: Description      # Alt text for accessibility
    :align: center         # Alignment (left, center, right)
    :scale: 50%           # Scale percentage
    :target: URL          # Make image clickable
```

### Common Widths

- **Small**: 300px - 400px
- **Medium**: 500px - 600px  
- **Large**: 700px - 800px
- **Extra Large**: 900px+

## Example: Enhanced DwtAnalyzer Docstrings

Here's how to enhance your analyzer docstrings with images:

```python
class DwtAnalyzer:
    def sankey_agents(self, subtitle=""):
        """
        Creates a Sankey diagram showing agent flow between TB states.
        
        .. image:: _static/sankey_diagram_example.png
            :width: 600px
            :alt: Sankey diagram showing TB state transitions
            :align: center
        
        The Sankey diagram above shows:
        - **Nodes**: TB states (Susceptible, Latent, Active, etc.)
        - **Flows**: Agent transitions between states
        - **Width**: Proportional to the number of agents
        - **Colors**: Different state categories
        
        Args:
            subtitle (str): Additional subtitle for the plot
            
        Returns:
            None: Displays interactive Plotly Sankey diagram
        """
        pass
```

## Best Practices

### 1. Image Quality
- Use high DPI (150-300) for crisp images
- Save as PNG for best quality
- Keep file sizes reasonable (< 200KB)

### 2. Documentation Structure
- Place images after the main description
- Add detailed explanations below each image
- Include mathematical models and examples

### 3. Accessibility
- Always include `:alt:` text
- Use descriptive alt text
- Consider colorblind-friendly palettes

### 4. File Organization
- Keep all images in `docs/_static/`
- Use descriptive filenames
- Group related images together

## Available Images

The following images are available for use in docstrings:

| Image | Filename | Description | Recommended Width |
|-------|----------|-------------|-------------------|
| Sankey Diagram | `sankey_diagram_example.png` | TB state transitions | 600px |
| Histogram with KDE | `histogram_kde_example.png` | Dwell time distributions | 500px |
| Network Graph | `network_graph_example.png` | State transition network | 700px |
| Reinfection Analysis | `reinfection_analysis_example.png` | Reinfection percentages | 800px |
| Interactive Bar Chart | `interactive_bar_example.png` | State transitions by time | 800px |

## Custom Image Generation

To create custom images for your specific functions:

1. **Add a new function** to `scripts/generate_doc_images.py`
2. **Generate the image** using matplotlib or other plotting libraries
3. **Save to `docs/_static/`** with a descriptive filename
4. **Update the main() function** to include your new image
5. **Run the script** to generate all images

Example:

```python
def create_my_custom_plot():
    """Create a custom plot for my function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... create your plot ...
    return fig

# In main():
examples = [
    # ... existing examples ...
    ('my_custom_plot.png', create_my_custom_plot),
]
```

## Testing Images

To test that your images are working:

1. **Build the documentation**: `cd docs && make html`
2. **Check the output**: Look for your images in `_build/html/_static/`
3. **View the documentation**: Open `_build/html/index.html` in a browser
4. **Verify images load**: Check that images appear correctly

## Troubleshooting

### Images Not Showing
- Check file paths are correct
- Ensure images are in `docs/_static/`
- Verify Sphinx image directive syntax
- Check file permissions

### Poor Image Quality
- Increase DPI when saving (150-300)
- Use vector formats when possible
- Optimize file sizes for web

### Build Errors
- Check RST syntax
- Verify image files exist
- Look for typos in filenames

## Summary

✅ **Yes, you can render pictures as part of docstrings!**

The process is:
1. Generate images → Save to `docs/_static/` → Reference in docstrings → Build docs

This approach provides:
- Fast-loading documentation
- Professional appearance
- Consistent visual examples
- Better user understanding

The images are automatically included when you build the documentation with Sphinx. 