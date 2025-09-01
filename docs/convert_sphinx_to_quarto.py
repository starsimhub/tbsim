#!/usr/bin/env python3
"""
Script to convert Sphinx RST files to Quarto QMD files.

This script helps migrate existing Sphinx documentation to the new Quarto format.
"""

import os
import re
import sys
from pathlib import Path

def convert_rst_to_qmd(rst_content):
    """Convert RST content to QMD content."""
    
    # Basic RST to Markdown conversions
    content = rst_content
    
    # Headers
    content = re.sub(r'^=+\s*$', '#', content, flags=re.MULTILINE)
    content = re.sub(r'^-+\s*$', '##', content, flags=re.MULTILINE)
    content = re.sub(r'^\^+\s*$', '###', content, flags=re.MULTILINE)
    content = re.sub(r'^~+\s*$', '####', content, flags=re.MULTILINE)
    
    # Links
    content = re.sub(r'`([^`]+)`_', r'[\1]()', content)
    content = re.sub(r'`([^`]+)`_', r'[\1]()', content)
    
    # Code blocks
    content = re.sub(r'\.\. code-block:: python', '```python', content)
    content = re.sub(r'\.\. code-block::', '```', content)
    
    # Inline code
    content = re.sub(r':code:`([^`]+)`', r'`\1`', content)
    
    # Bold and italic
    content = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', content)
    content = re.sub(r'\*([^*]+)\*', r'*\1*', content)
    
    # Lists
    content = re.sub(r'^\s*-\s+', '- ', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\*\s+', '- ', content, flags=re.MULTILINE)
    
    # Remove RST-specific directives
    content = re.sub(r'\.\. toctree::.*?\n\s*:maxdepth:.*?\n\s*:caption:.*?\n', '', content, flags=re.DOTALL)
    content = re.sub(r'\.\. note::', '::: {.callout-note}', content)
    content = re.sub(r'\.\. warning::', '::: {.callout-warning}', content)
    content = re.sub(r'\.\. tip::', '::: {.callout-tip}', content)
    
    # Close callout boxes
    content = re.sub(r'(\n\s*\n)', r'\1:::\n\n', content)
    
    return content

def add_yaml_header(content, title):
    """Add YAML header to QMD content."""
    yaml_header = f"""---
title: "{title}"
---

"""
    return yaml_header + content

def convert_file(rst_file, output_dir):
    """Convert a single RST file to QMD."""
    rst_path = Path(rst_file)
    
    # Read RST content
    with open(rst_file, 'r', encoding='utf-8') as f:
        rst_content = f.read()
    
    # Convert content
    qmd_content = convert_rst_to_qmd(rst_content)
    
    # Extract title from filename or content
    title = rst_path.stem.replace('_', ' ').title()
    
    # Add YAML header
    qmd_content = add_yaml_header(qmd_content, title)
    
    # Create output file
    qmd_file = output_dir / f"{rst_path.stem}.qmd"
    with open(qmd_file, 'w', encoding='utf-8') as f:
        f.write(qmd_content)
    
    print(f"Converted {rst_file} -> {qmd_file}")
    return qmd_file

def main():
    """Main conversion function."""
    if len(sys.argv) < 2:
        print("Usage: python convert_sphinx_to_quarto.py <rst_file_or_directory>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path("converted")
    output_dir.mkdir(exist_ok=True)
    
    if input_path.is_file():
        # Convert single file
        if input_path.suffix == '.rst':
            convert_file(input_path, output_dir)
        else:
            print(f"Error: {input_path} is not an RST file")
            sys.exit(1)
    elif input_path.is_dir():
        # Convert all RST files in directory
        rst_files = list(input_path.glob("**/*.rst"))
        if not rst_files:
            print(f"No RST files found in {input_path}")
            sys.exit(1)
        
        for rst_file in rst_files:
            # Preserve directory structure
            relative_path = rst_file.relative_to(input_path)
            file_output_dir = output_dir / relative_path.parent
            file_output_dir.mkdir(parents=True, exist_ok=True)
            convert_file(rst_file, file_output_dir)
        
        print(f"\nConverted {len(rst_files)} files to {output_dir}")
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

if __name__ == "__main__":
    main()
