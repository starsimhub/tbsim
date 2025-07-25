# Theme Switch Documentation

This documentation site includes a theme switch that allows users to toggle between light and dark themes.

## Features

- **Toggle Switch**: A moon icon (🌙) with a toggle switch in the top navigation bar
- **Persistent Storage**: Theme preference is saved in localStorage and persists across browser sessions
- **System Theme Detection**: Automatically detects and applies the user's system theme preference (if no manual preference is set)
- **Responsive Design**: The theme switch adapts to different screen sizes
- **Smooth Transitions**: CSS transitions provide smooth theme switching animations

## How to Use

1. **Toggle Theme**: Click the moon icon (🌙) or the toggle switch in the top navigation bar
2. **Automatic Persistence**: Your theme choice is automatically saved and will be remembered on future visits
3. **System Preference**: If you haven't manually set a theme, the site will automatically match your system's light/dark mode preference

## Technical Implementation

### Files Added

- `docs/_static/theme-switch.css` - CSS styles for the theme switch and dark/light theme variants
- `docs/_static/theme-switch.js` - JavaScript functionality for theme switching and persistence
- `docs/_templates/layout.html` - Custom layout template that includes the theme switch
- `docs/_templates/nav.html` - Custom navigation template with theme switch positioning

### Configuration

The theme switch is configured in `docs/conf.py`:

```python
# Custom CSS and JS files
html_css_files = [
    'theme-switch.css',
]

html_js_files = [
    'theme-switch.js',
]
```

### CSS Variables

The theme system uses CSS custom properties (variables) for consistent theming:

**Light Theme (default):**
- `--bg-color: #ffffff`
- `--text-color: #333333`
- `--link-color: #2980b9`
- `--code-bg: #f5f5f5`
- `--border-color: #ddd`

**Dark Theme:**
- `--bg-color: #1a1a1a`
- `--text-color: #e0e0e0`
- `--link-color: #4fc3f7`
- `--code-bg: #2d2d2d`
- `--border-color: #444`

### JavaScript API

The theme switch exposes a global API for programmatic control:

```javascript
// Set theme
window.themeSwitch.setTheme('dark'); // or 'light'

// Get current theme
const currentTheme = window.themeSwitch.getTheme();

// Toggle between themes
window.themeSwitch.toggle();
```

## Customization

### Changing Colors

To modify the theme colors, edit the CSS variables in `docs/_static/theme-switch.css`:

```css
[data-theme="dark"] {
    --bg-color: #your-dark-bg-color;
    --text-color: #your-dark-text-color;
    /* ... other variables */
}
```

### Changing Position

To move the theme switch to a different location, modify the positioning in `docs/_templates/nav.html` or update the JavaScript positioning logic in `docs/_static/theme-switch.js`.

### Adding New Theme Elements

To style additional elements for dark mode, add CSS rules to the `[data-theme="dark"]` section in `docs/_static/theme-switch.css`.

## Browser Support

The theme switch requires:
- Modern browsers with CSS custom properties support
- JavaScript enabled
- localStorage support for persistence

## Troubleshooting

If the theme switch doesn't appear:
1. Ensure the static files are properly built (`make html`)
2. Check that the CSS and JS files are included in the HTML output
3. Verify that the templates are being used correctly

If themes don't persist:
1. Check that localStorage is enabled in the browser
2. Ensure JavaScript is enabled
3. Check browser console for any JavaScript errors 