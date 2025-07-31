// Theme Switch JavaScript
(function() {
    'use strict';

    // Theme switch functionality
    function initThemeSwitch() {
        const themeSwitch = document.getElementById('theme-switch');
        const html = document.documentElement;
        
        // Get saved theme from localStorage or default to 'light'
        const currentTheme = localStorage.getItem('theme') || 'light';
        
        // Apply the saved theme
        html.setAttribute('data-theme', currentTheme);
        if (themeSwitch) {
            themeSwitch.checked = currentTheme === 'dark';
        }
        
        // Add event listener for theme switch
        if (themeSwitch) {
            themeSwitch.addEventListener('change', function() {
                const newTheme = this.checked ? 'dark' : 'light';
                html.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                
                // Trigger a custom event for other scripts that might need to know about theme changes
                const event = new CustomEvent('themeChanged', { detail: { theme: newTheme } });
                document.dispatchEvent(event);
            });
        }
        
        // Add theme switch to the navigation if it doesn't exist
        if (!themeSwitch) {
            addThemeSwitchToNav();
        }
    }
    
    function addThemeSwitchToNav() {
        const navTop = document.querySelector('.wy-nav-top');
        if (!navTop) {
            // Try alternative selectors
            const alternatives = [
                '.wy-nav-content-wrap',
                '.wy-nav-content',
                '.document',
                'header',
                'nav'
            ];
            
            for (let selector of alternatives) {
                const element = document.querySelector(selector);
                if (element) {
                    addThemeSwitchToElement(element);
                    return;
                }
            }
            return;
        }
        
        addThemeSwitchToElement(navTop);
    }
    
    function addThemeSwitchToElement(element) {
        // Check if theme switch already exists
        if (document.querySelector('.theme-switch-wrapper')) {
            return;
        }
        
        // Create theme switch container
        const themeSwitchWrapper = document.createElement('div');
        themeSwitchWrapper.className = 'theme-switch-wrapper';
        themeSwitchWrapper.style.cssText = `
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            align-items: center;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.1);
            padding: 5px 10px;
            border-radius: 20px;
            backdrop-filter: blur(5px);
        `;
        
        // Create label
        const label = document.createElement('span');
        label.className = 'theme-switch-label';
        label.textContent = '';
        label.style.cssText = `
            margin-right: 8px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
        `;
        
        // Create toggle switch
        const themeSwitch = document.createElement('label');
        themeSwitch.className = 'theme-switch';
        themeSwitch.innerHTML = `
            <input type="checkbox" id="theme-switch">
            <span class="slider"></span>
        `;
        
        // Add elements to wrapper
        themeSwitchWrapper.appendChild(label);
        themeSwitchWrapper.appendChild(themeSwitch);
        
        // Add to navigation
        element.appendChild(themeSwitchWrapper);
        
        // Initialize the switch
        initThemeSwitch();
    }
    
    // Initialize when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initThemeSwitch);
    } else {
        initThemeSwitch();
    }
    
    // Also try to add theme switch after a short delay to ensure DOM is fully loaded
    setTimeout(function() {
        if (!document.querySelector('.theme-switch-wrapper')) {
            addThemeSwitchToNav();
        }
    }, 1000);
    
    // Handle theme changes from other sources (like browser preferences)
    function handleSystemThemeChange() {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        
        function handleChange(e) {
            // Only change theme if user hasn't manually set a preference
            if (!localStorage.getItem('theme')) {
                const newTheme = e.matches ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', newTheme);
                const themeSwitch = document.getElementById('theme-switch');
                if (themeSwitch) {
                    themeSwitch.checked = newTheme === 'dark';
                }
            }
        }
        
        mediaQuery.addEventListener('change', handleChange);
        
        // Initial check
        if (!localStorage.getItem('theme')) {
            handleChange(mediaQuery);
        }
    }
    
    // Initialize system theme detection
    if (window.matchMedia) {
        handleSystemThemeChange();
    }
    
    // Export functions for potential external use
    window.themeSwitch = {
        setTheme: function(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            const themeSwitch = document.getElementById('theme-switch');
            if (themeSwitch) {
                themeSwitch.checked = theme === 'dark';
            }
        },
        getTheme: function() {
            return localStorage.getItem('theme') || 'light';
        },
        toggle: function() {
            const currentTheme = this.getTheme();
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            this.setTheme(newTheme);
        }
    };
    
})(); 