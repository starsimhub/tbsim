// Custom JavaScript for TBsim documentation with Furo theme

(function() {
    'use strict';

    // Initialize when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeCustomFeatures();
    });

    function initializeCustomFeatures() {
        // Add smooth scrolling for anchor links
        addSmoothScrolling();
        
        // Improve code block interactions
        enhanceCodeBlocks();
        
        // Add keyboard shortcuts
        addKeyboardShortcuts();
        
        // Improve mobile navigation
        enhanceMobileNavigation();
        
        // Add search enhancements
        enhanceSearch();
    }

    function addSmoothScrolling() {
        // Add smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update URL without jumping
                    if (history.pushState) {
                        history.pushState(null, null, this.getAttribute('href'));
                    }
                }
            });
        });
    }

    function enhanceCodeBlocks() {
        // Add click-to-copy functionality for code blocks
        document.querySelectorAll('.highlight').forEach(block => {
            const code = block.querySelector('pre code');
            if (code) {
                // Create copy button
                const copyButton = document.createElement('button');
                copyButton.className = 'copybtn';
                copyButton.innerHTML = '📋';
                copyButton.title = 'Copy to clipboard';
                copyButton.style.cssText = `
                    position: absolute;
                    top: 0.5rem;
                    right: 0.5rem;
                    background: var(--color-background-item);
                    border: 1px solid var(--color-background-border);
                    border-radius: 0.25rem;
                    padding: 0.25rem 0.5rem;
                    cursor: pointer;
                    font-size: 0.875rem;
                    opacity: 0.7;
                    transition: all 0.2s ease;
                `;
                
                // Make the block container relative
                block.style.position = 'relative';
                block.appendChild(copyButton);
                
                // Add click handler
                copyButton.addEventListener('click', function() {
                    const text = code.textContent;
                    navigator.clipboard.writeText(text).then(() => {
                        copyButton.innerHTML = '✓';
                        copyButton.style.opacity = '1';
                        setTimeout(() => {
                            copyButton.innerHTML = '📋';
                            copyButton.style.opacity = '0.7';
                        }, 2000);
                    }).catch(() => {
                        // Fallback for older browsers
                        const textArea = document.createElement('textarea');
                        textArea.value = text;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                        
                        copyButton.innerHTML = '✓';
                        copyButton.style.opacity = '1';
                        setTimeout(() => {
                            copyButton.innerHTML = '📋';
                            copyButton.style.opacity = '0.7';
                        }, 2000);
                    });
                });
                
                // Show/hide button on hover
                block.addEventListener('mouseenter', () => {
                    copyButton.style.opacity = '1';
                });
                
                block.addEventListener('mouseleave', () => {
                    if (copyButton.innerHTML === '📋') {
                        copyButton.style.opacity = '0.7';
                    }
                });
            }
        });
    }

    function addKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('.sidebar-search input');
                if (searchInput) {
                    searchInput.focus();
                }
            }
            
            // Escape to clear search
            if (e.key === 'Escape') {
                const searchInput = document.querySelector('.sidebar-search input');
                if (searchInput && document.activeElement === searchInput) {
                    searchInput.blur();
                    searchInput.value = '';
                    // Trigger search update if needed
                    searchInput.dispatchEvent(new Event('input'));
                }
            }
        });
    }

    function enhanceMobileNavigation() {
        // Improve mobile sidebar behavior
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        
        if (sidebarToggle && sidebar) {
            // Close sidebar when clicking outside on mobile
            document.addEventListener('click', function(e) {
                if (window.innerWidth <= 768 && sidebar.classList.contains('sidebar-open')) {
                    if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
                        sidebar.classList.remove('sidebar-open');
                    }
                }
            });
            
            // Close sidebar when pressing escape
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' && sidebar.classList.contains('sidebar-open')) {
                    sidebar.classList.remove('sidebar-open');
                }
            });
        }
    }

    function enhanceSearch() {
        const searchInput = document.querySelector('.sidebar-search input');
        if (searchInput) {
            // Add search suggestions placeholder
            searchInput.placeholder = 'Search documentation...';
            
            // Add search result highlighting
            searchInput.addEventListener('input', function() {
                const query = this.value.toLowerCase();
                if (query.length > 2) {
                    highlightSearchResults(query);
                } else {
                    clearSearchHighlights();
                }
            });
        }
    }

    function highlightSearchResults(query) {
        // Simple search highlighting for visible content
        const content = document.querySelector('.content');
        if (content) {
            const walker = document.createTreeWalker(
                content,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                if (node.textContent.toLowerCase().includes(query)) {
                    textNodes.push(node);
                }
            }
            
            textNodes.forEach(textNode => {
                const parent = textNode.parentNode;
                if (parent.tagName !== 'SCRIPT' && parent.tagName !== 'STYLE') {
                    const text = textNode.textContent;
                    const regex = new RegExp(`(${query})`, 'gi');
                    const highlighted = text.replace(regex, '<mark style="background-color: var(--color-highlighted-background); padding: 0.125rem 0.25rem; border-radius: 0.25rem;">$1</mark>');
                    
                    if (highlighted !== text) {
                        const wrapper = document.createElement('span');
                        wrapper.innerHTML = highlighted;
                        parent.replaceChild(wrapper, textNode);
                    }
                }
            });
        }
    }

    function clearSearchHighlights() {
        const marks = document.querySelectorAll('mark');
        marks.forEach(mark => {
            const parent = mark.parentNode;
            parent.replaceChild(document.createTextNode(mark.textContent), mark);
            parent.normalize();
        });
    }

    // Add API documentation specific enhancements
    function enhanceAPIDocumentation() {
        // Add parameter type hints
        document.querySelectorAll('.sig-param').forEach(param => {
            const text = param.textContent;
            if (text.includes(':')) {
                const [name, type] = text.split(':');
                param.innerHTML = `<span class="param-name">${name.trim()}</span>: <span class="param-type">${type.trim()}</span>`;
            }
        });
        
        // Add return type styling
        document.querySelectorAll('.sig-return-typehint').forEach(returnType => {
            returnType.style.color = 'var(--color-brand-primary)';
            returnType.style.fontWeight = '600';
        });
    }

    // Initialize API enhancements
    document.addEventListener('DOMContentLoaded', enhanceAPIDocumentation);

    // Add theme-aware functionality
    function initializeThemeAwareFeatures() {
        // Observe theme changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                    // Re-initialize features that depend on theme
                    setTimeout(enhanceCodeBlocks, 100);
                }
            });
        });
        
        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
    }

    // Initialize theme-aware features
    document.addEventListener('DOMContentLoaded', initializeThemeAwareFeatures);

    // Export functions for potential external use
    window.tbsimDocs = {
        highlightSearch: highlightSearchResults,
        clearHighlights: clearSearchHighlights,
        enhanceCodeBlocks: enhanceCodeBlocks
    };

})();
