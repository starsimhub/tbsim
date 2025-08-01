# Minimal Makefile for Sphinx documentation
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: html clean clean-html help

help:
	@echo "Available targets:"
	@echo "  html       - Build HTML documentation"
	@echo "  clean      - Remove build directory"
	@echo "  clean-html - Clean and rebuild HTML documentation"
	@echo "  help       - Show this help message"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

clean-html: clean html 