# Contributing to TBsim

We welcome contributions to the TBsim project! This guide explains how you can get involved.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** following the coding standards
5. **Test your changes** thoroughly
6. **Submit a pull request** with a clear description

## Development Setup

1. Install TBsim in development mode:

   ```bash
   git clone https://github.com/yourusername/tbsim.git
   cd tbsim
   pip install -e .
   ```

2. Install development dependencies:

   ```bash
   pip install -r tests/requirements.txt
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Code Standards

**Python Code Style**

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and concise

**Documentation**

- Update relevant documentation when changing functionality
- Add docstrings for new functions and classes
- Include examples in docstrings where helpful

**Testing**

- Write tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage

**Git Workflow**

- Use descriptive commit messages
- Keep commits focused and atomic
- Reference issues in commit messages when relevant

## Areas for Contribution

**Core Functionality**

- TB model improvements
- New intervention types
- Enhanced comorbidity modeling
- Network structure improvements

**Analysis Tools**

- New analyzers and visualizations
- Statistical analysis methods
- Export and reporting tools

**Documentation**

- Tutorial improvements
- API documentation updates
- User guide enhancements

**Testing and Quality**

- Additional test cases
- Performance improvements
- Bug fixes and error handling

**Examples and Tutorials**

- New use case examples
- Tutorial improvements
- Sample data and scenarios

## Submitting Changes

1. **Ensure your code works** and all tests pass
2. **Update documentation** as needed
3. **Write a clear pull request description** explaining:
   - What the change does
   - Why it's needed
   - How it was tested
4. **Reference related issues** if applicable
5. **Request review** from maintainers

## Review Process

- All contributions require review before merging
- Maintainers will review for:
  - Code quality and standards
  - Functionality and correctness
  - Documentation completeness
  - Test coverage
- Address feedback and make requested changes

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check existing docs first
- **Code Examples**: Look at existing implementations

Thank you for contributing to TBsim!
