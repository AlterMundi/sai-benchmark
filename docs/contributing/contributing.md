# Contributing to SAI-Benchmark

Thank you for your interest in contributing to SAI-Benchmark! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Getting Started

### Ways to Contribute

- **🐛 Bug Reports**: Report issues and bugs
- **💡 Feature Requests**: Suggest new features or improvements
- **🔧 Code Contributions**: Implement features, fix bugs, improve performance
- **📖 Documentation**: Improve docs, write tutorials, add examples
- **🧪 Testing**: Add tests, improve test coverage, test on different platforms
- **🎨 Examples**: Create example configurations, scripts, and use cases

### Before You Start

1. **Check existing issues** to see if your contribution is already being worked on
2. **Read the documentation** to understand the project structure
3. **Join the discussion** on GitHub Issues or Discussions

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub
   git clone https://github.com/YOUR_USERNAME/sai-benchmark.git
   cd sai-benchmark
   ```

2. **Create Development Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Verify Setup**
   ```bash
   # Run validation tests
   python validate_tests.py
   
   # Run unit tests
   pytest tests/unit/ -v
   
   # Run code quality checks
   black --check .
   ruff check .
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

2. **Make Changes**
   - Write code following our [code style guidelines](#code-style)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run relevant tests
   pytest tests/ -v
   
   # Run code quality checks
   black .
   ruff check .
   
   # Test with different configurations
   python run_suite.py --suite tests/fixtures/test_suite.yaml
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

## Contributing Guidelines

### Types of Contributions

#### 🐛 Bug Fixes
- Fix existing functionality that doesn't work as expected
- Include test cases that reproduce the bug
- Update documentation if the bug affects documented behavior

#### 💡 New Features
- Add new capabilities to the framework
- Include comprehensive tests
- Update documentation and examples
- Consider backward compatibility

#### 📖 Documentation
- Improve existing documentation
- Add tutorials and guides
- Create example configurations
- Fix typos and clarify unclear sections

#### 🧪 Testing
- Add test cases for untested code
- Improve test coverage
- Add integration tests
- Performance testing

### Contribution Standards

#### Code Quality
- Follow Python PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and single-purpose

#### Testing Requirements
- All new features must include tests
- Bug fixes must include regression tests
- Aim for >85% test coverage
- Tests should be fast and reliable

#### Documentation Requirements
- Update relevant documentation for any changes
- Include docstrings for new public APIs
- Add examples for new features
- Update configuration guides if needed

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_prompt_registry.py
│   ├── test_model_registry.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_end_to_end.py
│   └── ...
├── fixtures/               # Test data and configurations
└── conftest.py             # Shared test fixtures
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest -m "not slow"                  # Exclude slow tests
pytest -m "not external"             # Exclude tests requiring external services

# Run with coverage
pytest --cov=core --cov=engines tests/

# Run specific test file
pytest tests/unit/test_prompt_registry.py -v

# Run specific test
pytest tests/unit/test_prompt_registry.py::TestPromptRegistry::test_register_prompt -v
```

### Writing Tests

#### Unit Test Example

```python
import pytest
from core.prompt_registry import PromptRegistry, PromptTemplate, OutputSchema

class TestPromptRegistry:
    """Test PromptRegistry functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return PromptRegistry()
    
    def test_register_prompt(self, registry):
        """Test registering a new prompt."""
        schema = OutputSchema(type="json", format={"result": {"type": "string"}})
        prompt = PromptTemplate(
            id="test_prompt",
            name="Test Prompt",
            description="A test prompt",
            template="Test template",
            output_schema=schema,
            tags=["test"]
        )
        
        registry.register_prompt(prompt)
        
        assert "test_prompt" in registry.prompts
        assert registry.prompts["test_prompt"].name == "Test Prompt"
```

#### Integration Test Example

```python
@pytest.mark.integration
def test_complete_workflow(temp_workspace, mock_registries):
    """Test complete workflow from configuration to results."""
    # Create test suite
    suite_config = create_test_suite_config()
    
    # Run test
    runner = TestSuiteRunner(**mock_registries)
    result = runner.run_suite(suite_config)
    
    # Verify results
    assert result.suite_name == "test_suite"
    assert len(result.test_results) > 0
    assert "accuracy" in result.aggregated_metrics
```

### Test Guidelines

- **Use descriptive test names** that explain what is being tested
- **Test one thing at a time** - keep tests focused
- **Use fixtures** for common setup code
- **Mock external dependencies** (APIs, file systems, etc.)
- **Test edge cases** and error conditions
- **Make tests deterministic** - avoid random data unless testing randomness

## Documentation

### Documentation Standards

- **Use clear, concise language**
- **Include code examples** for all features
- **Provide context** - explain why, not just how
- **Keep examples realistic** and practical
- **Update documentation** with any code changes

### Documentation Types

#### API Documentation
- Docstrings for all public functions and classes
- Type hints for parameters and return values
- Usage examples in docstrings

```python
def register_prompt(self, prompt: PromptTemplate) -> None:
    """Register a new prompt template.
    
    Args:
        prompt: The prompt template to register
        
    Raises:
        ValueError: If prompt ID already exists
        
    Example:
        >>> registry = PromptRegistry()
        >>> prompt = PromptTemplate(id="test", name="Test", ...)
        >>> registry.register_prompt(prompt)
    """
```

#### User Guides
- Step-by-step instructions
- Real-world examples
- Common use cases
- Troubleshooting tips

#### Tutorials
- Complete walkthroughs
- Progressive complexity
- Copy-paste examples
- Expected outputs

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use type hints
def process_image(image_path: Path, model_id: str) -> Dict[str, Any]:
    """Process an image with the specified model."""
    pass

# Use descriptive variable names
user_prompt_template = registry.get_prompt("user_query")
model_response_time = engine.measure_latency()

# Keep lines under 100 characters
very_long_function_call_with_many_parameters(
    parameter_one="value_one",
    parameter_two="value_two",
    parameter_three="value_three"
)

# Use docstrings for classes and public methods
class PromptRegistry:
    """Centralized management of prompt templates.
    
    The PromptRegistry provides a unified interface for storing,
    retrieving, and validating prompt templates used across
    different vision-language models.
    """
```

### Code Quality Tools

We use the following tools to maintain code quality:

```bash
# Format code
black .

# Check code style
ruff check .

# Type checking
mypy core/ engines/

# Import sorting
isort .
```

### Configuration Files

**pyproject.toml**
```toml
[tool.black]
line-length = 100
target-version = ['py38']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "C", "N"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
   ```bash
   pytest tests/
   ```

2. **Run code quality checks**
   ```bash
   black .
   ruff check .
   ```

3. **Update documentation** if needed

4. **Add changelog entry** if applicable

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description explains the change

### PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Documentation updated
- [ ] Examples added/updated
- [ ] Changelog updated

## Additional Notes
Any additional information about the change
```

### Review Process

1. **Automated checks** run on all PRs
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review** if applicable
5. **Approval and merge** by maintainers

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check docs first for common questions

### Communication Guidelines

- **Be respectful** and constructive
- **Search existing issues** before creating new ones
- **Provide context** and examples when asking for help
- **Use clear titles** and descriptions

### Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation credits** for doc contributions

## Additional Resources

- **[Development Setup Guide](development.md)** - Detailed setup instructions
- **[Testing Guide](testing.md)** - Comprehensive testing documentation
- **[Code Style Guide](code-style.md)** - Detailed style guidelines
- **[Architecture Overview](../architecture.md)** - Framework design principles
- **[API Documentation](../api/core.md)** - Complete API reference

---

Thank you for contributing to SAI-Benchmark! Your contributions help make vision-language model evaluation more accessible and reliable for everyone.