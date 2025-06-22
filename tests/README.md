# SAI-Benchmark Test Suite

This directory contains the comprehensive test suite for the SAI-Benchmark framework.

## Structure

```
tests/
├── README.md           # This file
├── conftest.py         # Shared pytest fixtures and configuration
├── unit/               # Unit tests for individual components
│   ├── __init__.py
│   ├── test_prompt_registry.py    # Tests for PromptRegistry
│   ├── test_model_registry.py     # Tests for ModelRegistry
│   ├── test_engine_registry.py    # Tests for EngineRegistry
│   └── test_metrics_registry.py   # Tests for MetricsRegistry
├── integration/        # Integration tests (TODO)
└── benchmark/          # Performance benchmark tests (TODO)
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Or with pytest directly
pytest tests/

# Run specific test file
pytest tests/unit/test_prompt_registry.py

# Run with coverage
pytest tests/ --cov=core --cov=engines --cov-report=html

# Run tests in parallel
pytest tests/ -n auto
```

### Test Markers

Tests are marked with categories for selective execution:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring GPU
pytest -m "not gpu"

# Skip tests requiring external services
pytest -m "not external"
```

## Writing Tests

### Test Structure

Each test module follows this pattern:

```python
"""
Unit tests for ComponentName.

Tests cover:
- Feature 1
- Feature 2
- Error handling
- Edge cases
"""

import pytest
from module.to.test import ComponentName


class TestComponentName:
    """Test ComponentName functionality."""
    
    @pytest.fixture
    def component(self):
        """Create a test instance."""
        return ComponentName()
    
    def test_feature_one(self, component):
        """Test feature one works correctly."""
        result = component.do_something()
        assert result == expected_value
```

### Using Fixtures

Common fixtures are available in `conftest.py`:

- `temp_dir`: Temporary directory (auto-cleaned)
- `sample_image_path`: Test image file
- `mock_model_config`: Mock model configuration
- `mock_engine_response`: Mock engine response
- `create_test_sequence`: Factory for image sequences

### Testing Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Test names should describe what they test
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Edge Cases**: Include boundary and error conditions
6. **Performance**: Mark slow tests with `@pytest.mark.slow`

### Example Test Patterns

#### Testing with Mocks
```python
def test_external_service(self, mocker):
    mock_response = mocker.patch('requests.get')
    mock_response.return_value.status_code = 200
    
    result = function_using_requests()
    assert result == expected
```

#### Testing Exceptions
```python
def test_invalid_input_raises_error(self):
    with pytest.raises(ValueError, match="Invalid input"):
        function_with_validation("bad input")
```

#### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
    ("test3", "result3"),
])
def test_multiple_cases(self, input, expected):
    assert process(input) == expected
```

## Coverage Goals

We aim for:
- **Unit Test Coverage**: >80% for core modules
- **Critical Path Coverage**: 100% for registries and engines
- **Integration Coverage**: Key workflows tested end-to-end

## CI/CD Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

## Future Enhancements

- [ ] Add integration tests for full workflows
- [ ] Add performance benchmark tests
- [ ] Add mutation testing
- [ ] Add visual regression tests for UI components
- [ ] Add contract tests for API compatibility