"""
Pytest configuration and shared fixtures for SAI-Benchmark tests.

This module provides common fixtures and configuration for all tests,
including mock objects, test data generators, and utility functions.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, Mock

import pytest
from faker import Faker
from PIL import Image

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize Faker for test data generation
fake = Faker()


# --- Directory and File Management Fixtures ---

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample test image and return its path."""
    image_path = temp_dir / "test_image.jpg"
    # Create a simple test image (100x100 red square)
    img = Image.new('RGB', (100, 100), color='red')
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_images(temp_dir: Path) -> List[Path]:
    """Create multiple sample test images."""
    images = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for i, color in enumerate(colors):
        image_path = temp_dir / f"test_image_{i}.jpg"
        img = Image.new('RGB', (100, 100), color=color)
        img.save(image_path)
        images.append(image_path)
    return images


# --- Mock Objects and Test Data ---

@pytest.fixture
def mock_bbox() -> List[float]:
    """Generate a mock bounding box in normalized format [x_center, y_center, width, height]."""
    return [0.5, 0.5, 0.2, 0.3]


@pytest.fixture
def mock_detection_result() -> Dict[str, Any]:
    """Generate a mock detection result."""
    return {
        "has_smoke": True,
        "confidence": 0.85,
        "bbox": [0.5, 0.5, 0.2, 0.3],
        "timestamp": fake.date_time_this_year().isoformat(),
    }


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Generate a mock model configuration."""
    return {
        "name": "test-model",
        "version": "1.0",
        "engines": ["ollama", "hf"],
        "capabilities": {
            "max_tokens": 1024,
            "supports_streaming": True,
            "supports_json": True,
        },
        "default_params": {
            "temperature": 0.1,
            "max_tokens": 512,
        }
    }


@pytest.fixture
def mock_prompt_template() -> Dict[str, Any]:
    """Generate a mock prompt template configuration."""
    return {
        "id": "test_prompt",
        "version": "1.0",
        "template": "Analyze this image and detect {object}. Response format: {format}",
        "variables": ["object", "format"],
        "metadata": {
            "author": "test",
            "created": "2024-01-01",
            "description": "Test prompt for object detection",
        }
    }


@pytest.fixture
def mock_engine_response() -> Dict[str, Any]:
    """Generate a mock engine response."""
    return {
        "content": json.dumps({"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.3]}),
        "model": "test-model",
        "latency": 0.123,
        "tokens_used": 150,
        "metadata": {
            "engine": "mock",
            "timestamp": fake.date_time_this_year().isoformat(),
        }
    }


# --- Engine and Model Mocks ---

@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    client = MagicMock()
    client.generate.return_value = {
        "response": json.dumps({"has_smoke": False, "bbox": None}),
        "model": "qwen2.5-vl:7b",
        "created_at": "2024-01-01T00:00:00Z",
        "total_duration": 1_000_000_000,  # 1 second in nanoseconds
    }
    return client


@pytest.fixture
def mock_hf_model():
    """Create a mock HuggingFace model."""
    model = MagicMock()
    processor = MagicMock()
    
    # Mock model output
    model.generate.return_value = MagicMock()
    processor.decode.return_value = json.dumps({"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.3]})
    
    return model, processor


# --- Test Suite Configuration ---

@pytest.fixture
def sample_test_suite_config() -> Dict[str, Any]:
    """Generate a sample test suite configuration."""
    return {
        "name": "test_suite",
        "version": "1.0",
        "description": "Sample test suite for testing",
        "tests": [
            {
                "name": "smoke_detection_test",
                "models": ["qwen2.5-vl:7b"],
                "prompts": ["early_fire_json"],
                "engines": ["ollama"],
                "dataset": "test_sequences/smoke",
                "metrics": ["precision", "recall", "f1"],
            }
        ],
        "global_config": {
            "max_tokens": 512,
            "temperature": 0.1,
            "iou_threshold": 0.4,
        }
    }


# --- Utility Functions ---

@pytest.fixture
def create_test_sequence(temp_dir: Path):
    """Factory fixture to create test image sequences with ground truth."""
    def _create_sequence(
        name: str,
        num_images: int = 5,
        has_smoke: bool = False,
        bbox: List[float] = None
    ) -> Path:
        sequence_dir = temp_dir / name
        sequence_dir.mkdir(exist_ok=True)
        
        # Create images
        for i in range(num_images):
            img_path = sequence_dir / f"frame_{i:04d}.jpg"
            img = Image.new('RGB', (640, 480), color=(i * 50, 0, 0))
            img.save(img_path)
            
            # Create ground truth file
            gt_path = sequence_dir / f"frame_{i:04d}.txt"
            if has_smoke and bbox:
                gt_path.write_text(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
            else:
                gt_path.touch()  # Empty file for no smoke
        
        return sequence_dir
    
    return _create_sequence


# --- Environment and Resource Mocks ---

@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda x: "Mock GPU")


@pytest.fixture
def mock_no_gpu(monkeypatch):
    """Mock no GPU available."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)


# --- Async Support ---

@pytest.fixture
def async_mock():
    """Create an async mock object."""
    import asyncio
    
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)
    
    return AsyncMock


# --- Cleanup and Safety ---

@pytest.fixture(autouse=True)
def cleanup_test_outputs(request):
    """Automatically cleanup test output files after each test."""
    yield
    # Cleanup logic here if needed
    # For example, remove any test output files created during tests
    test_output_dir = Path("test_outputs")
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir, ignore_errors=True)


@pytest.fixture
def isolated_env(monkeypatch, temp_dir):
    """Create an isolated environment for tests."""
    # Change working directory to temp dir
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    # Clear environment variables that might affect tests
    env_vars_to_clear = ["HF_HOME", "OLLAMA_HOST", "SAI_CONFIG"]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    
    yield temp_dir
    
    # Restore original working directory
    os.chdir(original_cwd)


# --- Parametrized Test Helpers ---

def pytest_generate_tests(metafunc):
    """Generate parametrized tests based on markers."""
    if "model_name" in metafunc.fixturenames:
        models = ["qwen2.5-vl:7b", "llama3.2-vision:11b", "minicpm-v:8b"]
        metafunc.parametrize("model_name", models)
    
    if "engine_type" in metafunc.fixturenames:
        engines = ["ollama", "hf"]
        metafunc.parametrize("engine_type", engines)