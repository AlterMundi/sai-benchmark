"""
Unit tests for Resource Manager system.

Tests cover:
- ResourceType enum and ResourceRequirement dataclass
- ResourcePool resource allocation logic
- ResourceManager system resource detection
- Resource allocation and conflict resolution
- GPU/CPU/Memory management
- Ollama service coordination
- Thread-safe resource locking
"""

import pytest
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.resource_manager import (
    ResourceManager, ResourcePool, ResourceRequirement, ResourceType, ResourceError
)
from core.model_registry import ModelConfig, ModelCapability, EngineType


class TestResourceType:
    """Test ResourceType enum."""
    
    def test_resource_type_values(self):
        """Test that all expected resource types are defined."""
        expected_types = [
            "gpu_memory", "cpu_cores", "system_memory", 
            "ollama_service", "network_bandwidth"
        ]
        
        actual_values = [rt.value for rt in ResourceType]
        for expected in expected_types:
            assert expected in actual_values


class TestResourceRequirement:
    """Test ResourceRequirement dataclass."""
    
    def test_resource_requirement_creation(self):
        """Test creating ResourceRequirement with all fields."""
        req = ResourceRequirement(
            resource_type=ResourceType.GPU_MEMORY,
            amount=8.0,
            exclusive=True
        )
        
        assert req.resource_type == ResourceType.GPU_MEMORY
        assert req.amount == 8.0
        assert req.exclusive is True
    
    def test_resource_requirement_defaults(self):
        """Test ResourceRequirement with default values."""
        req = ResourceRequirement(
            resource_type=ResourceType.CPU_CORES,
            amount=4.0
        )
        
        assert req.resource_type == ResourceType.CPU_CORES
        assert req.amount == 4.0
        assert req.exclusive is False  # Default


class TestResourcePool:
    """Test ResourcePool functionality."""
    
    def test_resource_pool_creation(self):
        """Test creating ResourcePool with all resources."""
        pool = ResourcePool(
            gpu_memory_gb=16.0,
            cpu_cores=8,
            system_memory_gb=32.0,
            ollama_slots=2,
            network_slots=20
        )
        
        assert pool.gpu_memory_gb == 16.0
        assert pool.cpu_cores == 8
        assert pool.system_memory_gb == 32.0
        assert pool.ollama_slots == 2
        assert pool.network_slots == 20
    
    def test_resource_pool_defaults(self):
        """Test ResourcePool with default values."""
        pool = ResourcePool()
        
        assert pool.gpu_memory_gb == 0.0
        assert pool.cpu_cores == 0
        assert pool.system_memory_gb == 0.0
        assert pool.ollama_slots == 1
        assert pool.network_slots == 10
    
    def test_can_allocate_sufficient_resources(self):
        """Test allocation check with sufficient resources."""
        pool = ResourcePool(
            gpu_memory_gb=16.0,
            cpu_cores=8,
            system_memory_gb=32.0,
            ollama_slots=1,
            network_slots=10
        )
        
        requirements = [
            ResourceRequirement(ResourceType.GPU_MEMORY, 8.0),
            ResourceRequirement(ResourceType.CPU_CORES, 4.0),
            ResourceRequirement(ResourceType.SYSTEM_MEMORY, 16.0),
            ResourceRequirement(ResourceType.OLLAMA_SERVICE, 1.0),
            ResourceRequirement(ResourceType.NETWORK_BANDWIDTH, 5.0)
        ]
        
        assert pool.can_allocate(requirements) is True
    
    def test_can_allocate_insufficient_gpu_memory(self):
        """Test allocation check with insufficient GPU memory."""
        pool = ResourcePool(gpu_memory_gb=4.0)
        
        requirements = [
            ResourceRequirement(ResourceType.GPU_MEMORY, 8.0)
        ]
        
        assert pool.can_allocate(requirements) is False
    
    def test_can_allocate_insufficient_cpu_cores(self):
        """Test allocation check with insufficient CPU cores."""
        pool = ResourcePool(cpu_cores=2)
        
        requirements = [
            ResourceRequirement(ResourceType.CPU_CORES, 4.0)
        ]
        
        assert pool.can_allocate(requirements) is False
    
    def test_can_allocate_insufficient_system_memory(self):
        """Test allocation check with insufficient system memory."""
        pool = ResourcePool(system_memory_gb=8.0)
        
        requirements = [
            ResourceRequirement(ResourceType.SYSTEM_MEMORY, 16.0)
        ]
        
        assert pool.can_allocate(requirements) is False
    
    def test_can_allocate_ollama_service_unavailable(self):
        """Test allocation check with Ollama service unavailable."""
        pool = ResourcePool(ollama_slots=0)
        
        requirements = [
            ResourceRequirement(ResourceType.OLLAMA_SERVICE, 1.0)
        ]
        
        assert pool.can_allocate(requirements) is False
    
    def test_can_allocate_insufficient_network_bandwidth(self):
        """Test allocation check with insufficient network bandwidth."""
        pool = ResourcePool(network_slots=5)
        
        requirements = [
            ResourceRequirement(ResourceType.NETWORK_BANDWIDTH, 10.0)
        ]
        
        assert pool.can_allocate(requirements) is False
    
    def test_can_allocate_empty_requirements(self):
        """Test allocation check with empty requirements."""
        pool = ResourcePool()
        
        assert pool.can_allocate([]) is True
    
    def test_can_allocate_multiple_requirements(self):
        """Test allocation check with multiple requirements."""
        pool = ResourcePool(
            gpu_memory_gb=16.0,
            cpu_cores=8,
            system_memory_gb=32.0
        )
        
        # Test exact match
        requirements = [
            ResourceRequirement(ResourceType.GPU_MEMORY, 16.0),
            ResourceRequirement(ResourceType.CPU_CORES, 8.0),
            ResourceRequirement(ResourceType.SYSTEM_MEMORY, 32.0)
        ]
        
        assert pool.can_allocate(requirements) is True
        
        # Test exceeding limits
        requirements[0].amount = 17.0  # Exceeds GPU memory
        assert pool.can_allocate(requirements) is False


class TestResourceManager:
    """Test ResourceManager functionality."""
    
    @pytest.fixture
    def mock_system_resources(self):
        """Mock system resource detection."""
        with patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch.object(ResourceManager, '_detect_gpu_memory', return_value=16.0), \
             patch.object(ResourceManager, '_check_ollama_available', return_value=True):
            
            mock_memory.return_value.total = 32 * (1024**3)  # 32GB in bytes
            yield
    
    def test_resource_manager_initialization(self, mock_system_resources):
        """Test ResourceManager initialization with system detection."""
        manager = ResourceManager()
        
        assert manager._resource_pool.cpu_cores == 8
        assert manager._resource_pool.system_memory_gb == 32.0
        assert manager._resource_pool.gpu_memory_gb == 16.0
        assert manager._resource_pool.ollama_slots == 1
        assert len(manager._allocated_resources) == 0
        assert manager._lock is not None
        assert manager._gpu_lock is not None
        assert manager._ollama_lock is not None
    
    def test_detect_gpu_memory_nvidia_smi(self):
        """Test GPU memory detection via nvidia-smi."""
        manager = ResourceManager()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "16384\n8192\n"  # Two GPUs
            
            gpu_memory = manager._detect_gpu_memory()
            
            assert gpu_memory == 16.384  # First GPU in GB
            mock_run.assert_called_with(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
    
    def test_detect_gpu_memory_torch_cuda(self):
        """Test GPU memory detection via torch.cuda."""
        manager = ResourceManager()
        
        with patch('subprocess.run', side_effect=Exception("nvidia-smi not found")), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 8 * (1024**3)  # 8GB in bytes
            
            gpu_memory = manager._detect_gpu_memory()
            
            assert gpu_memory == 8.0
    
    def test_detect_gpu_memory_no_gpu(self):
        """Test GPU memory detection when no GPU available."""
        manager = ResourceManager()
        
        with patch('subprocess.run', side_effect=Exception("nvidia-smi not found")), \
             patch('torch.cuda.is_available', return_value=False):
            
            gpu_memory = manager._detect_gpu_memory()
            
            assert gpu_memory == 0.0
    
    def test_check_ollama_available_success(self):
        """Test Ollama availability check when service is running."""
        manager = ResourceManager()
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            available = manager._check_ollama_available()
            
            assert available is True
            mock_get.assert_called_with('http://localhost:11434/api/tags', timeout=5)
    
    def test_check_ollama_available_failure(self):
        """Test Ollama availability check when service is down."""
        manager = ResourceManager()
        
        with patch('requests.get', side_effect=Exception("Connection failed")):
            available = manager._check_ollama_available()
            
            assert available is False
    
    def test_calculate_model_requirements_qwen_gpu(self, mock_system_resources):
        """Test calculating requirements for Qwen model on GPU."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="qwen2.5-vl-7b-hf",
            name="Qwen 2.5 VL 7B",
            engine=EngineType.HUGGINGFACE,
            model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            capabilities={ModelCapability.VISION},
            gpu_memory_gb=14.0
        )
        
        requirements = manager.calculate_model_requirements(model_config)
        
        assert len(requirements) == 2
        
        # Check GPU memory requirement
        gpu_req = next(req for req in requirements if req.resource_type == ResourceType.GPU_MEMORY)
        assert gpu_req.amount == 14.0
        assert gpu_req.exclusive is True
        
        # Check system memory requirement  
        mem_req = next(req for req in requirements if req.resource_type == ResourceType.SYSTEM_MEMORY)
        assert mem_req.amount == 4.0  # Base requirement
    
    def test_calculate_model_requirements_ollama(self, mock_system_resources):
        """Test calculating requirements for Ollama model."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="qwen2.5-vl:7b",
            name="Qwen 2.5 VL 7B Ollama",
            engine=EngineType.OLLAMA,
            model_path="qwen2.5-vl:7b",
            capabilities={ModelCapability.VISION}
        )
        
        requirements = manager.calculate_model_requirements(model_config)
        
        assert len(requirements) == 2
        
        # Check Ollama service requirement
        ollama_req = next(req for req in requirements if req.resource_type == ResourceType.OLLAMA_SERVICE)
        assert ollama_req.amount == 1.0
        assert ollama_req.exclusive is True
        
        # Check system memory requirement
        mem_req = next(req for req in requirements if req.resource_type == ResourceType.SYSTEM_MEMORY)
        assert mem_req.amount == 8.0  # Ollama models need more RAM
    
    def test_calculate_model_requirements_openai(self, mock_system_resources):
        """Test calculating requirements for OpenAI model."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="gpt-4o",
            name="GPT-4 Omni",
            engine=EngineType.OPENAI,
            model_path="gpt-4o",
            capabilities={ModelCapability.VISION}
        )
        
        requirements = manager.calculate_model_requirements(model_config)
        
        assert len(requirements) == 1
        
        # Check network bandwidth requirement
        net_req = next(req for req in requirements if req.resource_type == ResourceType.NETWORK_BANDWIDTH)
        assert net_req.amount == 1.0
        assert net_req.exclusive is False
    
    def test_can_allocate_model_success(self, mock_system_resources):
        """Test successful model allocation check."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="small_model",
            name="Small Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.TEXT},
            gpu_memory_gb=4.0
        )
        
        can_allocate = manager.can_allocate_model(model_config)
        
        assert can_allocate is True
    
    def test_can_allocate_model_insufficient_resources(self, mock_system_resources):
        """Test model allocation check with insufficient resources."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="large_model",
            name="Large Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.VISION},
            gpu_memory_gb=32.0  # Exceeds available 16GB
        )
        
        can_allocate = manager.can_allocate_model(model_config)
        
        assert can_allocate is False
    
    def test_allocate_model_success(self, mock_system_resources):
        """Test successful model resource allocation."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="test_model",
            name="Test Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.TEXT},
            gpu_memory_gb=8.0
        )
        
        # Test context manager allocation
        with manager.allocate_model(model_config) as allocation_id:
            assert allocation_id in manager._allocated_resources
            assert len(manager._allocated_resources[allocation_id]) > 0
        
        # After context exit, resources should be freed
        assert allocation_id not in manager._allocated_resources
    
    def test_allocate_model_conflict(self, mock_system_resources):
        """Test model allocation with resource conflict."""
        manager = ResourceManager()
        
        # Create a large model that uses most GPU memory
        large_model = ModelConfig(
            id="large_model",
            name="Large Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.VISION},
            gpu_memory_gb=15.0
        )
        
        small_model = ModelConfig(
            id="small_model",
            name="Small Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.VISION},
            gpu_memory_gb=4.0
        )
        
        # Allocate large model first
        with manager.allocate_model(large_model):
            # Try to allocate small model (should fail due to insufficient remaining GPU memory)
            with pytest.raises(ResourceError, match="Cannot allocate model"):
                with manager.allocate_model(small_model):
                    pass
    
    def test_allocate_model_timeout(self, mock_system_resources):
        """Test model allocation timeout."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="timeout_model",
            name="Timeout Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.VISION},
            gpu_memory_gb=16.0  # Uses all GPU memory
        )
        
        # Allocate model with long timeout, then try to allocate another with short timeout
        with manager.allocate_model(model_config):
            start_time = time.time()
            
            with pytest.raises(ResourceError, match="Timeout waiting for resources"):
                with manager.allocate_model(model_config, timeout=0.1):  # Short timeout
                    pass
            
            elapsed = time.time() - start_time
            assert elapsed >= 0.1  # Should have waited at least the timeout period
    
    def test_thread_safety(self, mock_system_resources):
        """Test thread-safe resource allocation."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="concurrent_model",
            name="Concurrent Model",
            engine=EngineType.OLLAMA,
            model_path="test",
            capabilities={ModelCapability.TEXT}
        )
        
        results = []
        errors = []
        
        def allocate_model():
            try:
                with manager.allocate_model(model_config):
                    time.sleep(0.1)  # Hold resources briefly
                    results.append(threading.current_thread().ident)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads trying to allocate the same Ollama model
        threads = []
        for _ in range(5):
            t = threading.Thread(target=allocate_model)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Only one thread should succeed (Ollama exclusive), others should error
        assert len(results) == 1
        assert len(errors) == 4
        assert all(isinstance(e, ResourceError) for e in errors)
    
    def test_get_resource_stats(self, mock_system_resources):
        """Test resource statistics reporting."""
        manager = ResourceManager()
        
        stats = manager.get_resource_stats()
        
        assert "total_resources" in stats
        assert "available_resources" in stats
        assert "allocated_resources" in stats
        assert "utilization" in stats
        
        assert stats["total_resources"]["gpu_memory_gb"] == 16.0
        assert stats["total_resources"]["cpu_cores"] == 8
        assert stats["total_resources"]["system_memory_gb"] == 32.0
        
        # Initially, available should equal total
        assert stats["available_resources"]["gpu_memory_gb"] == 16.0
        assert stats["allocated_resources"]["gpu_memory_gb"] == 0.0
        assert stats["utilization"]["gpu_memory"] == 0.0
    
    def test_get_resource_stats_with_allocation(self, mock_system_resources):
        """Test resource statistics with active allocations."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="stats_model",
            name="Stats Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.VISION},
            gpu_memory_gb=8.0
        )
        
        with manager.allocate_model(model_config):
            stats = manager.get_resource_stats()
            
            assert stats["allocated_resources"]["gpu_memory_gb"] == 8.0
            assert stats["available_resources"]["gpu_memory_gb"] == 8.0  # 16 - 8
            assert stats["utilization"]["gpu_memory"] == 0.5  # 8/16
    
    def test_cleanup(self, mock_system_resources):
        """Test resource cleanup functionality."""
        manager = ResourceManager()
        
        model_config = ModelConfig(
            id="cleanup_model",
            name="Cleanup Model",
            engine=EngineType.HUGGINGFACE,
            model_path="test",
            capabilities={ModelCapability.TEXT},
            gpu_memory_gb=4.0
        )
        
        # Allocate resources manually (simulating incomplete cleanup)
        allocation_id = "test_allocation"
        requirements = manager.calculate_model_requirements(model_config)
        manager._allocated_resources[allocation_id] = requirements
        
        assert allocation_id in manager._allocated_resources
        
        # Cleanup specific allocation
        manager.cleanup(allocation_id)
        
        assert allocation_id not in manager._allocated_resources
    
    def test_cleanup_all(self, mock_system_resources):
        """Test cleanup of all resources."""
        manager = ResourceManager()
        
        # Simulate multiple allocations
        manager._allocated_resources["alloc1"] = [ResourceRequirement(ResourceType.GPU_MEMORY, 4.0)]
        manager._allocated_resources["alloc2"] = [ResourceRequirement(ResourceType.CPU_CORES, 2.0)]
        
        assert len(manager._allocated_resources) == 2
        
        # Cleanup all
        manager.cleanup()
        
        assert len(manager._allocated_resources) == 0


# Integration tests for resource management
@pytest.mark.integration
class TestResourceManagerIntegration:
    """Integration tests for ResourceManager with real system resources."""
    
    def test_real_system_detection(self):
        """Test detection with real system resources."""
        manager = ResourceManager()
        
        # Basic sanity checks
        assert manager._resource_pool.cpu_cores > 0
        assert manager._resource_pool.system_memory_gb > 0
        
        # GPU may or may not be available
        assert manager._resource_pool.gpu_memory_gb >= 0
        
        # Ollama may or may not be running
        assert manager._resource_pool.ollama_slots >= 0
    
    @pytest.mark.skipif(not Path("/usr/bin/nvidia-smi").exists(), 
                        reason="nvidia-smi not available")
    def test_nvidia_gpu_detection(self):
        """Test NVIDIA GPU detection on systems with nvidia-smi."""
        manager = ResourceManager()
        
        gpu_memory = manager._detect_gpu_memory()
        
        # Should detect some GPU memory if nvidia-smi is available
        assert gpu_memory > 0