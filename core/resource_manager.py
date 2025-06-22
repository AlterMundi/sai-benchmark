"""
Resource Manager

Manages hardware resources and scheduling to prevent conflicts between engines.
Implements resource-aware parallel execution with GPU memory management.
"""

import threading
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import psutil
import subprocess

from .model_registry import ModelConfig, EngineType


class ResourceType(Enum):
    """Types of system resources"""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    OLLAMA_SERVICE = "ollama_service"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ResourceRequirement:
    """Resource requirements for a model/engine"""
    resource_type: ResourceType
    amount: float  # Amount required (GB for memory, cores for CPU, etc.)
    exclusive: bool = False  # Whether resource needs exclusive access


@dataclass
class ResourcePool:
    """Available system resources"""
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    system_memory_gb: float = 0.0
    ollama_slots: int = 1  # Ollama typically handles one model at a time
    network_slots: int = 10  # Concurrent API calls allowed
    
    def can_allocate(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if requirements can be satisfied"""
        for req in requirements:
            if req.resource_type == ResourceType.GPU_MEMORY and req.amount > self.gpu_memory_gb:
                return False
            elif req.resource_type == ResourceType.CPU_CORES and req.amount > self.cpu_cores:
                return False
            elif req.resource_type == ResourceType.SYSTEM_MEMORY and req.amount > self.system_memory_gb:
                return False
            elif req.resource_type == ResourceType.OLLAMA_SERVICE and req.amount > self.ollama_slots:
                return False
            elif req.resource_type == ResourceType.NETWORK_BANDWIDTH and req.amount > self.network_slots:
                return False
        return True


class ResourceManager:
    """Manages system resources and coordinates parallel execution"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._allocated_resources: Dict[str, List[ResourceRequirement]] = {}
        self._resource_pool = self._detect_system_resources()
        self._gpu_lock = threading.Lock()  # Exclusive GPU access
        self._ollama_lock = threading.Lock()  # Exclusive Ollama access
        
    def _detect_system_resources(self) -> ResourcePool:
        """Detect available system resources"""
        # CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        
        # System memory
        memory = psutil.virtual_memory()
        system_memory_gb = memory.total / (1024**3)
        
        # GPU memory detection
        gpu_memory_gb = self._detect_gpu_memory()
        
        # Ollama availability
        ollama_slots = 1 if self._check_ollama_available() else 0
        
        return ResourcePool(
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory_gb,
            ollama_slots=ollama_slots,
            network_slots=10  # Default for API calls
        )
    
    def _detect_gpu_memory(self) -> float:
        """Detect available GPU memory"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Get first GPU memory in MB, convert to GB
                memory_mb = int(result.stdout.strip().split('\n')[0])
                return memory_mb / 1024
        except:
            pass
        
        try:
            # Try PyTorch detection
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        return 0.0  # No GPU detected
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_model_requirements(self, model_config: ModelConfig) -> List[ResourceRequirement]:
        """Get resource requirements for a model"""
        requirements = []
        
        if model_config.engine == EngineType.HUGGINGFACE:
            # HuggingFace models need GPU memory (exclusive)
            gpu_memory = model_config.gpu_memory_gb or 8.0  # Default estimate
            requirements.append(ResourceRequirement(
                resource_type=ResourceType.GPU_MEMORY,
                amount=gpu_memory,
                exclusive=True  # GPU models need exclusive access
            ))
            
        elif model_config.engine == EngineType.OLLAMA:
            # Ollama models need exclusive service access
            requirements.append(ResourceRequirement(
                resource_type=ResourceType.OLLAMA_SERVICE,
                amount=1,
                exclusive=True  # Ollama typically loads one model at a time
            ))
            
            # Also some system memory
            requirements.append(ResourceRequirement(
                resource_type=ResourceType.SYSTEM_MEMORY,
                amount=4.0,  # Estimate for quantized models
                exclusive=False
            ))
            
        elif model_config.engine in [EngineType.OPENAI, EngineType.ANTHROPIC, EngineType.GOOGLE]:
            # API-based models just need network slots
            requirements.append(ResourceRequirement(
                resource_type=ResourceType.NETWORK_BANDWIDTH,
                amount=1,
                exclusive=False
            ))
        
        return requirements
    
    @contextmanager
    def acquire_resources(self, model_config: ModelConfig, task_id: str):
        """Context manager to acquire resources for a model"""
        requirements = self.get_model_requirements(model_config)
        
        # Determine which locks we need
        locks_needed = []
        if any(req.resource_type == ResourceType.GPU_MEMORY for req in requirements):
            locks_needed.append(self._gpu_lock)
        if any(req.resource_type == ResourceType.OLLAMA_SERVICE for req in requirements):
            locks_needed.append(self._ollama_lock)
        
        # Acquire locks in order to prevent deadlock
        acquired_locks = []
        try:
            for lock in locks_needed:
                lock.acquire()
                acquired_locks.append(lock)
            
            # Check if resources are available
            with self._lock:
                if not self._can_allocate_requirements(requirements):
                    raise ResourceError(f"Insufficient resources for model {model_config.id}")
                
                # Allocate resources
                self._allocated_resources[task_id] = requirements
                self._update_available_resources(requirements, allocate=True)
            
            yield
            
        finally:
            # Release resources
            with self._lock:
                if task_id in self._allocated_resources:
                    requirements = self._allocated_resources[task_id]
                    self._update_available_resources(requirements, allocate=False)
                    del self._allocated_resources[task_id]
            
            # Release locks in reverse order
            for lock in reversed(acquired_locks):
                lock.release()
    
    def _can_allocate_requirements(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if requirements can be satisfied with current availability"""
        # For exclusive resources, check if any are currently allocated
        for req in requirements:
            if req.exclusive:
                # Check if any task is using this exclusive resource type
                for allocated_reqs in self._allocated_resources.values():
                    for allocated_req in allocated_reqs:
                        if (allocated_req.resource_type == req.resource_type and 
                            allocated_req.exclusive):
                            return False
        
        return self._resource_pool.can_allocate(requirements)
    
    def _update_available_resources(self, requirements: List[ResourceRequirement], allocate: bool):
        """Update available resource counts"""
        multiplier = -1 if allocate else 1
        
        for req in requirements:
            if req.resource_type == ResourceType.GPU_MEMORY:
                self._resource_pool.gpu_memory_gb += multiplier * req.amount
            elif req.resource_type == ResourceType.CPU_CORES:
                self._resource_pool.cpu_cores += multiplier * int(req.amount)
            elif req.resource_type == ResourceType.SYSTEM_MEMORY:
                self._resource_pool.system_memory_gb += multiplier * req.amount
            elif req.resource_type == ResourceType.OLLAMA_SERVICE:
                self._resource_pool.ollama_slots += multiplier * int(req.amount)
            elif req.resource_type == ResourceType.NETWORK_BANDWIDTH:
                self._resource_pool.network_slots += multiplier * int(req.amount)
    
    def get_optimal_parallelism(self, model_configs: List[ModelConfig]) -> int:
        """Determine optimal number of parallel workers for given models"""
        
        # Separate models by resource requirements
        gpu_models = [m for m in model_configs if m.engine == EngineType.HUGGINGFACE]
        ollama_models = [m for m in model_configs if m.engine == EngineType.OLLAMA]
        api_models = [m for m in model_configs if m.engine in [EngineType.OPENAI, EngineType.ANTHROPIC]]
        
        # GPU models: only 1 at a time
        if gpu_models:
            return 1
        
        # Ollama models: only 1 at a time
        if ollama_models:
            return 1
        
        # API models: can run many in parallel
        if api_models:
            return min(self._resource_pool.network_slots, 8)  # Cap at 8 for rate limiting
        
        # Mixed case: conservative approach
        if len(set(m.engine for m in model_configs)) > 1:
            return 1
        
        return 4  # Default fallback
    
    def can_run_parallel(self, model_configs: List[ModelConfig]) -> bool:
        """Check if models can run in parallel without conflicts"""
        
        # Check for exclusive resource conflicts
        exclusive_resources = set()
        
        for model in model_configs:
            requirements = self.get_model_requirements(model)
            for req in requirements:
                if req.exclusive:
                    if req.resource_type in exclusive_resources:
                        return False  # Conflict detected
                    exclusive_resources.add(req.resource_type)
        
        return True
    
    def get_resource_status(self) -> Dict[str, any]:
        """Get current resource status"""
        return {
            "available_resources": {
                "gpu_memory_gb": self._resource_pool.gpu_memory_gb,
                "cpu_cores": self._resource_pool.cpu_cores,
                "system_memory_gb": self._resource_pool.system_memory_gb,
                "ollama_slots": self._resource_pool.ollama_slots,
                "network_slots": self._resource_pool.network_slots
            },
            "allocated_tasks": len(self._allocated_resources),
            "can_use_gpu": self._resource_pool.gpu_memory_gb > 0,
            "ollama_available": self._resource_pool.ollama_slots > 0
        }


class ResourceError(Exception):
    """Exception raised when resources cannot be allocated"""
    pass


# Global resource manager instance
resource_manager = ResourceManager()