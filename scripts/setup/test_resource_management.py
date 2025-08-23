#!/usr/bin/env python3
"""
Resource Management Test

Test script to verify resource-aware scheduling works correctly.
Creates scenarios with different engine combinations to test conflict detection.
"""

from core.resource_manager import resource_manager
from core.model_registry import model_registry


def test_resource_detection():
    """Test system resource detection"""
    print("SYSTEM RESOURCE DETECTION")
    print("=" * 40)
    
    status = resource_manager.get_resource_status()
    
    print(f"GPU Memory: {status['available_resources']['gpu_memory_gb']:.1f} GB")
    print(f"CPU Cores: {status['available_resources']['cpu_cores']}")
    print(f"System Memory: {status['available_resources']['system_memory_gb']:.1f} GB")
    print(f"Ollama Available: {status['ollama_available']}")
    print(f"GPU Available: {status['can_use_gpu']}")
    print()


def test_model_requirements():
    """Test model resource requirement calculation"""
    print("MODEL RESOURCE REQUIREMENTS")
    print("=" * 40)
    
    test_models = [
        "qwen2.5-vl:7b",           # Ollama
        "qwen2.5-vl-7b-hf",        # HuggingFace 
        "gpt-4o",                  # OpenAI
    ]
    
    for model_id in test_models:
        try:
            model_config = model_registry.get_model(model_id)
            requirements = resource_manager.get_model_requirements(model_config)
            
            print(f"{model_id} ({model_config.engine.value}):")
            for req in requirements:
                exclusive_str = " (EXCLUSIVE)" if req.exclusive else ""
                print(f"  {req.resource_type.value}: {req.amount}{exclusive_str}")
            print()
        except ValueError as e:
            print(f"{model_id}: NOT FOUND")
            print()


def test_parallelism_analysis():
    """Test parallel execution analysis"""
    print("PARALLELISM ANALYSIS")
    print("=" * 40)
    
    test_scenarios = [
        # Scenario 1: Single Ollama model
        (["qwen2.5-vl:7b"], "Single Ollama model"),
        
        # Scenario 2: Single HuggingFace model  
        (["qwen2.5-vl-7b-hf"], "Single HuggingFace model"),
        
        # Scenario 3: Single OpenAI model
        (["gpt-4o"], "Single OpenAI model"),
        
        # Scenario 4: Multiple OpenAI models
        (["gpt-4o", "gpt-4-vision"], "Multiple OpenAI models"),
        
        # Scenario 5: Ollama + HuggingFace conflict
        (["qwen2.5-vl:7b", "qwen2.5-vl-7b-hf"], "Ollama + HuggingFace (conflict)"),
        
        # Scenario 6: Ollama + OpenAI
        (["qwen2.5-vl:7b", "gpt-4o"], "Ollama + OpenAI"),
        
        # Scenario 7: HuggingFace + OpenAI  
        (["qwen2.5-vl-7b-hf", "gpt-4o"], "HuggingFace + OpenAI"),
    ]
    
    for model_ids, description in test_scenarios:
        print(f"Scenario: {description}")
        
        try:
            model_configs = [model_registry.get_model(mid) for mid in model_ids]
            
            can_parallel = resource_manager.can_run_parallel(model_configs)
            optimal_workers = resource_manager.get_optimal_parallelism(model_configs)
            
            print(f"  Models: {', '.join(model_ids)}")
            print(f"  Can run in parallel: {can_parallel}")
            print(f"  Optimal workers: {optimal_workers}")
            
            if not can_parallel:
                print(f"  → Will use sequential execution")
            elif optimal_workers == 1:
                print(f"  → Will use single worker despite being parallel-safe")
            else:
                print(f"  → Will use {optimal_workers} parallel workers")
            
        except ValueError as e:
            print(f"  Error: {e}")
        
        print()


def test_resource_allocation():
    """Test resource allocation context manager"""
    print("RESOURCE ALLOCATION TEST")
    print("=" * 40)
    
    # Test Ollama model allocation
    try:
        model_config = model_registry.get_model("qwen2.5-vl:7b")
        print(f"Testing allocation for {model_config.id}...")
        
        with resource_manager.acquire_resources(model_config, "test_task_1"):
            print("  Resources acquired successfully")
            status = resource_manager.get_resource_status()
            print(f"  Active tasks: {status['allocated_tasks']}")
            
            # Try to acquire another conflicting resource
            try:
                with resource_manager.acquire_resources(model_config, "test_task_2"):
                    print("  ERROR: Should not be able to acquire conflicting resource!")
            except Exception as e:
                print(f"  Correctly blocked conflicting allocation: {type(e).__name__}")
        
        print("  Resources released successfully")
        
    except ValueError as e:
        print(f"  Model not found: {e}")
    except Exception as e:
        print(f"  Allocation test failed: {e}")
    
    print()


def main():
    """Run all resource management tests"""
    print("RESOURCE MANAGEMENT TESTING")
    print("=" * 50)
    print()
    
    test_resource_detection()
    test_model_requirements() 
    test_parallelism_analysis()
    test_resource_allocation()
    
    print("Resource management testing completed!")


if __name__ == "__main__":
    main()