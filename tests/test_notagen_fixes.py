#!/usr/bin/env python3
"""
Test script for NotaGen tensor fixes and quantization improvements.
"""

import os
import sys
import torch
import traceback
import logging

# Add the weavemuse directory to the path
sys.path.insert(0, '/home/manos/codes/weavemuse')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_manager():
    """Test the new model manager functionality."""
    print("=" * 60)
    print("🧪 Testing NotaGen Model Manager")
    print("=" * 60)
    
    try:
        from weavemuse.models.notagen.model_manager import NotaGenModelManager, get_optimal_notagen_model
        
        # Test system info
        manager = NotaGenModelManager()
        system_info = manager.get_system_info()
        print(f"System Info: {system_info}")
        
        # Test model selection
        best_model = manager.select_best_model()
        print(f"Selected model: {best_model}")
        
        # Test model config
        config = manager.get_model_config(best_model)
        print(f"Model config: {config}")
        
        print("✅ Model Manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_tensor_fixes():
    """Test the tensor type fixes in the PatchLevelDecoder."""
    print("\n" + "=" * 60)
    print("🧪 Testing Tensor Type Fixes")
    print("=" * 60)
    
    try:
        from weavemuse.models.notagen.utils import PatchLevelDecoder
        from transformers import GPT2Config
        
        # Create a test decoder
        config = GPT2Config(
            vocab_size=512,
            n_positions=1024,
            n_embd=1280,
            n_layer=20,
            n_head=16,
            n_inner=5120,
            activation_function="gelu_new",
        )
        
        decoder = PatchLevelDecoder(config)
        decoder.eval()
        
        # Test with different tensor types that might come from quantization
        batch_size = 2
        patch_size = 16  # Use the correct PATCH_SIZE from config
        
        print("Testing with LongTensor (should work)...")
        patches_long = torch.randint(0, 128, (batch_size, patch_size), dtype=torch.long)
        
        with torch.no_grad():
            result = decoder(patches_long)
            print(f"✅ LongTensor test passed! Output type: {type(result)}")
        
        print("Testing with FloatTensor (should be converted)...")
        patches_float = torch.rand(batch_size, patch_size) * 128
        patches_float = patches_float.to(torch.float32)
        
        with torch.no_grad():
            result = decoder(patches_float)
            print(f"✅ FloatTensor test passed! Output type: {type(result)}")
        
        print("Testing with values outside vocab range (should be clamped)...")
        patches_invalid = torch.randint(-10, 600, (1, patch_size), dtype=torch.long)  # Use correct dimensions
        
        with torch.no_grad():
            result = decoder(patches_invalid)
            print(f"✅ Invalid range test passed! Output shape: {result.shape}")
        
        print("✅ All tensor type tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tensor fix test failed: {e}")
        traceback.print_exc()
        return False

def test_quantization_logic():
    """Test the quantization decision logic."""
    print("\n" + "=" * 60)
    print("🧪 Testing Quantization Logic")
    print("=" * 60)
    
    try:
        from weavemuse.models.notagen.inference import should_use_quantization, get_system_memory_gb, get_gpu_memory_gb
        
        print("System Memory:", get_system_memory_gb(), "GB")
        
        if torch.cuda.is_available():
            print("GPU Memory:", get_gpu_memory_gb(), "GB")
        else:
            print("CUDA not available")
        
        should_quantize = should_use_quantization()
        print(f"Quantization recommended: {should_quantize}")
        
        print("✅ Quantization logic test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quantization logic test failed: {e}")
        traceback.print_exc()
        return False

def test_simple_generation():
    """Test simple music generation to ensure everything works end-to-end."""
    print("\n" + "=" * 60)
    print("🧪 Testing Simple Music Generation")
    print("=" * 60)
    
    try:
        # Use the NotaGen tool directly instead of low-level functions
        from weavemuse.tools.notagen_tool import NotaGenTool
        
        print("Creating NotaGen tool instance...")
        notagen = NotaGenTool()
        
        # Test a simple generation with proper parameters
        period = "Classical"
        composer = "Mozart"
        instrumentation = "Piano"
        print(f"Generating music: {period} - {composer} - {instrumentation}")
        
        # Try generation with the correct parameters
        result = notagen(period, composer, instrumentation)
        
        # Check if result contains expected content
        if result and isinstance(result, dict) and 'abc' in result:
            abc_content = result['abc']
            if abc_content and len(str(abc_content)) > 100:
                print(f"✅ Generation successful!")
                print(f"Result length: {len(str(abc_content))} characters")
                print(f"First 100 chars: {str(abc_content)[:100]}...")
                return True
        elif result and len(str(result)) > 100:
            print(f"✅ Generation successful!")
            print(f"Result length: {len(str(result))} characters")
            print(f"First 100 chars: {str(result)[:100]}...")
            return True
        else:
            print(f"❌ Generation test failed: Result format unexpected")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
            return False
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting NotaGen Tests")
    print("Using Python:", sys.version)
    print("Using PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    tests = [
        ("Model Manager", test_model_manager),
        ("Tensor Fixes", test_tensor_fixes),
        ("Quantization Logic", test_quantization_logic),
        ("Simple Generation", test_simple_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! NotaGen is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
