#!/usr/bin/env python3
"""
Quick test for the quantized NotaGen model
"""

import sys
import os
sys.path.append('/home/manos/codes/weavemuse')

def test_quantized_notagen():
    """Test the quantized NotaGen model."""
    print("🧪 Testing Quantized NotaGen Model")
    print("=" * 50)
    
    try:
        # Import the inference module (which will load the quantized model)
        from weavemuse.models.notagen import inference
        print("✅ Successfully imported inference module")
        
        # Check if quantization is enabled
        from weavemuse.models.notagen.config import USE_QUANTIZATION
        print(f"📊 Quantization enabled: {USE_QUANTIZATION}")
        
        # Test basic model functionality
        test_prompt = """X:1
T:Test Melody
C:Test Composer
K:C major
M:4/4
L:1/4"""
        
        print("🎵 Testing with prompt:")
        print(test_prompt)
        print("\n🔄 Generating...")
        
        # Test generation (short sequence)
        result = inference.inference(
            prompt=test_prompt,
            max_length=100,  # Short generation for testing
            verbose=False
        )
        
        if result:
            print("✅ Generation successful!")
            print("📄 Result (first 200 chars):")
            print(result[:200] + "..." if len(result) > 200 else result)
            return True
        else:
            print("❌ Generation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quantized_notagen()
    if success:
        print("\n🎉 Quantized model test passed!")
    else:
        print("\n💥 Quantized model test failed!")
        sys.exit(1)
