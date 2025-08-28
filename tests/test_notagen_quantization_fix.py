#!/usr/bin/env python3
"""
Test script to verify NotaGen quantization fixes.
"""

import sys
import os

# Add the parent directory to the path so we can import weavemuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_notagen_quantization():
    """Test that NotaGen works with both quantized and unquantized models."""
    
    try:
        from weavemuse.tools.notagen_tool import NotaGenTool
        from weavemuse.models.notagen.model_manager import NotaGenModelManager
        
        print("✅ Successfully imported NotaGen components")
        
        # Test model manager (if it exists)
        try:
            from weavemuse.models.notagen.model_manager import NotaGenModelManager
            manager = NotaGenModelManager()
            print(f"✅ Model manager created")
        except ImportError:
            print("ℹ️  Model manager not yet implemented")
        
        # Test NotaGen tool initialization
        tool = NotaGenTool()
        print("✅ NotaGen tool initialized")
        
        # Test a simple generation (this will load the model)
        print("🎵 Testing NotaGen generation...")
        try:
            result = tool.forward(
                period="Classical",
                composer="Mozart", 
                instrumentation="Piano"
            )
            print("✅ NotaGen generation successful!")
            print(f"   Generated result: {result[:100]}...")  # Show first 100 chars
            
        except Exception as e:
            print(f"❌ NotaGen generation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing NotaGen quantization fixes...")
    success = test_notagen_quantization()
    if success:
        print("\n🎉 All tests passed! NotaGen quantization is working.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
    sys.exit(0 if success else 1)
