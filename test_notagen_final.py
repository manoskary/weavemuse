#!/usr/bin/env python3
"""
Test script to verify NotaGen quantization fix is working
"""

import os
import sys
sys.path.append('/home/manos/codes/weavemuse')

from weavemuse.tools.notagen_tool import NotaGenTool

def test_notagen_generation():
    """Test NotaGen generation with simple prompt"""
    print("🎵 Testing NotaGen with simple prompt...")
    
    # Create NotaGen tool instance
    notagen = NotaGenTool()
    
    # Test generation
    try:
        result = notagen("Generate a simple classical piano piece")
        print(f"✅ NotaGen generation successful!")
        print(f"📝 Result preview: {str(result)[:200]}...")
        return True
    except Exception as e:
        print(f"❌ NotaGen generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing NotaGen quantization fix...")
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    success = test_notagen_generation()
    
    if success:
        print("🎉 All tests passed! NotaGen quantization is working correctly.")
    else:
        print("❌ Tests failed!")
        sys.exit(1)
