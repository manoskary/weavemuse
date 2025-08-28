#!/usr/bin/env python3
"""
Simple test to verify NotaGen is generating music properly
"""

import os
import sys
sys.path.append('/home/manos/codes/weavemuse')

def test_notagen_music_generation():
    """Test that NotaGen is actually generating music."""
    print("🎵 Testing NotaGen Music Generation")
    print("=" * 60)
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    try:
        print("📦 Importing NotaGen tool...")
        from weavemuse.tools.notagen_tool import NotaGenTool
        
        print("🔧 Creating NotaGen instance...")
        notagen = NotaGenTool()
        print(f"✅ Tool created with model: {notagen.model_id}")
        
        print("\n🎼 Testing music generation...")
        print("   Prompt: Classical - Bach - Piano")
        
        # Generate music with a simple prompt
        result = notagen("Classical", "Bach", "Piano")
        
        # Check the result
        if result:
            print("✅ Generation successful!")
            
            # Check if it's a dictionary with expected keys
            if isinstance(result, dict):
                print(f"📊 Result type: Dictionary with keys: {list(result.keys())}")
                
                # Check ABC content
                if 'abc' in result:
                    abc_content = result['abc']
                    print(f"📝 ABC content length: {len(abc_content)} characters")
                    
                    # Verify it looks like ABC notation
                    if 'X:' in abc_content and 'T:' in abc_content:
                        print("✅ ABC notation format detected")
                        
                        # Show a preview
                        lines = abc_content.split('\n')[:10]  # First 10 lines
                        print("\n📄 ABC Preview:")
                        print("-" * 40)
                        for line in lines:
                            print(line)
                        print("-" * 40)
                        
                        # Check for actual music content (notes)
                        music_lines = [line for line in abc_content.split('\n') 
                                     if any(note in line for note in 'ABCDEFGabcdefg')]
                        
                        if music_lines:
                            print(f"🎶 Found {len(music_lines)} lines with musical notes")
                            print("✅ Music generation appears successful!")
                            return True
                        else:
                            print("⚠️  No musical notes found in output")
                            return False
                    else:
                        print("❌ Output doesn't look like ABC notation")
                        return False
                else:
                    print("❌ No 'abc' key in result")
                    return False
            else:
                print(f"📊 Result type: {type(result)}")
                print(f"📝 Content: {str(result)[:200]}...")
                
                # Check if it's a string with ABC content
                if isinstance(result, str) and 'X:' in result:
                    print("✅ String result appears to be ABC notation")
                    return True
                else:
                    print("⚠️  Result format unexpected but not empty")
                    return False
        else:
            print("❌ Generation returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 NotaGen Music Generation Test")
    print("Testing that the quantized model actually produces music")
    print("=" * 60)
    
    success = test_notagen_music_generation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: NotaGen is generating music correctly!")
        print("✅ The quantized model is working as expected")
    else:
        print("❌ FAILURE: NotaGen is not generating music properly")
        print("⚠️  There may be an issue with the model configuration")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
