#!/usr/bin/env python3
"""
Test script for ChatMusicianTool import and basic functionality.
"""

import sys
import os
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_chat_musician_import():
    """Test that ChatMusicianTool can be imported."""
    print("=" * 60)
    print("🧪 Testing ChatMusicianTool Import")
    print("=" * 60)
    
    try:
        print("📍 Attempting to import ChatMusicianTool...")
        from weavemuse.tools.chat_musician_tool import ChatMusicianTool
        print("✅ Successfully imported ChatMusicianTool")
        
        # Check class attributes
        print(f"✅ Class name: {ChatMusicianTool.__name__}")
        print(f"✅ Module: {ChatMusicianTool.__module__}")
        print(f"✅ Tool name: {ChatMusicianTool.name}")
        print(f"✅ Tool description: {ChatMusicianTool.description}")
        print(f"✅ Inputs: {ChatMusicianTool.inputs}")
        print(f"✅ Output type: {ChatMusicianTool.output_type}")
        print(f"✅ Base classes: {[cls.__name__ for cls in ChatMusicianTool.__bases__]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import ChatMusicianTool: {e}")
        traceback.print_exc()
        return False

def test_chat_musician_instantiation():
    """Test that ChatMusicianTool can be instantiated."""
    print("\n" + "=" * 60)
    print("🧪 Testing ChatMusicianTool Instantiation")
    print("=" * 60)
    
    try:
        from weavemuse.tools.chat_musician_tool import ChatMusicianTool
        print("📍 Attempting to create ChatMusicianTool instance...")
        
        tool = ChatMusicianTool()
        print("✅ Successfully created ChatMusicianTool instance")
        print(f"✅ Tool type: {type(tool)}")
        print(f"✅ Tool name: {tool.name}")
        print(f"✅ Is loaded: {getattr(tool, 'is_loaded', 'Unknown')}")
        
        return tool
        
    except Exception as e:
        print(f"❌ Failed to create ChatMusicianTool instance: {e}")
        traceback.print_exc()
        return None

def test_chat_musician_forward():
    """Test the forward method of ChatMusicianTool."""
    print("\n" + "=" * 60)
    print("🧪 Testing ChatMusicianTool Forward Method")
    print("=" * 60)
    
    try:
        from weavemuse.tools.chat_musician_tool import ChatMusicianTool
        
        tool = ChatMusicianTool()
        print("📍 Testing forward method...")
        
        # Test with basic query
        result = tool.forward(query="What is a major scale?")
        print(f"✅ Forward method result: {result}")
        
        # Test with parameters
        result2 = tool.forward(
            query="Compose a simple melody", 
            max_tokens="256", 
            temperature="0.8"
        )
        print(f"✅ Forward method with params: {result2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test forward method: {e}")
        traceback.print_exc()
        return False

def test_module_contents():
    """Test what's actually in the chat_musician_tool module."""
    print("\n" + "=" * 60)
    print("🧪 Testing Module Contents")
    print("=" * 60)
    
    try:
        import weavemuse.tools.chat_musician_tool as module
        
        print("📍 Module attributes:")
        for name in dir(module):
            if not name.startswith('_'):
                value = getattr(module, name)
                print(f"  {name}: {type(value)} = {value}")
        
        print("\n📍 Module globals with 'Tool' in name:")
        for name, value in module.__dict__.items():
            if not name.startswith('_') and 'Tool' in str(value):
                print(f"  {name}: {value}")
                
        return True
        
    except Exception as e:
        print(f"❌ Failed to inspect module: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting ChatMusicianTool Tests")
    print("Environment:", os.environ.get('CONDA_DEFAULT_ENV', 'Unknown'))
    print("Python path:", sys.path[0])
    print()
    
    # Test 1: Module contents
    module_ok = test_module_contents()
    
    # Test 2: Import
    import_ok = test_chat_musician_import()
    
    if import_ok:
        # Test 3: Instantiation
        tool = test_chat_musician_instantiation()
        
        if tool:
            # Test 4: Forward method
            forward_ok = test_chat_musician_forward()
        else:
            forward_ok = False
    else:
        forward_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    print("=" * 60)
    print(f"Module inspection: {'✅ PASS' if module_ok else '❌ FAIL'}")
    print(f"Import test: {'✅ PASS' if import_ok else '❌ FAIL'}")
    print(f"Instantiation test: {'✅ PASS' if tool else '❌ FAIL'}")
    print(f"Forward method test: {'✅ PASS' if forward_ok else '❌ FAIL'}")
    
    overall_success = import_ok and tool and forward_ok
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
