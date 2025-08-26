#!/usr/bin/env python3
"""
Simple test to check abstract method implementation
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_abstract_methods():
    """Test what abstract methods are missing."""
    print("🧪 Testing Abstract Methods")
    
    try:
        from weavemuse.tools.chat_musician_tool import ChatMusicianTool
        from weavemuse.tools.base_tools import ManagedTransformersTool
        
        print("✅ Imports successful")
        
        # Check what abstract methods are defined in the base class
        print(f"📍 Base class: {ManagedTransformersTool}")
        print(f"📍 Abstract methods in base: {getattr(ManagedTransformersTool, '__abstractmethods__', 'None')}")
        
        # Check what methods are implemented in ChatMusicianTool
        print(f"📍 ChatMusicianTool methods:")
        for name in dir(ChatMusicianTool):
            if not name.startswith('__'):
                print(f"  - {name}")
        
        # Try to see if methods exist
        has_load_model = hasattr(ChatMusicianTool, '_load_model')
        has_call_model = hasattr(ChatMusicianTool, '_call_model')
        
        print(f"📍 Has _load_model: {has_load_model}")
        print(f"📍 Has _call_model: {has_call_model}")
        
        if has_load_model:
            load_method = getattr(ChatMusicianTool, '_load_model')
            print(f"📍 _load_model: {load_method}")
            
        if has_call_model:
            call_method = getattr(ChatMusicianTool, '_call_model')
            print(f"📍 _call_model: {call_method}")
        
        # Try to create instance
        print("📍 Attempting instantiation...")
        tool = ChatMusicianTool()
        print("✅ Successfully created instance!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_abstract_methods()
