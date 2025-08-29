"""
WeaveMuse Terminal Interface
A terminal-based interface for the WeaveMuse music agent framework
"""

import os
import sys
import traceback
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Only import essential smolagents components
from smolagents import CodeAgent, DuckDuckGoSearchTool
from smolagents.tools import Tool
from smolagents import InferenceClientModel


class WeaveMuseTerminal:
    """Terminal-based interface for WeaveMuse."""
    
    def __init__(self):
        """Initialize the terminal interface with lazy loading."""
        self.conversation_history = []
        self.uploaded_files = []
        self.agent = None
        self.tools: List[Tool] = []
        self.vram_manager = None
        self.tools_initialized = False
        self.model_choice = None
        
        print("⚡ WeaveMuse Terminal Interface - Fast Start Mode")
    
    def ask_model_choice(self) -> str:
        """Ask user to choose between local or HuggingFace model."""
        print("\n🤖 Choose your AI model:")
        print("1. Local Model (similar to GUI app - requires more resources)")
        print("2. HuggingFace InferenceClient (cloud-based - faster startup)")
        print("3. Minimal Mode (search-only, no AI model)")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                if choice in ['1', '2', '3']:
                    return choice
                else:
                    print("❌ Please enter 1, 2, or 3")
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def _setup_basic_tools(self):
        """Initialize basic tools only."""
        if not hasattr(self, '_basic_tools_loaded'):
            try:
                from weavemuse.tools.memory_manager import VRAMManager
                self.vram_manager = VRAMManager()
                self._basic_tools_loaded = True
            except Exception as e:
                print(f"⚠️  VRAM manager not available: {e}")
        
        # Always start with search tool
        self.tools: List[Tool] = [DuckDuckGoSearchTool()]
        self.available_tools = {
            'search': 'DuckDuckGoSearchTool',
            'notagen': 'NotaGenTool', 
            'chat_musician': 'ChatMusicianTool',
            'audio_flamingo': 'AudioFlamingoTool'
        }
    
    def _load_specific_tool(self, tool_name: str):
        """Load a specific tool only when needed."""
        print(f"🔄 Loading {tool_name} tool...")
        
        try:
            if tool_name == 'notagen':
                from weavemuse.tools.notagen_tool import NotaGenTool
                tool = NotaGenTool()
                self.tools.append(tool)
                print("✅ NotaGen tool loaded")
                return tool
            elif tool_name == 'chat_musician':
                from weavemuse.tools.chat_musician_tool import ChatMusicianTool
                tool = ChatMusicianTool()
                self.tools.append(tool)
                print("✅ ChatMusician tool loaded")
                return tool
            elif tool_name == 'audio_flamingo':
                from weavemuse.tools.audio_flamingo_tool import AudioFlamingoTool
                tool = AudioFlamingoTool()
                self.tools.append(tool)
                print("✅ AudioFlamingo tool loaded")
                return tool
        except Exception as e:
            print(f"❌ Error loading {tool_name}: {e}")
            return None
    
    def _analyze_query_needs(self, query: str) -> list:
        """Analyze what tools might be needed for this query."""
        query_lower = query.lower()
        needed_tools = []
        
        # Music generation keywords
        if any(word in query_lower for word in ['generate', 'compose', 'create', 'melody', 'song', 'music', 'abc', 'notation', 'midi']):
            needed_tools.append('notagen')
        
        # Music theory/analysis keywords  
        if any(word in query_lower for word in ['analyze', 'theory', 'chord', 'scale', 'harmony', 'explain']):
            needed_tools.append('chat_musician')
        
        # Audio processing keywords
        if any(word in query_lower for word in ['audio', 'sound', 'wav', 'mp3', 'process', 'convert']):
            needed_tools.append('audio_flamingo')
        
        # Search keywords
        if any(word in query_lower for word in ['search', 'find', 'look up', 'what is', 'who is']):
            # Search tool is always available, no need to load
            pass
        
        return needed_tools
    
    def _setup_agent(self):
        """Initialize the agent based on user choice."""
        if not self.model_choice:
            self.model_choice = self.ask_model_choice()
        
        print("🤖 Setting up AI agent...")
        
        try:
            if self.model_choice == '1':
                # Local model similar to GUI app
                print("🔄 Loading local model (similar to GUI app)...")
                try:
                    # Try to use a local model directly
                    print("⚠️  Local model mode not fully implemented yet.")
                    print("🔄 Falling back to HuggingFace model...")
                    self.model_choice = '2'
                except Exception as e:
                    print(f"❌ Error with local model: {e}")
                    print("🔄 Falling back to HuggingFace model...")
                    self.model_choice = '2'
            
            if self.model_choice == '2':
                # HuggingFace InferenceClient
                print("🔄 Setting up HuggingFace InferenceClient...")
                model = InferenceClientModel(
                    "Qwen/Qwen2.5-Coder-32B-Instruct",
                    max_tokens=1000
                )
                
                self.agent = CodeAgent(
                    tools=self.tools,
                    model=model,
                    stream_outputs=True
                )
                print("✅ HuggingFace agent initialized!")
            
            elif self.model_choice == '3':
                # Minimal mode - no AI model
                print("🔧 Minimal mode - search tools only")
                self.agent = None
                print("✅ Minimal mode ready!")
                
        except Exception as e:
            print(f"❌ Error initializing agent: {e}")
            print("🔄 Falling back to minimal mode...")
            self.agent = None
            traceback.print_exc()
    
    def print_banner(self):
        """Print the welcome banner."""
        print("\n" + "="*60)
        print("🎵 WeaveMuse Terminal Interface 🎵")
        print("="*60)
        print("A comprehensive music agent framework")
        
        if self.tools_initialized:
            print(f"Available tools: {len(self.tools)} loaded")
            for i, tool in enumerate(self.tools, 1):
                print(f"  {i}. {tool.__class__.__name__}")
        else:
            print("Tools: Will load when first needed")
        
        if self.model_choice:
            model_names = {'1': 'Local Model', '2': 'HuggingFace', '3': 'Minimal Mode'}
            print(f"AI Model: {model_names.get(self.model_choice, 'Unknown')}")
        
        print("\nCommands:")
        print("  /help     - Show help message")
        print("  /quit     - Exit interface")
        print("  /clear    - Clear conversation")
        print("  /history  - Show history")
        print("  /files    - Show uploaded files")
        print("  /tools    - Load/reload tools")
        print("  /model    - Change AI model")
        print("="*60)
    
    def print_help(self):
        """Print help information."""
        print("\n📖 WeaveMuse Terminal Help")
        print("-" * 30)
        print("Available commands:")
        print("  /help     - Show this help message")
        print("  /quit     - Exit the interface")
        print("  /clear    - Clear conversation history")
        print("  /history  - Show conversation history")
        print("  /files    - Show uploaded files")
        print("  /tools    - Load/reload music tools")
        print("  /model    - Change AI model")
        print("\nMusic Generation:")
        print("  - Ask to generate music in ABC notation")
        print("  - Request music analysis or explanation")
        print("  - Upload and analyze audio files")
        print("  - Convert between different music formats")
        print("\nExample queries:")
        print("  'Generate a happy melody in C major'")
        print("  'Convert this ABC notation to MIDI'")
        print("  'Analyze the uploaded audio file'")
        print("  'Create a blues progression'")
    
    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📜 No conversation history yet.")
            return
        
        print("\n📜 Conversation History:")
        print("-" * 40)
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"{i}. [{entry['type']}] {entry['content'][:100]}...")
    
    def show_files(self):
        """Show uploaded files."""
        if not self.uploaded_files:
            print("📁 No files uploaded yet.")
            return
        
        print("\n📁 Uploaded Files:")
        print("-" * 30)
        for i, file_info in enumerate(self.uploaded_files, 1):
            print(f"{i}. {file_info['name']} ({file_info['size']} bytes)")
    
    def handle_file_input(self, user_input: str) -> str:
        """Handle file upload from user input."""
        words = user_input.split()
        file_paths = []
        
        for word in words:
            if os.path.exists(word):
                file_paths.append(word)
        
        if file_paths:
            print(f"🔍 Found {len(file_paths)} file(s) in your message:")
            for file_path in file_paths:
                try:
                    file_size = os.path.getsize(file_path)
                    file_info = {
                        'name': os.path.basename(file_path),
                        'path': file_path,
                        'size': file_size
                    }
                    self.uploaded_files.append(file_info)
                    print(f"  ✅ {file_info['name']} ({file_size} bytes)")
                except Exception as e:
                    print(f"  ❌ Error reading {file_path}: {e}")
        
        return user_input
    
    def process_message(self, user_input: str) -> str:
        """Process user message and get agent response."""
        # Analyze what tools might be needed
        needed_tools = self._analyze_query_needs(user_input)
        
        # Load only the tools that are actually needed
        if needed_tools:
            print(f"🎯 Query needs: {', '.join(needed_tools)} tools")
            for tool_name in needed_tools:
                # Check if tool is already loaded
                tool_names = [tool.__class__.__name__ for tool in self.tools]
                if tool_name == 'notagen' and 'NotaGenTool' not in tool_names:
                    self._load_specific_tool('notagen')
                elif tool_name == 'chat_musician' and 'ChatMusicianTool' not in tool_names:
                    self._load_specific_tool('chat_musician')
                elif tool_name == 'audio_flamingo' and 'AudioFlamingoTool' not in tool_names:
                    self._load_specific_tool('audio_flamingo')
        
        # Ensure basic setup
        if not hasattr(self, '_basic_tools_loaded'):
            self._setup_basic_tools()
        
        if not self.agent and self.model_choice != '3':
            self._setup_agent()
        
        if not self.agent:
            if self.model_choice == '3':
                return "🔧 Minimal mode active. Try asking something that can be answered with search, or use /model to enable AI features."
            else:
                return "❌ Agent not available. Try /model to reconfigure."
        
        try:
            # Add user message to history
            self.conversation_history.append({
                'type': 'user',
                'content': user_input
            })
            
            # Get agent response
            print("🤔 Thinking...")
            response = self.agent.run(user_input)
            
            # Add agent response to history
            self.conversation_history.append({
                'type': 'agent',
                'content': str(response)
            })
            
            return str(response)
            
        except Exception as e:
            error_msg = f"❌ Error processing message: {e}"
            print(f"Error details: {traceback.format_exc()}")
            return error_msg
    
    def reload_tools(self):
        """Reload tools."""
        print("🔄 Reloading tools...")
        self.tools = [DuckDuckGoSearchTool()]  # Reset to basic tools
        self._basic_tools_loaded = False
        self._setup_basic_tools()
    
    def change_model(self):
        """Change AI model."""
        print("🔄 Changing AI model...")
        self.agent = None
        self.model_choice = None
        self._setup_agent()
    
    def run(self):
        """Main terminal interface loop."""
        try:
            self.print_banner()
            
            while True:
                try:
                    # Get user input
                    user_input = input("\n🎵 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        command = user_input[1:].lower()
                        
                        if command in ['quit', 'exit', 'q']:
                            print("👋 Goodbye!")
                            break
                        elif command == 'help':
                            self.print_help()
                        elif command == 'clear':
                            self.conversation_history.clear()
                            self.uploaded_files.clear()
                            print("🧹 Conversation and files cleared!")
                        elif command == 'history':
                            self.show_history()
                        elif command == 'files':
                            self.show_files()
                        elif command == 'tools':
                            self.reload_tools()
                        elif command == 'model':
                            self.change_model()
                        else:
                            print(f"❓ Unknown command: {command}")
                            print("Type /help for available commands.")
                        continue
                    
                    # Handle file input
                    user_input = self.handle_file_input(user_input)
                    
                    # Process message with agent
                    response = self.process_message(user_input)
                    print(f"\n🤖 Agent: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Unexpected error: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"❌ Fatal error in terminal interface: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            if self.vram_manager and hasattr(self.vram_manager, 'cleanup'):
                try:
                    self.vram_manager.cleanup()
                except:
                    pass
            print("🧹 Cleanup completed.")


if __name__ == "__main__":
    terminal = WeaveMuseTerminal()
    terminal.run()
