#!/usr/bin/env python3
"""
WeaveMuse CLI - A comprehensive music agent framework
"""

import argparse
import sys
from typing import Optional

def run_interface() -> None:
    """Launch the Gradio interface."""
    import sys
    import os
    
    try:
        print("Initializing WeaveMuse...")
        print("This may take a few minutes on first run as models are loaded...")
        
        # Simply run the main app module which contains the complete setup
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        app_path = os.path.join(project_root, "app.py")
        
        if os.path.exists(app_path):
            # Execute the app.py file
            exec(open(app_path).read())
        else:
            # Fallback: try to import and run a simpler version
            from weavemuse.agents.music_agent import MusicAgent
            agent = MusicAgent()
            agent.launch_gradio()
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching interface: {e}")
        sys.exit(1)

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WeaveMuse - A comprehensive music agent framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--interface", 
        action="store_true",
        help="Launch the Gradio web interface"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="WeaveMuse 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Default behavior: launch interface
    if len(sys.argv) == 1 or args.interface:
        print("Launching WeaveMuse interface...")
        run_interface()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()