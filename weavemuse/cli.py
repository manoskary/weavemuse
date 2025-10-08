#!/usr/bin/env python3
"""
WeaveMuse CLI - A comprehensive music agent framework
"""

import argparse
import sys
import os
from typing import Optional

def run_interface() -> None:
    """Launch the Gradio interface."""
    try:
        print("Initializing WeaveMuse...")
        print("This may take a few seconds on first run as dependencies are loaded...")
        
        from weavemuse.interfaces.gui import WeaveMuseGUI 
        demo = WeaveMuseGUI().launch_interface(share=False)   
        return demo
            
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching interface: {e}")
        sys.exit(1)

def run_gui_interface() -> None:
    """Launch the Gradio interface."""
    run_interface()

def run_terminal_interface() -> None:
    """Launch the terminal interface."""
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from weavemuse.interfaces.terminal_interface import WeaveMuseTerminal
        
        print("⚡ Starting WeaveMuse Terminal Interface...")
        print("Fast startup with on-demand loading...")
        
        # Use fast startup
        terminal = WeaveMuseTerminal()
        terminal.run()
        
    except ImportError as e:
        print(f"Error importing terminal interface: {e}")
        print("Please ensure all dependencies are properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching terminal interface: {e}")
        sys.exit(1)

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WeaveMuse - A comprehensive music agent framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add version argument at top level
    parser.add_argument(
        "--version", 
        action="version", 
        version="WeaveMuse 0.1.0"
    )
    
    # Create subparsers for different interface modes
    subparsers = parser.add_subparsers(
        dest='command',
        help='Interface mode to launch',
        metavar='{gui,terminal}'
    )
    
    # GUI interface subcommand
    gui_parser = subparsers.add_parser(
        'gui',
        help='Launch the Gradio web interface (default)'
    )
    
    # Terminal interface subcommand
    terminal_parser = subparsers.add_parser(
        'terminal',
        help='Launch the terminal-based interface'
    )
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.command == 'gui':
        print("Launching WeaveMuse GUI interface...")
        run_gui_interface()
    elif args.command == 'terminal':
        print("Launching WeaveMuse Terminal interface...")
        run_terminal_interface()
    else:
        # Default behavior: launch GUI interface (backward compatibility)
        if len(sys.argv) == 1:
            print("Launching WeaveMuse GUI interface (default)...")
            print("Use 'weavemuse gui' or 'weavemuse terminal' to be explicit.")
            run_gui_interface()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
