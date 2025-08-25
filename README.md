# WeaveMuse

A comprehensive music agent framework built on smolagents, integrating state-of-the-art music AI models for understanding, generation, and interaction.

## Features

### 🎼 Music Understanding
- **ChatMusician Integration**: Natural language music analysis and understanding
- **Music Theory Analysis**: Automatic analysis of musical structures, harmony, and form
- **Audio Understanding**: Content analysis of audio files

### 🎵 Music Generation
- **Symbolic Music Generation**: ABC notation generation using NotaGen
- **Audio Generation**: High-quality audio synthesis using Stable Audio Open
- **Conditional Generation**: Generate music based on text prompts, styles, and constraints

### 🤖 Agent Framework
- **smolagents Integration**: Intelligent agent system that decides which tools to use
- **Multi-modal Support**: Text, audio, and symbolic music inputs
- **Tool Orchestration**: Seamless integration between different music AI models

### 🎨 Visualization & Interface
- **Gradio Web Interface**: User-friendly web interface for interaction
- **Verovio Integration**: Beautiful sheet music rendering
- **Audio Playback**: Integrated audio player for generated content

### 🐳 Deployment
- **Docker Support**: Easy deployment with Docker containers
- **Cloud Ready**: Scalable deployment options
- **API Endpoints**: RESTful API for programmatic access

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the interface at http://localhost:7860
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/manoskary/music-agent.git
cd music-agent

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application
music-agent serve
```

### Quick Demo

After installation, you can run our demonstration scripts to see the AI capabilities:

```bash
# Run the success demo to verify everything is working
python demos/demo_success.py

# Run the complete AI demo to see all features
python demos/demo_ai_complete.py

# See all available demos
ls demos/
```

## Usage

### Web Interface

1. Open your browser to `http://localhost:7860`
2. Upload audio files or enter text prompts
3. Select the type of music task you want to perform
4. Let the agent decide which tools to use and generate results

### Python API

```python
from music_agent import MusicAgent

# Initialize the agent
agent = MusicAgent()

# Generate music from text
result = agent.run("Create a peaceful piano piece in C major")

# Analyze uploaded audio
analysis = agent.run("Analyze the harmony and structure of this piece", audio_file="song.wav")

# Generate ABC notation
abc_notation = agent.run("Convert this melody to ABC notation", audio_file="melody.wav")
```

### Command Line

```bash
# Generate music from text
music-agent generate --text "Create a jazz composition for piano and saxophone"

# Analyze audio file
music-agent analyze --audio "path/to/song.wav"

# Convert between formats
music-agent convert --input "song.abc" --output "song.wav"
```

## Architecture

The Music Agent Framework consists of several key components:

### Core Agent
- **MusicAgent**: Main agent class built on smolagents
- **Tool Router**: Intelligent routing between different music tools
- **Context Manager**: Maintains conversation and task context

### Music Tools
- **ChatMusicianTool**: Music understanding and analysis
- **NotaGenTool**: Symbolic music generation in ABC notation
- **StableAudioTool**: High-quality audio generation
- **AudioAnalysisTool**: Audio content understanding
- **VerovioTool**: Sheet music visualization

### Interfaces
- **Gradio UI**: Web-based user interface
- **FastAPI Server**: RESTful API endpoints
- **CLI**: Command-line interface

## Configuration

Create a `.env` file in the project root:

```env
# Model configurations
CHATMUSICIAN_MODEL_ID=m-a-p/ChatMusician
NOTAGEN_MODEL_PATH=./models/notagen
STABLE_AUDIO_MODEL_ID=stabilityai/stable-audio-open-1.0

# Hugging Face Hub
HF_TOKEN=your_huggingface_token

# Server configuration
HOST=0.0.0.0
PORT=7860
DEBUG=false

# GPU configuration
DEVICE=cuda
TORCH_DTYPE=float16
```

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
music-agent/
├── src/music_agent/           # Main package
│   ├── agents/               # Agent implementations
│   ├── tools/                # Music tool implementations
│   ├── interfaces/           # UI and API interfaces
│   ├── models/               # Model wrappers and utilities
│   └── utils/                # Utility functions
├── tests/                    # Test suite
├── demos/                    # Demo scripts and examples
├── dev-tools/                # Development and testing tools
├── scripts/                  # Utility and setup scripts
├── docker/                   # Docker configuration
├── examples/                 # Usage examples
├── docs/                     # Documentation
└── models/                   # Downloaded model files
```

## Models and Tools

### Integrated Models

1. **ChatMusician**: Music understanding and analysis
   - Model: `m-a-p/ChatMusician`
   - Capabilities: Music theory, harmony analysis, composition guidance

2. **NotaGen**: Symbolic music generation
   - Model: Custom implementation with pre-trained weights
   - Output: ABC notation format

3. **Stable Audio Open**: Audio generation
   - Model: `stabilityai/stable-audio-open-1.0`
   - Output: High-quality 44.1kHz stereo audio

4. **Audio Analysis**: Content understanding
   - Multiple models for different analysis tasks
   - Capabilities: Genre classification, mood detection, structure analysis

### Tool Capabilities

- **Music Composition**: Generate complete musical pieces
- **Harmony Analysis**: Analyze chord progressions and harmonic structure
- **Style Transfer**: Convert music between different styles
- **Audio Synthesis**: Convert symbolic music to audio
- **Format Conversion**: Between ABC, MIDI, MusicXML, and audio formats
- **Music Education**: Explain music theory concepts and analysis

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [smolagents](https://github.com/huggingface/smolagents) - Agent framework
- [ChatMusician](https://github.com/hf-lin/ChatMusician) - Music understanding
- [NotaGen](https://github.com/ElectricAlexis/NotaGen) - Symbolic music generation
- [Stable Audio](https://github.com/Stability-AI/stable-audio-tools) - Audio generation
- [Verovio](https://www.verovio.org/) - Music notation rendering

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{music_agent_framework,
  title={Music Agent Framework: Comprehensive Music AI with smolagents},
  author={Music Agent Team},
  year={2025},
  url={https://github.com/music-agent/music-agent}
}
```
