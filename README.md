# 🚗 Drive Like A Human

> Rethinking Autonomous Driving with Large Language Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An intelligent autonomous driving agent that leverages Large Language Models (LLMs) to make human-like driving decisions in highway scenarios. The system uses tool-calling capabilities to reason about traffic situations, check safety constraints, and execute actions in the Highway-Env simulation environment.

## ✨ Features

- 🤖 **Multi-LLM Support**: Compatible with Groq, OpenAI, and Ollama (local)
- 🛠️ **Tool-Based Reasoning**: Uses specialized tools for lane checking, conflict detection, and safety validation
- 🎯 **ReAct Pattern**: Implements reasoning and acting framework for decision-making
- 📊 **Decision Logging**: SQLite database for tracking all decisions and reasoning
- 🎥 **Video Recording**: Automatic recording of driving episodes
- 🔧 **Configurable**: Easy configuration via YAML file

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [LLM Providers](#llm-providers)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Or with pip
pip install uv
```

### Setup Project

```bash
# Clone the repository
git clone https://github.com/yourusername/DriveLikeAHuman.git
cd DriveLikeAHuman

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Configure Your LLM Provider

Copy the example config and edit it:

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your preferred LLM settings
```

**For Groq (Recommended for beginners):**
```yaml
LLM_PROVIDER: 'groq'
GROQ_KEY: 'your-groq-api-key'  # Get from https://console.groq.com
GROQ_MODEL: 'llama3-70b-8192'
```

**For Ollama (Local & Free):**
```bash
# Install and setup Ollama
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

```yaml
LLM_PROVIDER: 'ollama'
OLLAMA_BASE_URL: 'http://localhost:11434/v1'
OLLAMA_MODEL: 'llama3.1:8b'
```

**For OpenAI:**
```yaml
LLM_PROVIDER: 'openai'
OPENAI_KEY: 'sk-your-openai-api-key'
OPENAI_MODEL: 'gpt-4o'
```

### 2. Run the Simulation

```bash
python HELLM.py
```

The agent will:
1. Initialize the highway environment with 15 vehicles
2. Use LLM to analyze traffic situations
3. Make decisions using safety tools
4. Record video to `results-video/`
5. Log decisions to `results-db/highwayv0.db`

### 3. View Results

- **Videos**: Check `results-video/` for MP4 recordings
- **Decisions**: Query the SQLite database in `results-db/`
- **Console**: Watch real-time decision-making in the terminal

## ⚙️ Configuration

### config.yaml Structure

```yaml
# Provider Selection: 'groq', 'openai', or 'ollama'
LLM_PROVIDER: 'groq'

# Groq Configuration
GROQ_KEY: 'your-api-key'
GROQ_MODEL: 'llama3-70b-8192'  # or 'mixtral-8x7b-32768', 'llama3-8b-8192'

# OpenAI Configuration  
OPENAI_KEY: 'sk-...'
OPENAI_MODEL: 'gpt-4o'  # or 'gpt-4-turbo', 'gpt-3.5-turbo'

# Ollama Configuration (Local)
OLLAMA_BASE_URL: 'http://localhost:11434/v1'
OLLAMA_MODEL: 'llama3.1:8b'  # or 'llama3.1:70b', 'mistral:7b'
```

### Environment Configuration

You can modify the environment settings in `HELLM.py`:

```python
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of vehicles to track
        "see_behind": True,     # Can see vehicles behind
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),  # Speed range
    },
    "duration": 40,           # Episode duration in seconds
    "vehicles_density": 2,    # Traffic density
}
```

## 🎮 Usage

### Basic Usage

```bash
# Run with default config
python HELLM.py
```

### Advanced Usage

```python
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from scenario.scenario import Scenario

# Initialize scenario
sce = Scenario(vehicleCount=15, database="results-db/custom.db")

# Create driver agent
driver = DriverAgent(
    client=client,
    toolModels=tools,
    sce=sce,
    model_name="llama3-70b-8192",
    verbose=True
)

# Run decision loop
driver.agentRun(previous_decision)
thoughts = driver.exportThoughts()

# Parse output
parser = OutputParser(sce=sce, client=client)
action = parser.agentRun(thoughts)
```

### Scenario Replay

```python
from scenario.scenarioReplay import ScenarioReplay

# Replay a recorded scenario
replay = ScenarioReplay(database="results-db/highwayv0.db")
replay.show_frame(frame_number=10)
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Highway Environment                     │
│                    (Gymnasium + highway-env)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ Observations (vehicle positions, speeds)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Scenario Manager                         │
│              (Tracks vehicles, lanes, state)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ Structured scenario data
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Driver Agent                            │
│                  (LLM with Tool Calling)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Tools:                                               │   │
│  │ • Get Available Actions                              │   │
│  │ • Get Available Lanes                                │   │
│  │ • Get Lane Involved Cars                             │   │
│  │ • Check Change Lane Conflict                         │   │
│  │ • Check Acceleration Safety                          │   │
│  │ • Check Keep Speed Safety                            │   │
│  │ • Check Deceleration Safety                          │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ Thoughts & reasoning
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Parser                            │
│              (Extracts structured action)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ Action (lane change, accelerate, etc.)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Action Execution                          │
│            (Apply to environment, record video)              │
└─────────────────────────────────────────────────────────────┘
```

### Decision-Making Flow

1. **Observation**: Environment provides vehicle states
2. **Scenario Update**: Parse and structure observations
3. **Agent Reasoning**: 
   - LLM analyzes current situation
   - Calls tools to check safety constraints
   - Reasons about best action
4. **Output Parsing**: Extract structured action from LLM response
5. **Execution**: Apply action to environment
6. **Recording**: Log decision and update video

### Available Actions

| Action ID | Action Name | Description |
|-----------|-------------|-------------|
| 0 | `LANE_LEFT` | Change to left lane |
| 1 | `IDLE` | Keep current speed and lane |
| 2 | `LANE_RIGHT` | Change to right lane |
| 3 | `FASTER` | Accelerate |
| 4 | `SLOWER` | Decelerate |

### Safety Tools

The agent uses specialized tools for safety validation:

- **Get_Available_Actions**: Lists legal actions in current situation
- **Get_Available_Lanes**: Returns lanes accessible to a vehicle
- **Get_Lane_Involved_Car**: Finds vehicles in a specific lane
- **Is_Change_Lane_Conflict_With_Car**: Checks lane change safety
- **Is_Acceleration_Conflict_With_Car**: Validates acceleration safety
- **Is_Keep_Speed_Conflict_With_Car**: Checks if maintaining speed is safe
- **Is_Deceleration_Safe**: Validates deceleration won't cause collision

## 🔌 LLM Providers

### Comparison

| Provider | Speed | Cost | Privacy | Best For |
|----------|-------|------|---------|----------|
| **Groq** | ⚡⚡⚡ Very Fast | 💰 Low | ☁️ Cloud | Development & Testing |
| **OpenAI** | ⚡⚡ Fast | 💰💰 Medium | ☁️ Cloud | Production & Research |
| **Ollama** | ⚡ Varies | 🆓 Free | 🔒 Local | Privacy & Offline Use |

### Groq Setup

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Update config.yaml:
   ```yaml
   LLM_PROVIDER: 'groq'
   GROQ_KEY: 'gsk_...'
   GROQ_MODEL: 'llama3-70b-8192'
   ```

**Recommended Models:**
- `llama3-70b-8192` - Best balance (recommended)
- `mixtral-8x7b-32768` - Good for complex reasoning
- `llama3-8b-8192` - Faster, lighter

### OpenAI Setup

1. Get API key from [platform.openai.com](https://platform.openai.com)
2. Update config.yaml:
   ```yaml
   LLM_PROVIDER: 'openai'
   OPENAI_KEY: 'sk-...'
   OPENAI_MODEL: 'gpt-4o'
   ```

**Recommended Models:**
- `gpt-4o` - Best quality
- `gpt-4-turbo` - Fast and capable
- `gpt-3.5-turbo` - Economical option

### Ollama Setup (Local)

1. **Install Ollama:**
   ```bash
   brew install ollama
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   ollama pull llama3.1:8b
   ```

4. **Update config.yaml:**
   ```yaml
   LLM_PROVIDER: 'ollama'
   OLLAMA_MODEL: 'llama3.1:8b'
   ```

**Recommended Models:**
- `llama3.1:8b` - Best balance (8GB RAM)
- `llama3.1:70b` - Better reasoning (40GB RAM)
- `mistral:7b` - Fast alternative
- `qwen2.5:14b` - Good reasoning


## 📁 Project Structure

```
DriveLikeAHuman/
├── HELLM.py                    # Main simulation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── config.yaml.example        # Example configuration
│
├── LLMDriver/                 # Core agent logic
│   ├── __init__.py
│   ├── driverAgent.py         # Main driving agent (ReAct loop)
│   ├── outputAgent.py         # Action parser
│   ├── customTools.py         # Safety checking tools
│   ├── agent_propmts.py       # System prompts
│   └── callbackHandler.py     # Logging utilities
│
├── scenario/                  # Environment management
│   ├── __init__.py
│   ├── scenario.py            # Scenario state tracking
│   ├── scenarioReplay.py      # Replay recorded scenarios
│   └── baseClass.py           # Base classes
│
├── results-video/             # Recorded episode videos
│   └── highwayv0-episode-*.mp4
│
├── results-db/                # Decision logs (SQLite)
│   └── highwayv0.db
│
└── assets/                    # Documentation images
    └── *.png
```

## 🔍 Key Components

### DriverAgent (`LLMDriver/driverAgent.py`)

Implements the ReAct (Reasoning + Acting) pattern:
- Receives scenario observations
- Uses LLM to reason about situation
- Calls safety tools to validate decisions
- Outputs final decision with explanation

### OutputParser (`LLMDriver/outputAgent.py`)

Extracts structured actions from LLM responses:
- Parses natural language output
- Converts to action ID and name
- Provides safety fallback (keep_speed) on errors

### Scenario (`scenario/scenario.py`)

Manages simulation state:
- Tracks all vehicles (position, speed, lane)
- Maintains lane information
- Exports structured JSON for LLM
- Logs to SQLite database

### Custom Tools (`LLMDriver/customTools.py`)

Specialized functions for safety validation:
- Lane checking and availability
- Conflict detection with other vehicles
- Time-headway calculations
- Safety validation for all actions

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/DriveLikeAHuman.git
cd DriveLikeAHuman

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# Install development dependencies (optional)
uv pip install black ruff pytest
```

### Running Tests

```bash
# Run the simulation
python HELLM.py

# Check for syntax errors
python -m py_compile LLMDriver/*.py scenario/*.py

# Format code (if black installed)
black LLMDriver/ scenario/
```

### Adding a New Tool

1. Create tool class in `LLMDriver/customTools.py`:
```python
class MyNewTool:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
    
    @prompts(
        name='My_Tool_Name',
        description="Tool description for LLM"
    )
    def inference(self, input: str) -> str:
        # Tool logic here
        return "Tool output"
```

2. Register in `HELLM.py`:
```python
toolModels = [
    # ... existing tools ...
    MyNewTool(sce),
]
```

### Database Schema

The SQLite database stores decisions with this schema:

```sql
CREATE TABLE decisionINFO (
    frame INTEGER PRIMARY KEY,
    scenario TEXT,           -- JSON of scenario state
    thinkAndThoughts TEXT,   -- Reasoning process
    finalAnswer TEXT,        -- Final decision
    outputParser TEXT        -- Parsed action (base64)
);
```

## 📊 Example Output

```
[bold green]Starting Simulation with Model: llama3-70b-8192[/bold green]
Decision at frame 0 is running ...
[green]Driver agent is running...[/green]

Action: Get_Available_Actions
Action Input: {'query': 'ego'}
[blue]Observation: You can ONLY use one of the following actions:
IDLE--remain in the current lane with current speed; 
FASTER--accelerate the vehicle; 
SLOWER--decelerate the vehicle;[/blue]

Action: Get_Lane_Involved_Car
Action Input: {'query': 'lane_1'}
[blue]Observation: vehicle_3 is driving at 28.5m/s on lane_1, 
and it's driving in front of ego car for 45.2 meters.[/blue]

Final Answer: 
    "decision":{"IDLE"},
    "explanations":{"Current lane is safe, vehicle ahead is far enough"}

[cyan]Parser finished.[/cyan]
Frame 0: Action 1 | keep_speed
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv) - Gymnasium environment for autonomous driving
- [Groq](https://groq.com) - Fast LLM inference
- [Ollama](https://ollama.ai) - Local LLM runtime
- [OpenAI](https://openai.com) - GPT models and API

## 📚 Additional Resources

- **Example Config**: [config.yaml.example](config.yaml.example)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

Made with ❤️ for autonomous driving research

