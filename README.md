# Oprel SDK

**Local LLM inference library with Ollama-compatible API**

[![PyPI version](https://badge.fury.io/py/oprel.svg)](https://pypi.org/project/oprel/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/oprel)](https://pepy.tech/project/oprel)

Oprel is a Python library for running large language models locally. It provides both a native Python API and full Ollama API compatibility, making it a drop-in replacement for Ollama-based applications.

## Installation

```bash
pip install oprel
```

For server mode with conversation management:
```bash
pip install oprel[server]
```

## Quick Start

### Ollama-Compatible API

```python
from oprel import chat, ChatResponse

response: ChatResponse = chat(
    model='qwencoder',
    messages=[
        {'role': 'user', 'content': 'What is Python?'}
    ]
)
print(response.message.content)
```

### Native Python API

```python
from oprel import Model

model = Model("qwencoder")
response = model.generate("What is Python?")
print(response)
```

### Interactive CLI

```bash
# Start interactive session
oprel run qwencoder

# One-shot generation
oprel run qwencoder "Explain quantum computing"
```

## Features

- **Ollama API Compatibility**: Drop-in replacement for Ollama Python client
- **Native Python API**: Direct model integration without server overhead
- **Server Mode**: Persistent model caching for fast subsequent requests
- **Conversation Memory**: Built-in multi-turn conversation support
- **Model Aliases**: 50+ predefined aliases for popular models
- **Interactive CLI**: Chat interface similar to Ollama
- **Automatic Downloads**: Models downloaded from HuggingFace on first use
- **Hardware Detection**: Auto-selects optimal quantization and GPU layers
- **Memory Management**: Prevents system freezes with configurable limits
- **Process Management**: Automatic recovery from backend crashes

## API Reference

### Ollama-Compatible API

Oprel provides full compatibility with the Ollama Python API:

```python
from oprel import chat, generate, list, Client

# Module-level functions
response = chat(model='qwencoder', messages=[...])
response = generate(model='qwencoder', prompt='Hello')
models = list()

# Client class
client = Client(host='http://localhost:11434')
response = client.chat(model='qwencoder', messages=[...])
```

See [OLLAMA_API.md](OLLAMA_API.md) for complete API documentation.

### Native Model API

```python
from oprel import Model

# Server mode (default) - fast subsequent loads
model = Model("qwencoder", use_server=True)
response = model.generate("prompt")

# Direct mode - no server required
model = Model("qwencoder", use_server=False)
model.load()
response = model.generate("prompt")
model.unload()
```

## Comparison with Ollama

| Feature | Ollama | Oprel |
|---------|--------|-------|
| Installation | Separate daemon | pip install |
| API Compatibility | Python client | Full compatibility |
| Direct Python API | No | Yes |
| Server Mode | Required | Optional |
| Model Aliases | Yes | Yes (50+) |
| Conversation Memory | Yes | Yes |
| Memory Protection | Basic | Configurable limits |
| Crash Recovery | Manual | Automatic |
| Hidden Processes | Yes | Yes |

## Usage Examples

### Ollama API Examples

#### Chat Completion
```python
from oprel import chat

response = chat(
    model='qwencoder',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain Python decorators.'}
    ]
)
print(response.message.content)
```

#### Streaming Chat
```python
from oprel import chat

stream = chat(
    model='qwencoder',
    messages=[{'role': 'user', 'content': 'Write a story'}],
    stream=True
)

for chunk in stream:
    print(chunk.message.content, end='', flush=True)
```

#### Using Client Class
```python
from oprel import Client

client = Client(host='http://localhost:11434')

response = client.chat(
    model='qwencoder',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### Native API Examples

#### Basic Generation
```python
from oprel import Model

model = Model("qwencoder")
response = model.generate("Explain quantum computing")
print(response)
```

#### Conversation Memory
```python
from oprel import Model

model = Model("qwencoder")

response1 = model.generate(
    "My name is Alice",
    conversation_id="chat-1"
)
Model Aliases

Oprel provides 50+ predefined aliases for popular models:

```python
Model("llama3")          # Meta-Llama-3-8B-Instruct
Model("llama3.1")        # Meta-Llama-3.1-8B-Instruct
Model("qwencoder")       # Qwen2.5-Coder-7B-Instruct
Model("gemma2")          # gemma-2-9b-it
Model("mistral")         # Mistral-7B-Instruct-v0.3
Model("phi3.5")          # Phi-3.5-mini-instruct
Model("deepseek-coder")  # DeepSeek-Coder-V2-Instruct
```

View all aliases:
```bash
oprel list-models
```

## Configuration

### Model Configuration

```python
from oprel import Model, Config

config = Config(
    cache_dir="/path/to/cache",
    binary_dir="/path/to/binaries",
    default_max_memory_mb=8192,
    ctx_size=4096,
    batch_size=512
)

model = Model(
    "qwencoder",
    quantization="Q4_K_M",      # Quantization level
    max_memory_mb=4096,          # Memory limit
    backend="llama.cpp",         # Backend engine
    config=config,               # Custom configuration
    use_server=True              # Enable server mode
)
```

### Quantization Levels

- `Q2_K` - Smallest, lowest quality (2-3GB)
- `Q3_K_M` - Small, medium quality (3-4GB)
- `Q4_K_M` - Balanced (4-5GB, recommended)
- `Q5_K_M` - Large, high quality (5-6GB)
- `Q6_K` - Very large, very high quality (6-7GB)
- `Q8_0` - Largest, highest quality (7-8GB)

Oprel auto-selects quantization based on available memory.
Built-in multi-turn conversation support:
Command Line Interface

### Server Management
```bash
oprel serve                  # Start server on port 11434
oprel serve --port 8080      # Custom port
oprel models                 # List loaded models in server
oprel stop                   # Unload all models
```

### Interactive Mode
```bash
oprel run qwencoder          # Start interactive session
oprel run qwencoder "prompt" # One-shot generation
```

Interactive commands:
- `/exit`, `/bye`, `/quit` - Exit session
- `/reset` - Clear conversation history
- `/?` - Show help

### Chat Mode
```Advanced Features

### Memory Protection

```python
from oprel import Model
from oprel.core.exceptions import MemoryError

model = Model("qwencoder", max_memory_mb=4096)

try:
    model.generate("prompt")
except MemoryError as e:
    print(f"Memory limit exceeded: {e}")
```

### Streaming

```python
from oprel import generate

stream = generate(
    model='qwencoder',
    prompt='Write a story',
    stream=True
)

for chunk in stream:
    print(chunk.response, end='', flush=True)
```

### Context Manager

```python
from oprel import Model

with Model("qwencoder") as model:
    response = model.generate("Hello")
```

### Conversation Management

```python
from oprel import Model

model = Model("qwencoder")

# Start conversation with system prompt
response1 = model.generate(
    "What's 2+2?",
    conversation_id="math-session",
    system_prompt="You are a math tutor."
)

# Continue conversation
response2 = model.generate(
    "And 10+10?",
    conversation_id="math-session"
)

# Reset conversation
response3 = model.generate(
    "New topic",
    conversation_id="math-session",
    reset_conversation=True
)
```Supported Models

Oprel works with any GGUF model from HuggingFace. Recommended models:

| Model | Alias | Parameters | Use Case |
|-------|-------|------------|----------|
| Llama 3.1 | `llama3.1` | 8B | General purpose |
| Qwen 2.5 Coder | `qwencoder` | 7B | Code generation |
| Gemma 2 | `gemma2` | 9B | General purpose |
| Mistral | `mistral` | 7B | General purpose |
| Phi 3.5 | `phi3.5` | 3.8B | Efficient inference |
| DeepSeek Coder | `deepseek-coder` | 16B | Code & reasoning |

## System Requirements

- Python 3.9 or higher
- Operating System: Windows, macOS, Linux
- Memory: 4GB minimum, 8GB+ recommended
- GPU: Optional (CUDA/Metal supported)

## Dependencies

### Core Dependencies
- huggingface-hub >= 0.20.0
- psutil >= 5.9.0
- requests >= 2.31.0
- pydantic >= 2.0.0
- rich >= 13.0.0

### Optional Dependencies

```bash
pip install oprel[server]  # Server mode (FastAPI, Uvicorn)
pip install oprel[cuda]    # NVIDIA GPU support
pip install oprel[all]     # All optional dependencies

```Documentation

- [Ollama API Reference](OLLAMA_API.md) - Complete Ollama-compatible API documentation
- [Interactive Mode Guide](INTERACTIVE_MODE.md) - CLI interactive mode usage
- [API Documentation](docs/api_reference.md) - Native Python API reference
- [Architecture Overview](docs/architecture.md) - System architecture and design
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Error Handling

```python
from oprel import Model
from oprel.core.exceptions import (
    OprelError,
    ModelNotFoundError,
    MemoryError,
    BackendError
)

try:
    model = Model("qwencoder")
    response = model.generate("prompt")
except ModelNotFoundError:
    print("Model not found on HuggingFace")
except MemoryError:
    print("Insufficient memory for model")
except BackendError as e:
    print(f"Backend error: {e}")
except OprelError as e:
    print(f"General error: {e}")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/test_client_api.py -v

# Run with coverage
pytest tests/ --cov=oprel --cov-report=html
```

## Contributing

Contributions are welcome. Please submit pull requests to the GitHub repository.

## License

MIT License - see LICENSE file for details.

## Links

- PyPI: https://pypi.org/project/oprel/
- GitHub: https://github.com/ragultv/oprel-SDK
- Issues: https://github.com/ragultv/oprel-SDK/issues

Server mode keeps models cached in memory, just like Ollama!

## 🗂️ Supported Models

Works with any **GGUF** model from HuggingFace:

| Family | Recommended Alias | Use Case |
|--------|-------------------|----------|
| **Llama 3.1** | `llama3.1` | General purpose |
| **Qwen 2.5 Coder** | `qwencoder` | Best for coding |
| **Gemma 2** | `gemma2` | Fast, efficient |
| **Mistral** | `mistral` | Great all-rounder |
| **Phi 3.5** | `phi3.5` | Small but powerful |
| **DeepSeek** | `deepseek-coder` | Strong reasoning |

## 🛠️ Requirements

- **Python**: 3.9+
- **OS**: macOS, Linux, Windows
- **RAM**: 4GB minimum (8GB+ recommended)
- **GPU**: Optional (CUDA/Metal auto-detected)

## 📦 Optional Dependencies

```bash
pip install oprel[server]  # FastAPI + Uvicorn for server mode
pip install oprel[cuda]    # NVIDIA GPU support
pip install oprel[all]     # Everything
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🔗 Links

- **PyPI**: [pypi.org/project/oprel](https://pypi.org/project/oprel/)
- **GitHub**: [github.com/ragultv/oprel-SDK](https://github.com/ragultv/oprel-SDK)
- **Issues**: [github.com/ragultv/oprel-SDK/issues](https://github.com/ragultv/oprel-SDK/issues)

---

**Keywords**: llm, local-llm, ollama-alternative, llama3, qwen, gemma, mistral, gguf, llama.cpp, python-llm, local-ai, offline-ai, conversational-ai, text-generation, model-server, ai-runtime

**Made with ❤️ for developers who want local AI without the hassle**