# Oprel SDK Documentation

Welcome to the Oprel SDK documentation. Oprel is a Python library for running large language models locally with full Ollama API compatibility.

## Overview

Oprel provides a simple, efficient way to run LLMs locally without the complexity of managing separate daemon processes. It supports both a native Python API and complete Ollama compatibility.

## Key Features

- **Ollama API Compatibility**: Drop-in replacement for Ollama Python client
- **Native Python API**: Direct model integration without server overhead  
- **Server Mode**: Persistent model caching for fast subsequent requests
- **50+ Model Aliases**: Simple names for popular models
- **Interactive CLI**: Chat interface with conversation memory
- **Automatic Downloads**: Models fetched from HuggingFace on demand
- **Hardware Detection**: Auto-optimization for your system
- **Memory Protection**: Configurable limits to prevent system freezes
- **Crash Recovery**: Automatic backend restart on failures

## Quick Start

### Installation

```bash
pip install oprel
pip install oprel[server]  # With server mode support
```

### Basic Usage

```python
from oprel import Model

model = Model("qwencoder")
response = model.generate("What is Python?")
print(response)
```

### Ollama API

```python
from oprel import chat

response = chat(
    model='qwencoder',
    messages=[
        {'role': 'user', 'content': 'What is Python?'}
    ]
)
print(response.message.content)
```

### CLI

```bash
# Interactive mode
oprel run qwencoder

# One-shot generation
oprel run qwencoder "Explain quantum computing"
```

## Documentation Sections

### [API Reference](api_reference.md)
Complete API documentation covering:
- Model class and methods
- Ollama-compatible API functions
- Client classes
- Response models
- Configuration options
- Exception handling

### [Architecture](architecture.md)
System architecture and design:
- Component overview
- Data flow diagrams
- Operating modes
- Backend management
- Testing strategy
- Extension points

### [Quick Start Guide](quickstart.md)
Step-by-step tutorials:
- Installation instructions
- First model example
- Conversation management
- Streaming responses
- Error handling
- Best practices

### [Troubleshooting](troubleshooting.md)
Common issues and solutions:
- Installation problems
- Model loading errors
- Performance issues
- Backend crashes
- Memory errors
- Platform-specific issues

## API Overview

### Model API

```python
from oprel import Model

# Initialize model
model = Model(
    "qwencoder",
    quantization="Q4_K_M",
    max_memory_mb=4096,
    use_server=True
)

# Generate text
response = model.generate(
    "Explain Python decorators",
    max_tokens=1024,
    temperature=0.7
)

# Conversation
response1 = model.generate(
    "My name is Alice",
    conversation_id="chat-1"
)
response2 = model.generate(
    "What's my name?",
    conversation_id="chat-1"
)
```

### Ollama API

```python
from oprel import Client

client = Client()

# Chat completion
response = client.chat(
    model='qwencoder',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello!'}
    ]
)

# Text generation
response = client.generate(
    model='qwencoder',
    prompt='Write a haiku'
)

# List models
models = client.list()

# Model info
info = client.show('qwencoder')
```

## System Requirements

- Python 3.9 or higher
- Operating System: Windows, macOS, Linux
- RAM: 4GB minimum, 8GB+ recommended
- GPU: Optional (CUDA/Metal supported)
- Disk Space: 4-8GB per model

## Supported Models

Oprel works with any GGUF model from HuggingFace. Pre-configured aliases include:

| Model | Alias | Size | Use Case |
|-------|-------|------|----------|
| Llama 3.1 | `llama3.1` | 8B | General purpose |
| Qwen 2.5 Coder | `qwencoder` | 7B | Code generation |
| Gemma 2 | `gemma2` | 9B | General purpose |
| Mistral | `mistral` | 7B | General purpose |
| Phi 3.5 | `phi3.5` | 3.8B | Efficient inference |
| DeepSeek Coder | `deepseek-coder` | 16B | Code & reasoning |

View all 50+ aliases:
```bash
oprel list-models
```

## Operating Modes

### Server Mode (Default)

Models persist in a daemon process for fast subsequent requests.

```python
model = Model("qwencoder", use_server=True)
response = model.generate("prompt")
```

**Advantages:**
- Fast subsequent requests (2-5 seconds vs 60-120 seconds)
- Shared model across multiple clients
- Automatic crash recovery
- Conversation state management

**Performance:**
- First load: 60-120 seconds
- Subsequent requests: 2-5 seconds

### Direct Mode

Models load directly in the current process.

```python
model = Model("qwencoder", use_server=False)
model.load()
response = model.generate("prompt")
model.unload()
```

**Advantages:**
- No server dependency
- Full lifecycle control
- Suitable for embedded use
- Simpler debugging

**Performance:**
- Each load: 60-120 seconds

## Configuration

### Environment Variables

```bash
OPREL_CACHE_DIR=/path/to/cache     # Model cache directory
OPREL_BINARY_DIR=/path/to/binaries # Binary files directory
OPREL_SERVER_PORT=11434            # Server port
OPREL_LOG_LEVEL=INFO               # Logging level
```

### Configuration File

```python
from oprel.core.config import Config

config = Config(
    cache_dir="/data/models",
    binary_dir="/data/binaries",
    default_max_memory_mb=8192,
    ctx_size=4096,
    batch_size=512,
    n_threads=8,
    n_gpu_layers=-1
)

model = Model("qwencoder", config=config)
```

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
    print("Model not found")
except MemoryError:
    print("Insufficient memory")
except BackendError as e:
    print(f"Backend error: {e}")
except OprelError as e:
    print(f"General error: {e}")
```

## CLI Reference

### Server Management
```bash
oprel serve                     # Start server
oprel serve --port 8080         # Custom port
oprel models                    # List loaded models
oprel stop                      # Unload all models
```

### Generation
```bash
oprel run qwencoder             # Interactive mode
oprel run qwencoder "prompt"    # One-shot generation
```

Interactive commands:
- `/exit`, `/bye`, `/quit` - Exit
- `/reset` - Clear conversation
- `/?` - Help

### Model Management
```bash
oprel list-models               # All aliases
oprel search llama              # Search models
oprel cache list                # Cached models
oprel cache clear               # Clear cache
```

## Examples

### Multi-turn Conversation

```python
from oprel import Model

model = Model("qwencoder")

# Start conversation
response1 = model.generate(
    "I'm learning Python. What should I start with?",
    conversation_id="learning",
    system_prompt="You are a helpful programming tutor."
)

# Continue conversation
response2 = model.generate(
    "Can you explain variables?",
    conversation_id="learning"
)

# Reset and start fresh
response3 = model.generate(
    "New topic: web development",
    conversation_id="learning",
    reset_conversation=True
)
```

### Streaming

```python
from oprel import generate

stream = generate(
    model='qwencoder',
    prompt='Write a story about a robot',
    stream=True
)

for chunk in stream:
    print(chunk.response, end='', flush=True)
```

### Custom Generation Parameters

```python
response = model.generate(
    "Write code to sort a list",
    max_tokens=2048,
    temperature=0.3,  # More deterministic
    top_p=0.95,
    top_k=40
)
```

## Performance Tips

1. **Use Server Mode**: Enable persistent caching for repeated requests
2. **Choose Right Quantization**: Balance quality and memory (Q4_K_M recommended)
3. **Enable GPU**: Automatic if available, significantly faster
4. **Adjust Context Size**: Reduce if running out of memory
5. **Optimize Threads**: Match to your CPU core count

## Getting Help

- Report issues: https://github.com/ragultv/oprel-SDK/issues
- API Reference: [api_reference.md](api_reference.md)
- Troubleshooting: [troubleshooting.md](troubleshooting.md)

## License

MIT License - see LICENSE file for details.

## Next Steps

- Read the [Quick Start Guide](quickstart.md) for detailed tutorials
- Explore the [API Reference](api_reference.md) for complete documentation
- Review the [Architecture](architecture.md) to understand internals
- Check [Troubleshooting](troubleshooting.md) if you encounter issues
