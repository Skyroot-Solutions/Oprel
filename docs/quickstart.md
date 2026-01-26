# Quick Start Guide

This guide will help you get started with Oprel SDK.

## Installation

### Basic Installation

```bash
pip install oprel
```

### With Server Mode Support

```bash
pip install oprel[server]
```

This installs FastAPI and Uvicorn for server mode functionality.

### With GPU Support

```bash
# For NVIDIA GPUs
pip install oprel[cuda]

# For all features
pip install oprel[all]
```

### Verify Installation

```bash
python -c "from oprel import Model; print('Oprel installed successfully')"
```

## First Steps

### 1. Simple Generation

```python
from oprel import Model

# Create model instance
model = Model("qwencoder")

# Generate text
response = model.generate("What is Python?")
print(response)
```

Output:
```
Python is a high-level, interpreted programming language...
```

### 2. Using Ollama API

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

### 3. Interactive CLI

```bash
oprel run qwencoder
```

This starts an interactive session:
```
>>> What is Python?
Python is a high-level programming language...

>>> Explain decorators
Decorators are a way to modify functions...

>>> /exit
```

## Core Concepts

### Model Aliases

Instead of full HuggingFace paths, use simple aliases:

```python
Model("qwencoder")       # Instead of bartowski/Qwen2.5-Coder-7B-Instruct-GGUF
Model("llama3.1")        # Instead of bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
Model("gemma2")          # Instead of bartowski/gemma-2-9b-it-GGUF
```

View all aliases:
```bash
oprel list-models
```

### Server vs Direct Mode

**Server Mode (Default):**
```python
model = Model("qwencoder", use_server=True)
response = model.generate("prompt")
```
- Model persists in daemon process
- Fast subsequent requests (2-5 seconds)
- Automatic management

**Direct Mode:**
```python
model = Model("qwencoder", use_server=False)
model.load()
response = model.generate("prompt")
model.unload()
```
- Model loaded in current process
- Full control over lifecycle
- No server dependency

### Quantization

Models are available in different quantization levels:

```python
Model("qwencoder", quantization="Q4_K_M")  # 4-5GB, good quality (recommended)
Model("qwencoder", quantization="Q2_K")    # 2-3GB, lower quality
Model("qwencoder", quantization="Q8_0")    # 7-8GB, highest quality
```

Oprel auto-selects based on available memory if not specified.

## Common Use Cases

### Multi-turn Conversations

```python
from oprel import Model

model = Model("qwencoder")

# First message
response1 = model.generate(
    "My name is Alice",
    conversation_id="chat-1",
    system_prompt="You are a helpful assistant."
)

# Follow-up (context retained)
response2 = model.generate(
    "What's my name?",
    conversation_id="chat-1"
)
print(response2)  # "Your name is Alice"

# Reset conversation
response3 = model.generate(
    "Start fresh",
    conversation_id="chat-1",
    reset_conversation=True
)
```

### Streaming Responses

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

### Custom Parameters

```python
response = model.generate(
    "Write Python code to sort a list",
    max_tokens=1024,        # Maximum length
    temperature=0.3,        # Lower = more deterministic
    top_p=0.95,             # Nucleus sampling
    top_k=40                # Top-k sampling
)
```

### Code Generation

```python
from oprel import Model

model = Model("qwencoder")  # Specialized for code

response = model.generate(
    "Write a Python function to calculate fibonacci numbers"
)
print(response)
```

### Using Client Class

```python
from oprel import Client

client = Client(host='http://localhost:11434')

# Chat
response = client.chat(
    model='qwencoder',
    messages=[
        {'role': 'system', 'content': 'You are a Python expert.'},
        {'role': 'user', 'content': 'Explain list comprehensions'}
    ]
)

print(response.message.content)

# Generate
response = client.generate(
    model='qwencoder',
    prompt='Hello, world!'
)

print(response.response)

# List models
models = client.list()
for model in models.models:
    print(model.name)
```

## Configuration

### Using Config Object

```python
from oprel import Model
from oprel.core.config import Config

config = Config(
    cache_dir="/data/models",
    binary_dir="/data/binaries",
    default_max_memory_mb=8192
)

model = Model("qwencoder", config=config)
```

### Environment Variables

```bash
export OPREL_CACHE_DIR=/data/models
export OPREL_BINARY_DIR=/data/binaries
export OPREL_SERVER_PORT=11434
export OPREL_LOG_LEVEL=DEBUG
```

Then use normally:
```python
from oprel import Model
model = Model("qwencoder")  # Uses environment variables
```

### Memory Limits

```python
# Set memory limit
model = Model("qwencoder", max_memory_mb=4096)

# Or in config
config = Config(default_max_memory_mb=4096)
model = Model("qwencoder", config=config)
```

## Error Handling

### Basic Error Handling

```python
from oprel import Model
from oprel.core.exceptions import OprelError

try:
    model = Model("qwencoder")
    response = model.generate("prompt")
    print(response)
except OprelError as e:
    print(f"Error: {e}")
```

### Specific Exceptions

```python
from oprel import Model
from oprel.core.exceptions import (
    ModelNotFoundError,
    MemoryError,
    BackendError
)

try:
    model = Model("qwencoder")
    response = model.generate("prompt")
except ModelNotFoundError:
    print("Model not found. Check model name or internet connection.")
except MemoryError:
    print("Insufficient memory. Try lighter quantization (Q2_K, Q3_K_M)")
except BackendError as e:
    print(f"Backend failed: {e}")
```

### Context Manager for Cleanup

```python
from oprel import Model

with Model("qwencoder", use_server=False) as model:
    response = model.generate("prompt")
    print(response)
# Automatic cleanup
```

## CLI Usage

### Server Management

```bash
# Start server
oprel serve

# Start on custom port
oprel serve --port 8080

# List loaded models
oprel models

# Unload all models
oprel stop
```

### Generation

```bash
# Interactive mode
oprel run qwencoder

# One-shot generation
oprel run qwencoder "Explain quantum computing"

# With system prompt
oprel run qwencoder --system "You are a physicist"
```

### Interactive Commands

When in interactive mode (`oprel run qwencoder`):

- **Type your message** and press Enter to generate
- `/exit`, `/bye`, `/quit` - Exit the session
- `/reset` - Clear conversation history
- `/?` - Show help

### Model Management

```bash
# List all available aliases
oprel list-models

# Search for models
oprel search llama

# Show cache
oprel cache list

# Clear cache
oprel cache clear
```

## Best Practices

### 1. Choose the Right Model

```python
# For code generation
Model("qwencoder")      # Qwen 2.5 Coder - best for code

# For general purpose
Model("llama3.1")       # Llama 3.1 - excellent general model

# For efficiency
Model("phi3.5")         # Phi 3.5 - small but capable
```

### 2. Use Server Mode for Repeated Requests

```python
# Good: Use server mode (default)
model = Model("qwencoder")  # use_server=True by default
for i in range(10):
    response = model.generate(f"Question {i}")

# Avoid: Direct mode for repeated use
model = Model("qwencoder", use_server=False)
for i in range(10):
    model.load()  # Slow: reloads each time
    response = model.generate(f"Question {i}")
    model.unload()
```

### 3. Set Appropriate Parameters

```python
# For creative writing
response = model.generate(
    "Write a story",
    temperature=0.8,  # More creative
    max_tokens=2048   # Longer output
)

# For code generation
response = model.generate(
    "Write Python code",
    temperature=0.2,  # More deterministic
    max_tokens=1024
)

# For factual responses
response = model.generate(
    "Explain concept",
    temperature=0.3,  # More focused
    top_p=0.95
)
```

### 4. Manage Conversations

```python
# Use unique IDs for different conversations
chat_id_1 = "support-ticket-123"
chat_id_2 = "code-review-456"

response1 = model.generate("Issue with login", conversation_id=chat_id_1)
response2 = model.generate("Review this code", conversation_id=chat_id_2)

# Each conversation maintains separate context
```

### 5. Handle Errors Gracefully

```python
from oprel import Model
from oprel.core.exceptions import MemoryError

model = Model("qwencoder", max_memory_mb=4096)

try:
    response = model.generate(prompt)
except MemoryError:
    # Fallback to lighter model
    model = Model("phi3.5", quantization="Q2_K")
    response = model.generate(prompt)
```

## Performance Optimization

### GPU Acceleration

GPU is automatically detected and used:

```python
model = Model("qwencoder")  # Automatically uses GPU if available
```

Check if GPU is being used in logs:
```bash
oprel serve --log-level DEBUG
```

### Adjust Context Size

```python
from oprel.core.config import Config

# Reduce context size to save memory
config = Config(ctx_size=2048)  # Default is 4096
model = Model("qwencoder", config=config)
```

### Batch Processing

```python
from oprel import Model

model = Model("qwencoder")

prompts = [
    "Question 1",
    "Question 2",
    "Question 3"
]

responses = [model.generate(p) for p in prompts]
```

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed documentation
- Read the [Architecture](architecture.md) to understand how Oprel works
- Check [Troubleshooting](troubleshooting.md) if you encounter issues
- View example code in the `examples/` directory

## Common Questions

**Q: How do I know which model to use?**

A: For code: `qwencoder`. For general: `llama3.1`. For efficiency: `phi3.5`.

**Q: Why is the first request slow?**

A: The model needs to download (1-2 minutes) and load (60-120 seconds) on first use. Server mode caches it for fast subsequent requests (2-5 seconds).

**Q: How much RAM do I need?**

A: Minimum 4GB. Recommended 8GB+. Q4_K_M quantization typically uses 4-5GB.

**Q: Can I use custom models?**

A: Yes. Use full HuggingFace path:
```python
Model("username/model-name-GGUF", filename="model-Q4_K_M.gguf")
```

**Q: How do I update a model?**

A: Delete from cache and redownload:
```bash
oprel cache delete qwencoder
```

## Troubleshooting

If you encounter issues, see the [Troubleshooting Guide](troubleshooting.md) or:

1. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Check processes:
```bash
# Windows
Get-Process | Where-Object { $_.ProcessName -like "*llama-server*" }

# Linux/macOS
ps aux | grep llama-server
```

3. Clear cache:
```bash
oprel cache clear
```

4. Report issues: https://github.com/ragultv/oprel-SDK/issues
