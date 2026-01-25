# Oprel SDK

**Run LLMs locally with one line of Python** - The SQLite of AI

[![PyPI version](https://badge.fury.io/py/oprel.svg)](https://pypi.org/project/oprel/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/oprel)](https://pepy.tech/project/oprel)

> **Ollama alternative** that's a Python library, not a daemon. Server mode with persistent model caching, conversation memory, and 50+ model aliases.

```python
from oprel import Model

# Uses server mode by default - 2 second responses after first load!
model = Model("llama3")  # or "qwencoder", "mistral", "gemma2", etc.
print(model.generate("What is Python?"))
```

## 🔥 What's New in v0.3.0

- **🚀 Server Mode**: Persistent model caching like Ollama (2 min → 2 sec)
- **💬 Conversation Memory**: Multi-turn chat with context retention
- **🏷️ 50+ Model Aliases**: Use `llama3`, `qwencoder`, `gemma2` instead of full paths
- **📡 Full CLI**: `oprel serve`, `oprel chat`, `oprel run`, `oprel models`

## 🎯 Why Oprel vs Ollama?

| Feature | Ollama | Oprel |
|---------|--------|-------|
| **Installation** | Separate daemon required | `pip install oprel` |
| **Usage** | HTTP API to background service | Python library + optional server |
| **Desktop Apps** | Users must install Ollama | Just a pip dependency |
| **Memory Protection** | ❌ Can freeze your PC | ✅ Graceful OOM handling |
| **Model Aliases** | `ollama run llama3` | `Model("llama3")` same! |
| **Conversation Memory** | ✅ | ✅ Built-in |
| **Server Mode** | Always required | Optional (default on) |
| **Direct Mode** | ❌ | ✅ No server needed |

## 🚀 Quick Start

### Installation

```bash
pip install oprel

# With server mode dependencies (recommended)
pip install oprel[server]
```

### 1. Quick Generation (Server Mode - Default)

```python
from oprel import Model

model = Model("llama3")  # Auto-starts server if needed
response = model.generate("Explain quantum computing in 3 sentences")
print(response)
```

### 2. Interactive Chat with Memory

```python
from oprel import Model

model = Model("qwencoder")

# Conversation automatically tracked
response1 = model.generate("My name is Alice", conversation_id="chat-1")
response2 = model.generate("What's my name?", conversation_id="chat-1")
# Response: "Your name is Alice!" ✅ Context retained!
```

### 3. CLI Usage (Like Ollama)

```bash
# Start the server
oprel serve

# Run a quick prompt
oprel run llama3 "Write a haiku about Python"

# Interactive chat
oprel chat qwencoder --system "You are a senior Python developer"

# List available models
oprel list-models

# Search models
oprel search llama
```

### 4. Direct Mode (No Server)

```python
from oprel import Model

# Bypass server, load directly in this process
model = Model("gemma2", use_server=False)
model.load()
response = model.generate("Hello!")
model.unload()
```

## 🏷️ Model Aliases

Use simple names instead of full HuggingFace paths:

```python
# These all work!
Model("llama3")          # → bartowski/Meta-Llama-3-8B-Instruct-GGUF
Model("llama3.1")        # → bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
Model("qwencoder")       # → bartowski/Qwen2.5-Coder-7B-Instruct-GGUF
Model("gemma2")          # → bartowski/gemma-2-9b-it-GGUF
Model("mistral")         # → bartowski/Mistral-7B-Instruct-v0.3-GGUF
Model("phi3.5")          # → bartowski/Phi-3.5-mini-instruct-GGUF
Model("deepseek-coder")  # → bartowski/DeepSeek-Coder-V2-Instruct-GGUF
```

**50+ aliases** for Llama, Qwen, Gemma, Mistral, Phi, DeepSeek, Yi, and more!

```bash
# See all available aliases
oprel list-models
```

## 💬 Conversation Memory

Built-in multi-turn conversation support:

```python
model = Model("llama3")

# With system prompt
response = model.generate(
    "What's 2+2?",
    conversation_id="math-tutor",
    system_prompt="You are a helpful math tutor. Be encouraging!"
)

# Continue the conversation
response = model.generate(
    "Now what's 10+10?",
    conversation_id="math-tutor"
)

# Reset conversation but keep system prompt
response = model.generate(
    "Start fresh",
    conversation_id="math-tutor",
    reset_conversation=True
)
```

## 🖥️ CLI Reference

```bash
# Server management
oprel serve              # Start daemon on port 11434
oprel serve --port 8080  # Custom port
oprel stop               # Stop the server
oprel models             # List loaded models

# Generation
oprel run <model> "prompt"           # Quick generation
oprel run llama3 "Hello" --stream    # Streaming output

# Chat
oprel chat <model>                   # Interactive chat
oprel chat llama3 --system "..."     # With system prompt

# Model discovery
oprel list-models                    # All 50+ aliases
oprel search llama                   # Search aliases

# Cache management
oprel cache list                     # Show cached models
oprel cache clear                    # Clear all cache
oprel cache delete <model>           # Delete specific model
```

## 🛡️ Key Features

### Memory Protection
Unlike Ollama, Oprel won't freeze your computer:

```python
from oprel import Model
from oprel.core.exceptions import MemoryError

model = Model("llama3", max_memory_mb=4096)

try:
    model.generate("Write a novel...")
except MemoryError as e:
    print(e)  # "Model exceeded 4GB limit. Try Q4_K_M quantization."
```

### Streaming Responses

```python
for token in model.generate("Tell me a story", stream=True):
    print(token, end="", flush=True)
```

### Context Manager

```python
with Model("llama3") as model:
    response = model.generate("Hello!")
# Auto cleanup
```

## 📊 Performance

| Mode | First Load | Subsequent Loads |
|------|------------|------------------|
| **Server Mode** (default) | ~2 minutes | **~2 seconds** |
| **Direct Mode** | ~2 minutes | ~2 minutes |

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