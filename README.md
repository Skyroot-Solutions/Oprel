# Oprel SDK

**The SQLite of LLMs** - Local-first AI runtime for Python

[![PyPI version](https://badge.fury.io/py/oprel.svg)](https://pypi.org/project/oprel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Oprel SDK makes running AI models locally as simple as opening a database. No Docker, no daemons, no hassle.

```python
from oprel import Model

with Model("TheBloke/Llama-2-7B-GGUF") as model:
    response = model.generate("What is Python?")
    print(response)
```

## 🎯 Why Oprel?

| Problem | Solution |
|---------|----------|
| **"Ollama requires a background daemon"** | Oprel is a library - import and go |
| **"Models crash and freeze my computer"** | Built-in OOM protection with graceful errors |
| **"I can't ship apps that need Ollama installed"** | Oprel is just a pip dependency |
| **"Setup is complicated"** | Auto-detects hardware, downloads binaries, picks optimal settings |

## 🚀 Quick Start

### Installation

```bash
pip install oprel
```

That's it. No additional setup required.

### Basic Usage

```python
from oprel import Model

# Simple generation
model = Model("TheBloke/Llama-2-7B-Chat-GGUF")
response = model.generate("Explain quantum computing")
print(response)
model.unload()
```

### Recommended: Context Manager

```python
with Model("TheBloke/Llama-2-7B-Chat-GGUF") as model:
    response = model.generate("Write a haiku about Python")
    print(response)
# Automatically cleaned up
```

### Streaming Responses

```python
with Model("TheBloke/Llama-2-7B-Chat-GGUF") as model:
    for token in model.generate("Tell me a story", stream=True):
        print(token, end="", flush=True)
```

## 🛡️ Key Features

### 1. **Zero Configuration**
- Auto-detects CPU/GPU capabilities
- Automatically selects optimal quantization (Q4_K_M, Q8_0, etc.)
- Downloads and caches models on first use

### 2. **Memory Protection**
Unlike other tools, Oprel won't freeze your computer:

```python
from oprel import Model, MemoryError

model = Model("large-model", max_memory_mb=4096)

try:
    model.generate("Write a novel...")
except MemoryError as e:
    print(e)  # "Model exceeded 4GB limit. Try Q4_K_M quantization."
```

### 3. **Embedded in Your Apps**
Perfect for desktop applications:

```python
# Your PyQt/Tkinter/Electron app
import oprel

class MyApp:
    def __init__(self):
        self.model = oprel.Model("your-model")
    
    def on_user_query(self, text):
        return self.model.generate(text)
```

### 4. **100% Offline**
- No API keys required
- No data leaves your machine
- Works on airplanes, in secure environments, anywhere

## 📊 Performance

Oprel uses the same high-performance engines as other tools:
- **llama.cpp** for CPU/Metal inference
- **vLLM** for GPU server workloads (coming soon)
- **ExLlamaV2** for NVIDIA GPUs (coming soon)

**The difference:** Better developer experience, not raw speed.

## 🔧 Advanced Usage

### Custom Quantization

```python
model = Model(
    "TheBloke/Llama-2-70B-Chat-GGUF",
    quantization="Q8_0",      # Higher quality
    max_memory_mb=16384       # 16GB limit
)
```

### Pre-loading Models

```python
model = Model("your-model")
model.load()  # Download and start now

# Later, instant response
response = model.generate("Hello")
```

### Hardware Detection

```python
from oprel import get_hardware_info

info = get_hardware_info()
print(f"RAM: {info['ram_total_gb']}GB")
print(f"GPU: {info.get('gpu_name', 'None')}")
```

## 🗂️ Supported Models

Oprel works with any GGUF model from HuggingFace:

- **Llama 2** - `TheBloke/Llama-2-7B-Chat-GGUF`
- **Mistral** - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- **Mixtral** - `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF`
- **Phi** - `microsoft/phi-2-GGUF`
- And thousands more...

## 🆚 Comparison

| Feature | Ollama | Oprel |
|---------|--------|-------|
| Installation | Separate daemon | `pip install` |
| Usage | Background service | Python library |
| Memory protection | ❌ | ✅ Graceful OOM handling |
| Desktop apps | Requires users to install Ollama | Just a dependency |
| API | HTTP (slower) | Direct IPC (faster) |
| Ideal for | Personal chatbots | Embedded AI features |

## 🛠️ Requirements

- **Python**: 3.9+
- **OS**: macOS, Linux, Windows
- **RAM**: 4GB minimum (8GB+ recommended)
- **GPU**: Optional (CUDA/Metal auto-detected)

## 📚 Documentation

- [API Reference](https://oprel.readthedocs.io/api)
- [Troubleshooting](https://oprel.readthedocs.io/troubleshooting)
- [Architecture](https://oprel.readthedocs.io/architecture)

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - High-performance inference engine
- [HuggingFace Hub](https://huggingface.co) - Model distribution

---

**Made with ❤️ for developers who want local AI without the hassle**