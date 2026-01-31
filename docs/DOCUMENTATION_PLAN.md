# Oprel SDK Documentation Plan

## Complete Documentation Structure

```
docs/
├── index.md                    # Home/Landing
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── first-model.md
├── cli-reference/
│   ├── overview.md
│   ├── run.md
│   ├── chat.md
│   ├── generate.md
│   ├── serve.md
│   ├── models.md
│   ├── stop.md
│   ├── info.md
│   ├── list-models.md
│   ├── search.md
│   └── cache.md
├── python-api/
│   ├── overview.md
│   ├── model-class.md
│   ├── config-class.md
│   ├── client-api.md          # Ollama-compatible
│   └── exceptions.md
├── models/
│   ├── supported-models.md
│   ├── model-aliases.md
│   └── adding-models.md
├── guides/
│   ├── memory-optimization.md
│   ├── server-mode.md
│   ├── direct-mode.md
│   ├── chat-templates.md
│   └── troubleshooting.md
├── architecture/
│   ├── overview.md
│   ├── daemon-server.md
│   └── backends.md

```

---

## 1. CLI Reference - Complete Command List

### Global Options
| Option | Description |
|--------|-------------|
| `--version` | Show oprel version |
| `--verbose` | Enable debug logging |
| `--quiet` | Suppress all logging |

### Commands

#### `oprel run <model> [prompt]`
Fast inference using server mode. Models stay loaded for instant follow-up.

| Argument/Option | Type | Default | Description |
|-----------------|------|---------|-------------|
| `model` | required | - | Model alias or full ID |
| `prompt` | optional | - | Omit for interactive mode |
| `--quantization` | string | auto | Q4_K_M, Q8_0, etc. |
| `--max-tokens` | int | 512 | Max tokens to generate |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--stream` | flag | true | Stream response |
| `--no-stream` | flag | false | Disable streaming |
| `--system` | string | - | System prompt |
| `--allow-low-quality` | flag | false | Allow Q2_K and below |

**Interactive Commands:**
- `/exit`, `/bye`, `/quit` - Exit chat
- `/reset` - Clear conversation history
- `/?` - Show help

---

#### `oprel chat <model>`
Interactive chat with conversation history.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | required | - | Model ID |
| `--quantization` | string | auto | Quantization level |
| `--max-memory` | int | - | Max memory in MB |
| `--stream` | flag | true | Stream responses |
| `--system` | string | - | System prompt |
| `--no-server` | flag | false | Use direct mode |
| `--allow-low-quality` | flag | false | Allow Q2_K |

---

#### `oprel generate <model> <prompt>`
Single-shot text generation.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | required | - | Model ID |
| `prompt` | required | - | Input prompt |
| `--quantization` | string | auto | Quantization level |
| `--max-memory` | int | - | Max memory MB |
| `--max-tokens` | int | 512 | Max tokens |
| `--temperature` | float | 0.7 | Temperature |
| `--stream` | flag | false | Stream output |
| `--no-server` | flag | false | Direct mode |

---

#### `oprel serve`
Start the Oprel daemon server.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | 127.0.0.1 | Host to bind |
| `--port` | int | 11434 | Port number |

---

#### `oprel models`
List models loaded in server.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | 127.0.0.1 | Server host |
| `--port` | int | 11434 | Server port |

---

#### `oprel stop`
Stop server and unload all models.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | 127.0.0.1 | Server host |
| `--port` | int | 11434 | Server port |

---

#### `oprel info`
Show system hardware information. No options.

---

#### `oprel list-models`
List all available model aliases. No options.

---

#### `oprel search <query>`
Search for models by name.

| Option | Type | Description |
|--------|------|-------------|
| `query` | required | Search term |

---

#### `oprel cache list`
List all cached models.

#### `oprel cache clear`
Clear all cached models.

| Option | Description |
|--------|-------------|
| `--yes` | Skip confirmation |

#### `oprel cache delete <model_name>`
Delete specific model from cache.

---

## 2. Python API Reference

### Core Classes

#### `oprel.Model`
```python
class Model:
    def __init__(
        self,
        model_id: str,                    # Model alias or full ID
        quantization: Optional[str] = None,  # Q4_K_M, Q8_0, etc.
        max_memory_mb: Optional[int] = None, # Memory limit
        use_server: bool = True,          # Use daemon server
        allow_low_quality: bool = False,  # Allow Q2_K
    )
    
    def load(self) -> None
    def unload(self) -> None
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]
```

#### `oprel.Config`
```python
class Config:
    model_id: str
    quantization: Optional[str]
    max_memory_mb: Optional[int]
    ctx_size: int = 4096
    batch_size: int = 512
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    kv_cache_type: str = "f16"      # f16, q8_0, q4_0
    flash_attention: bool = False
    mmap: bool = True
```

### Ollama-Compatible API

#### `oprel.Client`
```python
class Client:
    def __init__(self, host: str = "http://127.0.0.1:11434")
    
    def chat(
        self,
        model: str,
        messages: List[dict],
        stream: bool = False,
        **kwargs
    ) -> ChatResponse
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> GenerateResponse
    
    def list(self) -> ListResponse
    def show(self, model: str) -> ShowResponse
    def pull(self, model: str) -> None
    def delete(self, model: str) -> None
```

#### Standalone Functions
```python
oprel.chat(model, messages, stream=False)
oprel.generate(model, prompt, stream=False)
oprel.list()
oprel.show(model)
oprel.pull(model)
oprel.delete(model)
```

### Utility Functions
```python
oprel.download_model(model_id, quantization=None)
oprel.get_hardware_info()  # Returns dict with CPU, RAM, GPU info
```

### Exceptions
```python
oprel.OprelError          # Base exception
oprel.ModelNotFoundError  # Model not found
oprel.MemoryError         # Out of memory
oprel.BackendError        # Backend failure
```

### Response Models
```python
ChatResponse(message, done, model, created_at, ...)
GenerateResponse(response, done, model, ...)
ListResponse(models: List[ModelInfo])
ShowResponse(modelfile, parameters, ...)
Message(role, content)
ModelInfo(name, size, modified_at, ...)
```

---

## 3. Supported Models (25+ Aliases)

### Llama Family (Meta)
| Alias | Points To |
|-------|-----------|
| `llama3` | lmstudio-community/Llama-3.2-3B-Instruct-GGUF |
| `llama3.1` | lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF |
| `llama3.2` | lmstudio-community/Llama-3.2-3B-Instruct-GGUF |
| `llama3.2:1b` | lmstudio-community/Llama-3.2-1B-Instruct-GGUF |
| `llama3.3` | lmstudio-community/Llama-3.3-70B-Instruct-GGUF |
| `codellama` | TheBloke/CodeLlama-7B-Instruct-GGUF |
| `codellama:13b` | TheBloke/CodeLlama-13B-Instruct-GGUF |

### Gemma Family (Google)
| Alias | Points To |
|-------|-----------|
| `gemma2` | bartowski/gemma-2-9b-it-GGUF |
| `gemma2:2b` | bartowski/gemma-2-2b-it-GGUF |
| `gemma2:27b` | bartowski/gemma-2-27b-it-GGUF |

### Qwen Family (Alibaba)
| Alias | Points To |
|-------|-----------|
| `qwen2.5` | Qwen/Qwen2.5-7B-Instruct-GGUF |
| `qwen2.5:3b` | Qwen/Qwen2.5-3B-Instruct-GGUF |
| `qwen2.5:14b` | Qwen/Qwen2.5-14B-Instruct-GGUF |
| `qwencoder` | Qwen/Qwen2.5-Coder-7B-Instruct-GGUF |
| `qwencoder:1.5b` | Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF |

### Mistral Family
| Alias | Points To |
|-------|-----------|
| `mistral` | mistralai/Mistral-7B-Instruct-v0.3-GGUF |
| `mixtral` | mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF |

### Phi Family (Microsoft)
| Alias | Points To |
|-------|-----------|
| `phi3` | microsoft/Phi-3-mini-4k-instruct-gguf |
| `phi3:medium` | microsoft/Phi-3-medium-4k-instruct-gguf |

### DeepSeek
| Alias | Points To |
|-------|-----------|
| `deepseek` | TheBloke/deepseek-coder-6.7B-instruct-GGUF |
| `deepseek:33b` | TheBloke/deepseek-coder-33B-instruct-GGUF |

### Other
| Alias | Points To |
|-------|-----------|
| `yi` | TheBloke/Yi-6B-Chat-GGUF |
| `yi:34b` | TheBloke/Yi-34B-Chat-GGUF |
| `smollm` | HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF |

---

## 4. Guides

### Memory Optimization Guide
- KV Cache Quantization (f16 → q8_0 → q4_0)
- Flash Attention (when supported)
- Memory mapping (mmap)
- GPU layer offloading
- Context size tuning

### Server vs Direct Mode
- When to use each
- Auto-starting behavior
- Model caching benefits
- Process isolation

### Chat Templates
- Supported formats: ChatML, Llama2, Llama3, Gemma
- How detection works
- Custom templates

---

## 5. Architecture Documentation

### Process Model
- Python daemon (FastAPI on :11434)
- Child process: llama-server binary
- HTTP communication between daemon and backend
- Model caching in RAM

### Backend Support
- llama.cpp (primary, fully working)
- vLLM (placeholder)
- ExLlama (placeholder)

---

# Brutally Honest Oprel vs Ollama Comparison

## Rating: 4/10 (Pre-Alpha Quality)

### What Oprel Does Right
| Feature | Status |
|---------|--------|
| Model aliases like Ollama | ✅ Works |
| Chat templates | ✅ Works |
| Python API | ✅ Works |
| Ollama-compatible Client | ✅ Works |
| Memory optimization options | ✅ Added |
| CLI basics | ✅ Works |

### What's Broken or Missing (The Ugly Truth)

#### 🔴 CRITICAL ISSUES

**1. No Real Model Management**
- Ollama: `ollama pull llama3` → downloads, verifies, tracks versions
- Oprel: Downloads GGUF from HuggingFace, zero version control
- **No `Modelfile` equivalent** - you can't customize models
- **No layers** - can't create specialized models from base

**2. No Model Creation**
- Ollama: `ollama create mymodel -f ./Modelfile`
- Oprel: Nothing. Zero. Can't create custom models.

**3. No GPU Acceleration (Currently Broken)**
- CUDA binary crashes with "device busy"
- Forced to CPU-only fallback
- Ollama: Just works with GPU

**4. No Concurrent Requests**
- Ollama: Handles multiple simultaneous users
- Oprel: One request at a time, blocks everything

**5. No REST API Parity**
- Ollama has `/api/push`, `/api/copy`, `/api/embeddings`
- Oprel: Missing most endpoints

**6. No Embeddings**
- Ollama: `ollama.embeddings("model", "text")`
- Oprel: Not implemented

**7. No Model Verification**
- Ollama: SHA256 checksums, integrity verification
- Oprel: Downloads and trusts blindly

#### 🟡 MEDIUM ISSUES

**8. Memory Usage Still Higher**
- Even with KV cache optimization, not as efficient as Ollama's Go runtime
- Ollama's embedded approach has less overhead

**9. No systemd/Service Integration**
- Ollama: `systemctl enable ollama`
- Oprel: Run it yourself, hope it doesn't crash

**10. No Docker Image**
- Ollama: `docker run ollama/ollama`
- Oprel: Nothing

**11. No OpenAI-Compatible API**
- Ollama: Full `/v1/chat/completions` compatible
- Oprel: Only partial Ollama API

**12. No Model Tags**
- Ollama: `llama3:8b-instruct-q4_K_M` - specific variant
- Oprel: Just downloads whatever quantization it finds

#### 🟢 Minor Issues

- No WebUI
- No telemetry opt-out toggle
- Limited error messages
- No progress bars for downloads

---

## Feature Comparison Matrix

| Feature | Ollama | Oprel | Gap |
|---------|--------|-------|-----|
| Model Pull | ✅ | ⚠️ Basic | Large |
| Model Push | ✅ | ❌ | Total |
| Model Create | ✅ | ❌ | Total |
| Model Copy | ✅ | ❌ | Total |
| Modelfile | ✅ | ❌ | Total |
| Chat | ✅ | ✅ | None |
| Generate | ✅ | ✅ | None |
| Embeddings | ✅ | ❌ | Total |
| GPU Support | ✅ | ⚠️ Broken | Critical |
| Multi-user | ✅ | ❌ | Total |
| REST API | ✅ Full | ⚠️ Partial | Large |
| OpenAI API | ✅ | ❌ | Total |
| Docker | ✅ | ❌ | Total |
| Memory Efficiency | ✅ Excellent | ⚠️ OK | Medium |
| Model Registry | ✅ ollama.com | ❌ HuggingFace only | Large |
| Version Control | ✅ | ❌ | Total |
| Integrity Check | ✅ SHA256 | ❌ | Total |
| Service Mode | ✅ systemd | ❌ | Medium |

---

## Honest Assessment

### Why Would Anyone Use Oprel Over Ollama?

**Currently? Almost no reason.**

The only valid use cases:
1. You need Python-native integration (not just calling subprocess)
2. You want to modify the internals (Ollama is Go, harder to hack)
3. You're building it yourself for learning

### What Oprel Needs to Be Competitive

1. **Working GPU support** (priority #1)
2. **Embeddings API**
3. **OpenAI-compatible endpoint**
4. **Model versioning and integrity checks**
5. **Concurrent request handling**
6. **Docker image**
7. **Proper documentation website**

### The Reality

Oprel is a **proof of concept**, not a production tool. Ollama has:
- 100,000+ GitHub stars
- Dedicated team
- Native binaries
- Production users

Oprel has:
- A few Python files
- One developer
- Broken GPU support
- Zero production deployments

---

## Recommendation

If you're choosing for **production**: Use Ollama. No contest.

If you're choosing for **learning/hacking**: Oprel's Python codebase is readable and hackable.

If you're building **Oprel**: Focus on the gaps above. Otherwise, you're just a worse Ollama.

---

## Documentation Website Structure (If Built)

```
oprel.dev/
├── /                    # Hero: "The SQLite of LLMs" + quick install
├── /docs/               # Full documentation
├── /models/             # Model library browser
├── /api/                # API reference
├── /compare/            # vs Ollama (honest version)
└── /blog/               # Development updates
```

### Tech Stack for Docs Site
- **Framework**: VitePress or Docusaurus
- **Hosting**: GitHub Pages or Vercel (free)
- **Search**: Algolia DocSearch (free for OSS)

---

## Files That Need Documentation

| File | Purpose | Doc Priority |
|------|---------|--------------|
| `oprel/__init__.py` | Public exports | HIGH |
| `oprel/core/model.py` | Model class | HIGH |
| `oprel/core/config.py` | Config class | HIGH |
| `oprel/client_api.py` | Ollama-compatible API | HIGH |
| `oprel/cli/main.py` | CLI commands | HIGH |
| `oprel/downloader/aliases.py` | Model aliases | MEDIUM |
| `oprel/utils/chat_templates.py` | Chat formatting | MEDIUM |
| `oprel/server/daemon.py` | Server internals | LOW |
| `oprel/runtime/backends/llama_cpp.py` | Backend | LOW |

---

## Summary

**Oprel today**: A hackable Python wrapper around llama.cpp with broken GPU support.

**Ollama today**: Production-ready local AI runtime with massive community.

**Gap to close**: Enormous. Probably 6-12 months of full-time work minimum.

**Should you use it?** Only if you're building/learning. Never for production.
