# Architecture Overview

This document describes the internal architecture and design of the Oprel SDK.

## System Architecture

Oprel is designed as a modular system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   CLI Tool  │  │  Python API  │  │  Ollama API      │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Model Class │  │    Config    │  │   Exceptions     │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Server    │  │  Downloader  │  │   Telemetry      │  │
│  │  (FastAPI)  │  │              │  │                  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Process   │  │   Backends   │  │    Binaries      │  │
│  │  Manager    │  │  (llama.cpp) │  │   (llama-server) │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model API (oprel/core/model.py)

The central interface for loading and running language models.

**Responsibilities:**
- Model lifecycle management (load/unload)
- Generation request handling
- Server mode coordination
- Conversation memory management

**Key Classes:**
- `Model`: Main user-facing class

**Operating Modes:**

**Server Mode (default):**
```python
model = Model("qwencoder", use_server=True)
response = model.generate("prompt")
```

- Model persists in separate daemon process
- Fast subsequent requests (2-5 seconds)
- Automatic crash recovery
- Shared across multiple client instances

**Direct Mode:**
```python
model = Model("qwencoder", use_server=False)
model.load()
response = model.generate("prompt")
model.unload()
```

- Model loaded directly in current process
- No server dependency
- Full control over lifecycle
- Suitable for single-use or embedded scenarios

### 2. Server Daemon (oprel/server/daemon.py)

FastAPI-based HTTP server for persistent model hosting.

**Endpoints:**

- `POST /load`: Load a model into memory
- `POST /generate`: Generate text from loaded model
- `GET /models`: List currently loaded models
- `POST /unload`: Unload a model from memory
- `GET /health`: Server health check

**Features:**

- Automatic backend health monitoring
- Crash detection and recovery
- Conversation state management
- Multiple concurrent model support

**Implementation:**

```python
# Server lifecycle
app = FastAPI()
loaded_models: Dict[str, Model] = {}

@app.post("/generate")
async def generate(request: GenerateRequest):
    model = loaded_models[request.model]
    if not model.is_running():
        model.reload()  # Auto-recovery
    return model.generate(request.prompt)
```

### 3. Client Layer (oprel/client/)

Communication interfaces for different transport protocols.

**HTTPClient** (client/http.py):
- REST API communication with server
- Connection pooling
- Health checks
- Error handling

**PipeClient** (client/pipe.py):
- Named pipe communication (Windows)
- Low-latency local IPC

**SocketClient** (client/socket.py):
- Unix socket communication (Linux/macOS)
- Alternative to HTTP for local communication

**BaseClient** (client/base.py):
- Abstract base class
- Common interface definition

### 4. Downloader (oprel/downloader/)

Model download and caching system.

**Components:**

**Hub** (downloader/hub.py):
- HuggingFace Hub integration
- Model search and discovery
- Metadata retrieval

**Cache** (downloader/cache.py):
- Local model storage
- Cache management (list, delete, clear)
- Disk space monitoring

**Verification** (downloader/verification.py):
- File integrity checks
- Checksum validation
- Corruption detection

**Aliases** (downloader/aliases.py):
- 50+ predefined model shortcuts
- Mapping to HuggingFace repositories
- Version management

**Download Process:**

```
1. User requests model "qwencoder"
2. Alias resolver → bartowski/Qwen2.5-Coder-7B-Instruct-GGUF
3. Check local cache
4. If not cached:
   a. Connect to HuggingFace Hub
   b. Download GGUF file
   c. Verify integrity
   d. Store in cache
5. Return local path
```

### 5. Runtime Layer (oprel/runtime/)

Backend process management and execution.

**Process Manager** (runtime/process.py):
- Subprocess creation with hidden windows (Windows)
- Process lifecycle tracking
- Health monitoring (is_running())
- Automatic restart on crash

**Backends** (runtime/backends/):

**Base Backend** (backends/base.py):
- Abstract interface for inference engines

**llama.cpp Backend** (backends/llama_cpp.py):
- Primary inference engine
- GGUF model support
- GPU acceleration (CUDA/Metal)
- Configurable context size, batch size

**ExLlama Backend** (backends/exllama.py):
- Alternative engine (experimental)

**vLLM Backend** (backends/vllm.py):
- High-throughput inference (experimental)

**Binaries** (runtime/binaries/):
- llama-server executable management
- Platform-specific builds
- Version management
- Automatic installation

**Process Creation (Windows):**

```python
import subprocess

process = subprocess.Popen(
    [binary_path, "--model", model_path, "--port", str(port)],
    creationflags=subprocess.CREATE_NO_WINDOW,  # Hide CMD window
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
```

### 6. Telemetry (oprel/telemetry/)

Hardware detection and optimization.

**Hardware Detection** (telemetry/hardware.py):
- CPU capabilities
- GPU detection (NVIDIA/AMD/Apple)
- Memory available
- Platform identification

**Recommender** (telemetry/recommender.py):
- Quantization selection based on available RAM
- GPU layer count optimization
- Context size recommendations

**Benchmarks** (telemetry/benchmarks.py):
- Inference speed measurement
- Token throughput
- Memory usage tracking

### 7. Ollama API (oprel/client_api.py, oprel/api_models.py)

Full Ollama compatibility layer.

**client_api.py:**
- `chat()`, `generate()`, `list()`, `show()` functions
- `Client` and `AsyncClient` classes
- Streaming support
- Error translation

**api_models.py:**
- `ChatResponse`, `GenerateResponse`
- `Message`, `ModelInfo`
- `ListResponse`, `ShowResponse`
- Dual access (attribute and dictionary)

**Design:**

```python
# Module functions delegate to Model class
def chat(model: str, messages: List[Dict], stream: bool = False):
    m = Model(model)
    # Convert messages to conversation
    response = m.generate(...)
    return ChatResponse(...)

# Client class wraps module functions
class Client:
    def chat(self, model: str, messages: List[Dict]):
        return chat(model, messages)
```

### 8. CLI (oprel/cli/main.py)

Command-line interface using argparse.

**Commands:**

- `serve`: Start server daemon
- `run`: Generate text (one-shot or interactive)
- `chat`: Interactive chat session
- `models`: List loaded models
- `list-models`: Show all available aliases
- `search`: Search for models
- `cache`: Cache management

**Interactive Mode:**

```python
def _run_interactive(model: Model, system_prompt: Optional[str]):
    conversation_id = f"cli-{uuid.uuid4()}"
    
    while True:
        user_input = input(">>> ")
        
        if user_input in ["/exit", "/bye", "/quit"]:
            break
        elif user_input == "/reset":
            conversation_id = f"cli-{uuid.uuid4()}"
            continue
        
        response = model.generate(
            user_input,
            conversation_id=conversation_id,
            system_prompt=system_prompt
        )
        print(response)
```

## Data Flow

### Generation Request Flow (Server Mode)

```
1. User: model.generate("What is Python?")
   │
2. Model checks if server running
   │
3. If not running: Start daemon process
   │
4. HTTPClient.post("/generate", {
       model: "qwencoder",
       prompt: "What is Python?"
   })
   │
5. Server receives request
   │
6. Server checks if model loaded
   │
7. If not loaded: Download and load model
   │
8. Server checks backend health (is_running())
   │
9. If crashed: Reload backend
   │
10. Backend generates text
    │
11. Server returns response
    │
12. HTTPClient returns to Model
    │
13. Model returns to user
```

### Generation Request Flow (Direct Mode)

```
1. User: model.load()
   │
2. Download model if needed
   │
3. Start llama-server subprocess
   │
4. Wait for server ready
   │
5. User: model.generate("prompt")
   │
6. Send request to local backend
   │
7. Backend generates text
   │
8. Return response
   │
9. User: model.unload()
   │
10. Terminate subprocess
```

## Configuration System

### Configuration Hierarchy

```
1. Default values (hardcoded)
   ↓
2. Configuration file (config.yaml)
   ↓
3. Environment variables
   ↓
4. Constructor arguments (highest priority)
```

### Configuration Files

**config.yaml:**
```yaml
cache_dir: ~/.cache/oprel
binary_dir: ~/.oprel/binaries
default_max_memory_mb: 8192
ctx_size: 4096
batch_size: 512
n_threads: 8
n_gpu_layers: -1
```

### Environment Variables

- `OPREL_CACHE_DIR`: Override cache directory
- `OPREL_BINARY_DIR`: Override binary directory
- `OPREL_SERVER_PORT`: Default server port
- `OPREL_LOG_LEVEL`: Logging verbosity

## Error Handling

### Exception Hierarchy

```
OprelError (base)
├── ModelNotFoundError
│   └── Model file or alias not found
├── DownloadError
│   └── HuggingFace download failed
├── BackendError
│   ├── Backend process crashed
│   └── Backend initialization failed
├── MemoryError
│   └── Insufficient RAM for model
└── ValidationError
    └── Invalid parameters
```

### Error Recovery

**Backend Crash Recovery:**

```python
# In server/daemon.py
@app.post("/generate")
async def generate(request: GenerateRequest):
    model = loaded_models[request.model]
    
    if not model.is_running():
        logger.warning(f"Backend crashed, reloading {request.model}")
        model.reload()
    
    return model.generate(request.prompt)
```

**Memory Management:**

```python
# Check available memory before loading
available_mb = psutil.virtual_memory().available / (1024 * 1024)

if model_size_mb > available_mb:
    raise MemoryError(
        f"Model requires {model_size_mb}MB but only "
        f"{available_mb}MB available"
    )
```

## Performance Optimization

### Server Mode Caching

- Models remain loaded in daemon process
- Eliminates 60-120 second load time on subsequent requests
- Reduces to 2-5 second response time

### GPU Acceleration

- Automatic GPU detection
- Dynamic layer offloading based on VRAM
- Fallback to CPU if GPU unavailable

### Quantization

| Level | Size | Quality | Use Case |
|-------|------|---------|----------|
| Q2_K | 2-3GB | Low | Memory-constrained |
| Q3_K_M | 3-4GB | Medium | Balanced |
| Q4_K_M | 4-5GB | Good | Recommended |
| Q5_K_M | 5-6GB | Very Good | High quality |
| Q6_K | 6-7GB | Excellent | Maximum quality |
| Q8_0 | 7-8GB | Near-original | Archival |

### Batch Processing

- Configurable batch size for token generation
- Trade-off between latency and throughput

## Security Considerations

### Process Isolation

- Backend runs in separate subprocess
- Limited access to system resources
- Automatic cleanup on termination

### Network Security

- Server binds to localhost by default
- No authentication (local use only)
- Consider reverse proxy for remote access

### File System Security

- Models downloaded only from HuggingFace
- Cache directory permissions
- Temporary file cleanup

## Testing Architecture

### Test Organization

```
tests/
├── unit/
│   ├── test_client_api.py      # Ollama API tests
│   ├── test_http_client.py     # HTTP client tests
│   ├── test_downloader_full.py # Download tests
│   ├── test_process_full.py    # Process mgmt tests
│   └── test_telemetry_full.py  # Hardware tests
├── integration/
│   ├── test_backends_full.py   # Backend integration
│   └── test_end_to_end_full.py # E2E workflows
└── fixtures/
    ├── mock_models/            # Test model files
    └── mock_utils.py           # Mock utilities
```

### Testing Strategy

**Unit Tests:**
- Mock external dependencies
- Test individual components
- Fast execution

**Integration Tests:**
- Test component interactions
- Use real backends where possible
- Slower but comprehensive

**Fixtures:**
- `mock_config`: Configuration for tests
- `mock_model_path`: Path to test GGUF file
- `temp_cache_dir`: Temporary cache directory

## Extension Points

### Adding New Backends

1. Extend `BaseBackend` in `runtime/backends/base.py`
2. Implement required methods:
   - `start()`
   - `stop()`
   - `generate()`
   - `is_healthy()`
3. Register in backend factory

### Adding New Model Aliases

1. Edit `oprel/downloader/aliases.py`
2. Add entry to `MODEL_ALIASES` dict:
```python
"mymodel": {
    "repo": "user/model-name-GGUF",
    "filename": "model-Q4_K_M.gguf"
}
```

### Adding New Clients

1. Extend `BaseClient` in `client/base.py`
2. Implement transport-specific methods
3. Register in client factory

## Future Architecture Considerations

### Planned Enhancements

1. Multi-model Server: Support multiple models loaded simultaneously
2. Distributed Inference: Split large models across machines
3. Caching Layer: Redis integration for conversation state
4. Authentication: API key support for remote access
5. Monitoring: Prometheus metrics export
6. Load Balancing: Multiple backend instances per model

### Scalability

Current design supports:
- Single machine deployment
- Local-only access
- One model per server instance

For production scale:
- Add load balancer layer
- Implement distributed state management
- Add authentication/authorization
- Containerize components

## Debugging

### Logging

```python
import logging
from oprel.utils.logging import setup_logging

setup_logging(level=logging.DEBUG)
```

Logs include:
- Model loading/unloading events
- Backend process lifecycle
- HTTP request/response details
- Error stack traces

### Process Monitoring

```bash
# List running llama-server processes (Windows)
Get-Process | Where-Object { $_.ProcessName -like "*llama-server*" }

# Monitor server logs
oprel serve --log-level DEBUG
```

### Common Issues

1. Model not loading: Check cache directory permissions
2. Backend crashes: Increase memory limit or use lighter quantization
3. Slow generation: Enable GPU or reduce context size
4. Server not starting: Check port availability

## Conclusion

Oprel architecture balances simplicity with flexibility. The modular design allows for easy extension while maintaining a clean separation of concerns. Server mode provides Ollama-like performance, while direct mode offers full control for embedded use cases.
