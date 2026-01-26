# API Reference

Complete API documentation for the Oprel SDK.

## Table of Contents

- [Model API](#model-api)
- [Ollama-Compatible API](#ollama-compatible-api)
- [Configuration](#configuration)
- [Client Classes](#client-classes)
- [Response Models](#response-models)
- [Exceptions](#exceptions)

## Model API

### Model Class

Main class for loading and running language models.

```python
from oprel import Model
```

#### Constructor

```python
Model(
    model_identifier: str,
    quantization: Optional[str] = None,
    max_memory_mb: Optional[int] = None,
    backend: str = "llama.cpp",
    use_server: bool = True,
    config: Optional[Config] = None
)
```

**Parameters:**

- `model_identifier` (str): Model name or alias (e.g., "qwencoder", "llama3")
- `quantization` (Optional[str]): Quantization level ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"). Auto-selected if None.
- `max_memory_mb` (Optional[int]): Maximum memory limit in megabytes
- `backend` (str): Backend engine, currently only "llama.cpp" supported
- `use_server` (bool): Use server mode for persistent caching (default: True)
- `config` (Optional[Config]): Custom configuration object

**Example:**

```python
model = Model("qwencoder", quantization="Q4_K_M", max_memory_mb=4096)
```

#### Methods

##### load()

Load the model into memory.

```python
model.load() -> None
```

Only required in direct mode (`use_server=False`). Server mode loads automatically.

**Raises:**
- `ModelNotFoundError`: Model file not found
- `MemoryError`: Insufficient memory
- `BackendError`: Backend initialization failed

**Example:**

```python
model = Model("qwencoder", use_server=False)
model.load()
```

##### generate()

Generate text from a prompt.

```python
model.generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    stream: bool = False,
    conversation_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    reset_conversation: bool = False
) -> Union[str, Iterator[str]]
```

**Parameters:**

- `prompt` (str): Input text prompt
- `max_tokens` (int): Maximum tokens to generate (default: 512)
- `temperature` (float): Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p` (float): Nucleus sampling threshold (default: 0.9)
- `top_k` (int): Top-k sampling value (default: 40)
- `stream` (bool): Enable streaming mode (default: False)
- `conversation_id` (Optional[str]): Conversation identifier for multi-turn chat
- `system_prompt` (Optional[str]): System prompt for conversation context
- `reset_conversation` (bool): Clear conversation history (default: False)

**Returns:**
- `str`: Generated text (if `stream=False`)
- `Iterator[str]`: Token iterator (if `stream=True`)

**Example:**

```python
# Basic generation
response = model.generate("What is Python?")

# With custom parameters
response = model.generate(
    "Explain quantum computing",
    max_tokens=1024,
    temperature=0.5
)

# Streaming
for token in model.generate("Write a story", stream=True):
    print(token, end='', flush=True)

# Multi-turn conversation
response1 = model.generate(
    "My name is Alice",
    conversation_id="chat-1",
    system_prompt="You are a helpful assistant."
)
response2 = model.generate(
    "What's my name?",
    conversation_id="chat-1"
)
```

##### unload()

Unload the model from memory.

```python
model.unload() -> None
```

Only applies in direct mode. Server mode manages lifecycle automatically.

**Example:**

```python
model = Model("qwencoder", use_server=False)
model.load()
response = model.generate("prompt")
model.unload()
```

##### is_loaded()

Check if model is currently loaded.

```python
model.is_loaded() -> bool
```

**Returns:**
- `bool`: True if loaded, False otherwise

##### Context Manager Support

```python
with Model("qwencoder") as model:
    response = model.generate("prompt")
# Automatic cleanup
```

## Ollama-Compatible API

Full compatibility with Ollama Python client API.

### Module-Level Functions

```python
from oprel import chat, generate, list, show, create, pull, delete
```

#### chat()

Chat with a model using message history.

```python
chat(
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    options: Optional[Dict] = None
) -> Union[ChatResponse, Iterator[ChatResponse]]
```

**Parameters:**

- `model` (str): Model name or alias
- `messages` (List[Dict]): List of message dictionaries with 'role' and 'content'
- `stream` (bool): Enable streaming mode
- `options` (Optional[Dict]): Generation options (temperature, max_tokens, etc.)

**Returns:**
- `ChatResponse`: Response object (if `stream=False`)
- `Iterator[ChatResponse]`: Response iterator (if `stream=True`)

**Example:**

```python
from oprel import chat

response = chat(
    model='qwencoder',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is Python?'}
    ]
)
print(response.message.content)

# Streaming
for chunk in chat(model='qwencoder', messages=[...], stream=True):
    print(chunk.message.content, end='', flush=True)
```

#### generate()

Generate text from a prompt.

```python
generate(
    model: str,
    prompt: str,
    stream: bool = False,
    options: Optional[Dict] = None
) -> Union[GenerateResponse, Iterator[GenerateResponse]]
```

**Parameters:**

- `model` (str): Model name or alias
- `prompt` (str): Input prompt
- `stream` (bool): Enable streaming mode
- `options` (Optional[Dict]): Generation options

**Returns:**
- `GenerateResponse`: Response object (if `stream=False`)
- `Iterator[GenerateResponse]`: Response iterator (if `stream=True`)

**Example:**

```python
from oprel import generate

response = generate(model='qwencoder', prompt='What is Python?')
print(response.response)
```

#### list()

List available model aliases.

```python
list() -> ListResponse
```

**Returns:**
- `ListResponse`: Object containing list of models

**Example:**

```python
from oprel import list as list_models

models = list_models()
for model in models.models:
    print(model.name)
```

#### show()

Show model information.

```python
show(model: str) -> ShowResponse
```

**Parameters:**

- `model` (str): Model name or alias

**Returns:**
- `ShowResponse`: Model information

**Example:**

```python
from oprel import show

info = show('qwencoder')
print(info.modelfile)
```

#### create()

Create a model (not yet implemented).

```python
create(model: str, modelfile: str) -> Dict
```

#### pull()

Download a model.

```python
pull(model: str) -> Dict
```

**Parameters:**

- `model` (str): Model name or alias

**Returns:**
- `Dict`: Download status

#### delete()

Delete a cached model.

```python
delete(model: str) -> Dict
```

**Parameters:**

- `model` (str): Model name or alias

**Returns:**
- `Dict`: Deletion status

### Client Class

Object-oriented interface for Ollama API.

```python
from oprel import Client

client = Client(host='http://localhost:11434')
```

#### Constructor

```python
Client(host: str = 'http://localhost:11434')
```

**Parameters:**

- `host` (str): Server URL

#### Methods

All methods match the module-level functions but use instance configuration.

```python
# Chat
response = client.chat(model='qwencoder', messages=[...])

# Generate
response = client.generate(model='qwencoder', prompt='...')

# List models
models = client.list()

# Show model info
info = client.show('qwencoder')

# Pull model
client.pull('qwencoder')

# Delete model
client.delete('qwencoder')
```

### AsyncClient Class

Asynchronous version of Client (not fully implemented).

```python
from oprel import AsyncClient

client = AsyncClient(host='http://localhost:11434')
```

## Response Models

### ChatResponse

Response from chat() function.

```python
class ChatResponse:
    model: str              # Model name
    created_at: str         # Timestamp
    message: Message        # Response message
    done: bool              # Completion status
```

**Attributes:**

- `model` (str): Model identifier
- `created_at` (str): ISO 8601 timestamp
- `message` (Message): Message object with role and content
- `done` (bool): True when generation complete

**Access Methods:**

```python
# Attribute access
print(response.message.content)

# Dictionary access
print(response['message']['content'])
```

### GenerateResponse

Response from generate() function.

```python
class GenerateResponse:
    model: str              # Model name
    created_at: str         # Timestamp
    response: str           # Generated text
    done: bool              # Completion status
```

### ListResponse

Response from list() function.

```python
class ListResponse:
    models: List[ModelInfo]  # List of available models
```

### ShowResponse

Response from show() function.

```python
class ShowResponse:
    modelfile: str          # Model configuration
    parameters: str         # Model parameters
    template: str           # Prompt template
```

### Message

Chat message object.

```python
class Message:
    role: str               # 'system', 'user', or 'assistant'
    content: str            # Message text
    images: Optional[List]  # Image data (optional)
```

### ModelInfo

Model information object.

```python
class ModelInfo:
    name: str               # Model name
    modified_at: str        # Last modified timestamp
    size: int               # Model size in bytes
```

## Configuration

### Config Class

Global configuration for Oprel.

```python
from oprel.core.config import Config

config = Config(
    cache_dir="/path/to/cache",
    binary_dir="/path/to/binaries",
    default_max_memory_mb=8192,
    ctx_size=4096,
    batch_size=512,
    n_threads=8,
    n_gpu_layers=-1
)
```

**Attributes:**

- `cache_dir` (str): Model cache directory
- `binary_dir` (str): Binary files directory
- `default_max_memory_mb` (int): Default memory limit
- `ctx_size` (int): Context window size
- `batch_size` (int): Batch processing size
- `n_threads` (int): CPU threads to use
- `n_gpu_layers` (int): GPU layers (-1 for auto)

**Methods:**

```python
# Load configuration from file
config = Config.from_file("config.yaml")

# Save configuration
config.save("config.yaml")

# Get default configuration
config = Config.default()
```

## Exceptions

### OprelError

Base exception for all Oprel errors.

```python
from oprel.core.exceptions import OprelError
```

### ModelNotFoundError

Raised when model file or alias not found.

```python
from oprel.core.exceptions import ModelNotFoundError

try:
    model = Model("nonexistent")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
```

### MemoryError

Raised when insufficient memory available.

```python
from oprel.core.exceptions import MemoryError

try:
    model.load()
except MemoryError as e:
    print(f"Not enough memory: {e}")
```

### BackendError

Raised when backend process fails.

```python
from oprel.core.exceptions import BackendError

try:
    response = model.generate("prompt")
except BackendError as e:
    print(f"Backend error: {e}")
```

### DownloadError

Raised during model download failures.

```python
from oprel.core.exceptions import DownloadError
```

### ValidationError

Raised for invalid parameters or configuration.

```python
from oprel.core.exceptions import ValidationError
```

## HTTP Client

Low-level HTTP client for server communication.

```python
from oprel.client import HTTPClient

client = HTTPClient(base_url="http://localhost:11434")
response = client.post("/generate", json={"model": "qwencoder", "prompt": "..."})
```

### Methods

```python
# Send POST request
response = client.post(endpoint, json=data)

# Send GET request
response = client.get(endpoint)

# Check server health
is_healthy = client.health_check()
```

## CLI Interface

Command-line interface reference.

```bash
# Server management
oprel serve [--port PORT] [--host HOST]
oprel models
oprel stop

# Generation
oprel run MODEL [PROMPT]
oprel chat MODEL [--system SYSTEM_PROMPT]

# Model management
oprel list-models
oprel search QUERY
oprel cache list
oprel cache clear
oprel cache delete MODEL

# Help
oprel --help
oprel run --help
```

## Environment Variables

```bash
OPREL_CACHE_DIR       # Model cache directory
OPREL_BINARY_DIR      # Binary files directory
OPREL_SERVER_PORT     # Default server port (11434)
OPREL_SERVER_HOST     # Default server host (localhost)
OPREL_LOG_LEVEL       # Logging level (INFO, DEBUG, ERROR)
```

## Usage Examples

### Complete Example

```python
from oprel import Model, Config
from oprel.core.exceptions import OprelError

# Custom configuration
config = Config(
    cache_dir="/data/models",
    default_max_memory_mb=8192
)

# Initialize model
model = Model(
    "qwencoder",
    quantization="Q4_K_M",
    config=config
)

# Generate with error handling
try:
    response = model.generate(
        "Explain Python decorators",
        max_tokens=1024,
        temperature=0.7
    )
    print(response)
except OprelError as e:
    print(f"Error: {e}")
```

### Ollama API Example

```python
from oprel import Client

client = Client()

# Chat completion
response = client.chat(
    model='qwencoder',
    messages=[
        {'role': 'system', 'content': 'You are a Python expert.'},
        {'role': 'user', 'content': 'Explain decorators'}
    ]
)

print(response.message.content)
```

## Notes

- Server mode (`use_server=True`) provides persistent model caching for fast subsequent requests
- Direct mode (`use_server=False`) loads models in the current process
- All model identifiers can be either full HuggingFace paths or predefined aliases
- Quantization is auto-selected based on available memory if not specified
- GPU acceleration is automatically enabled when available
