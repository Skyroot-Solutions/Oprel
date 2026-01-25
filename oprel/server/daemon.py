"""
Oprel Daemon Server - Persistent model caching for fast inference.

This server keeps models loaded in memory, eliminating the 2-minute
startup time on every script execution. It also manages conversation
history and model discovery.

Usage:
    oprel serve                  # Start server on default port 11434
    oprel serve --port 8080      # Start on custom port

API Endpoints:
    POST /load      - Load a model into cache
    POST /generate  - Generate text (supports chat history)
    GET /models     - List all loaded and available cached models
    DELETE /unload/{model_id} - Unload a specific model
    GET /health     - Health check
    
    # Conversation APIs
    GET /conversations - List active conversations
    GET /conversations/{id} - Get conversation history
    DELETE /conversations/{id} - Delete conversation
    POST /conversations/{id}/reset - Reset conversation
"""

import os
import sys
import signal
import atexit
import uuid
import time as time_module
from datetime import datetime
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from oprel.core.config import Config

# Import Model with use_server=False to avoid circular dependency
# The server itself uses direct mode internally

# Initialize Config
CONFIG = Config()
CONFIG.ensure_dirs()

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"      # 2xx success
    CYAN = "\033[96m"       # 3xx redirect
    YELLOW = "\033[93m"     # 4xx client error
    RED = "\033[91m"        # 5xx server error
    BLUE = "\033[94m"       # Method
    MAGENTA = "\033[95m"    # Path
    WHITE = "\033[97m"      # IP
    GRAY = "\033[90m"       # Timestamp prefix


def get_status_color(status_code: int) -> str:
    """Get color based on HTTP status code"""
    if 200 <= status_code < 300: return Colors.GREEN
    elif 300 <= status_code < 400: return Colors.CYAN
    elif 400 <= status_code < 500: return Colors.YELLOW
    else: return Colors.RED


def format_duration(duration_ms: float) -> str:
    """Format duration in human-readable format"""
    if duration_ms < 1: return f"{duration_ms * 1000:.2f}µs"
    elif duration_ms < 1000: return f"{duration_ms:.2f}ms"
    else: return f"{duration_ms / 1000:.2f}s"


class GinStyleLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs requests in Gin-like format with colors"""
    async def dispatch(self, request: Request, call_next):
        start_time = time_module.perf_counter()
        response = await call_next(request)
        duration = (time_module.perf_counter() - start_time) * 1000  # ms
        
        status_code = response.status_code
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        timestamp = datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
        status_color = get_status_color(status_code)
        
        log_line = (
            f"{Colors.GRAY}[OPREL]{Colors.RESET} "
            f"{timestamp} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{status_color}{Colors.BOLD}{status_code}{Colors.RESET} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{format_duration(duration):>12} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{Colors.WHITE}{client_ip:>15}{Colors.RESET} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{Colors.BLUE}{method:<8}{Colors.RESET} "
            f"{Colors.MAGENTA}\"{path}\"{Colors.RESET}"
        )
        print(log_line)
        return response


# --- Data Models ---

class LoadRequest(BaseModel):
    """Request to load a model"""
    model_id: str
    quantization: Optional[str] = None
    max_memory_mb: Optional[int] = None
    backend: str = "llama.cpp"


class GenerateRequest(BaseModel):
    """Request to generate text"""
    model_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    
    # New fields for conversational API
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    reset_conversation: bool = False


class GenerateResponse(BaseModel):
    """Response from text generation"""
    text: str
    model_id: str
    conversation_id: str
    message_count: int


class ModelInfo(BaseModel):
    """Information about a model"""
    model_id: str
    quantization: Optional[str]
    backend: str
    loaded: bool
    size_gb: Optional[float] = None
    name: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ConversationInfo(BaseModel):
    id: str
    created_at: str
    last_updated: str
    message_count: int
    model_id: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    active_conversations: int


class LoadResponse(BaseModel):
    success: bool
    model_id: str
    message: str


class UnloadResponse(BaseModel):
    success: bool
    model_id: str
    message: str


class UnloadRequest(BaseModel):
    """Request to unload a model"""
    model_id: str





# --- Global State ---

_models: Dict[str, Any] = {}  # model_id -> Model instance
_model_configs: Dict[str, dict] = {}  # model_id -> config info
_conversations: Dict[str, List[Dict[str, str]]] = {} # conversation_id -> list of messages
_conversation_meta: Dict[str, Dict] = {} # conversation_id -> metadata
MAX_CONVERSATIONS = 100
MAX_HISTORY_MSGS = 50


# --- Helper Functions ---

def _cleanup_models():
    """Unload all models on shutdown"""
    global _models
    for model_id, model in list(_models.items()):
        try:
            model.unload()
            print(f"Unloaded model: {model_id}")
        except Exception as e:
            print(f"Error unloading {model_id}: {e}")
    _models.clear()
    _model_configs.clear()


def _cleanup_conversations():
    """LRU cleanup for conversational memory"""
    global _conversations, _conversation_meta
    if len(_conversations) > MAX_CONVERSATIONS:
        # Sort by last updated (approximation, Python dicts preserve insertion order mostly)
        # Using _conversation_meta['last_updated'] would be better but expensive to sort
        # Simple FIFO removal
        to_remove = len(_conversations) - MAX_CONVERSATIONS
        keys = list(_conversations.keys())[:to_remove]
        for k in keys:
            del _conversations[k]
            if k in _conversation_meta:
                del _conversation_meta[k]


def _scan_cached_models() -> List[ModelInfo]:
    """Scan cache directory for available models"""
    available = []
    
    # First add loaded models
    for model_id, config in _model_configs.items():
        if model_id not in [m.model_id for m in available]:
             available.append(ModelInfo(
                model_id=model_id,
                quantization=config.get("quantization"),
                backend=config.get("backend", "llama.cpp"),
                loaded=True,
                name=model_id.split("/")[-1] if "/" in model_id else model_id
            ))

    # Scan cache dir
    cache_dir = CONFIG.cache_dir
    if not cache_dir.exists():
        return available
        
    try:
        # Files are stored as 'models--Author--Name'
        for model_dir in cache_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                try:
                    # Parse model_id: models--TheBloke--Llama-2 -> TheBloke/Llama-2
                    parts = model_dir.name.split("--")
                    if len(parts) >= 3:
                        model_id = f"{parts[1]}/{parts[2]}"
                        
                        # Check snapshots
                        snapshots_dir = model_dir / "snapshots"
                        if snapshots_dir.exists():
                            for snapshot in snapshots_dir.iterdir():
                                if snapshot.is_dir():
                                    # Find GGUF files
                                    gguf_files = list(snapshot.glob("*.gguf"))
                                    for file in gguf_files:
                                        # Deduplicate if already loaded
                                        if any(m.model_id == model_id for m in available):
                                            continue
                                            
                                        size_gb = file.stat().st_size / (1024**3)
                                        
                                        # Attempt to detect quantization from filename
                                        quant = "Unknown"
                                        name_upper = file.name.upper()
                                        for q in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]:
                                             if q in name_upper:
                                                 quant = q
                                                 break
                                        
                                        available.append(ModelInfo(
                                            model_id=model_id,
                                            quantization=quant,
                                            backend="llama.cpp",
                                            loaded=False,
                                            size_gb=round(size_gb, 2),
                                            name=model_id.split("/")[-1]
                                        ))
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error scanning cache: {e}")
        
    return available


def _build_chat_prompt(model_id: str, history: List[Dict[str, str]], system_prompt: Optional[str] = None, new_user_msg: str = "") -> str:
    """Build a prompt based on model type and chat history"""
    
    prompt = ""
    model_id_lower = model_id.lower()
    
    # --- Chat Templates ---
    
    # 1. Llama 2: [INST] <<SYS>>...<</SYS>> ... [/INST]
    if "llama-2" in model_id_lower:
        if system_prompt:
            prompt += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        else:
            prompt += "[INST] "
            
        for msg in history:
            if msg["role"] == "user":
                prompt += f"{msg['content']} [/INST] "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']} </s><s>[INST] "
                
        prompt += f"{new_user_msg} [/INST]"
        
    # 2. Llama 3: <|start_header_id|>...
    elif "llama-3" in model_id_lower:
        prompt += "<|begin_of_text|>"
        if system_prompt:
             prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        
        for msg in history:
            prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{new_user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # 3. Gemma: <start_of_turn>...
    elif "gemma" in model_id_lower:
        for msg in history:
             role = "model" if msg["role"] == "assistant" else "user"
             prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
             
        prompt += f"<start_of_turn>user\n{new_user_msg}<end_of_turn>\n<start_of_turn>model\n"

    # 4. Mistral / Default: [INST]
    else:
        # Default fallback (similar to Llama 2 / Mistral)
        if system_prompt:
            prompt += f"[INST] System: {system_prompt}\n\n"
        else:
            prompt += "[INST] "
            
        for msg in history:
            if msg["role"] == "user":
                prompt += f"{msg['content']} [/INST] "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']} [INST] "
                
        prompt += f"{new_user_msg} [/INST]"

    return prompt


# --- Startup/Shutdown ---

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nReceived shutdown signal, cleaning up...")
    _cleanup_models()
    sys.exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    print(f"{Colors.GREEN}Oprel daemon starting...{Colors.RESET}")
    print(f"Cache Dir: {CONFIG.cache_dir}")
    
    atexit.register(_cleanup_models)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    yield
    
    print(f"{Colors.YELLOW}Oprel daemon shutting down...{Colors.RESET}")
    _cleanup_models()


app = FastAPI(
    title="Oprel Daemon",
    description="Persistent model server for fast LLM inference",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(GinStyleLoggingMiddleware)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        models_loaded=len(_models),
        active_conversations=len(_conversations)
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded and cached models"""
    return _scan_cached_models()


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    """Load a model into the cache"""
    global _models, _model_configs
    
    if request.model_id in _models:
        return LoadResponse(
            success=True,
            model_id=request.model_id,
            message="Model already loaded"
        )
    
    try:
        from oprel.core.model import Model
        
        # Create model with use_server=False (direct mode inside server)
        model = Model(
            model_id=request.model_id,
            quantization=request.quantization,
            max_memory_mb=request.max_memory_mb,
            backend=request.backend,
            use_server=False,
        )
        
        # Load the model
        model.load()
        
        # Cache it
        _models[request.model_id] = model
        _model_configs[request.model_id] = {
            "quantization": request.quantization,
            "max_memory_mb": request.max_memory_mb,
            "backend": request.backend,
        }
        
        return LoadResponse(
            success=True,
            model_id=request.model_id,
            message="Model loaded successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text (Conversational)"""
    
    # Auto-load logic
    if request.model_id not in _models:
        load_req = LoadRequest(model_id=request.model_id)
        await load_model(load_req)
    
    model = _models[request.model_id]
    model._loaded = True # Fix reload bug
    
    if not hasattr(model, '_client') or model._client is None:
        raise HTTPException(status_code=500, detail="Model client not available")

    # --- Conversation Management ---
    conv_id = request.conversation_id
    if not conv_id:
        conv_id = str(uuid.uuid4())
        
    if request.reset_conversation or conv_id not in _conversations:
        _conversations[conv_id] = []
        _conversation_meta[conv_id] = {
            "created_at": str(datetime.now()),
            "model_id": request.model_id,
        }
        
    history = _conversations[conv_id]
    _cleanup_conversations() # Prune if needed
    
    # Update metadata
    _conversation_meta[conv_id]["last_updated"] = str(datetime.now())
    
    # Build prompt with history
    # If conversation_id was NOT passed explicitly (one-off), we might still want to use template
    # but the request.prompt is the "new user message"
    
    # If specific system prompt requested, use it, otherwise use stored or None
    sys_prompt = request.system_prompt
    
    full_prompt = _build_chat_prompt(
        request.model_id, 
        history, 
        sys_prompt, 
        request.prompt
    )
    
    # If raw prompt mode requested? (Future feature). For now, always use Chat template if model detection works.
    
    try:
        final_text = ""
        
        if request.stream:
            def generate_stream():
                full_resp = ""
                try:
                    for token in model._client.generate(
                        prompt=full_prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True,
                    ):
                        full_resp += token
                        yield f"data: {token}\n\n"
                    
                    # Update history after full generation
                    # We can't update per-token in the generator safely due to async/yield
                    # But Python generators run in the worker... 
                    # We'll update history when stream is DONE
                    if len(history) >= MAX_HISTORY_MSGS:
                        history.pop(0) # Remove oldest pair? Ideally remove 0 and 1.
                        if len(history) > 0: history.pop(0)

                    history.append({"role": "user", "content": request.prompt})
                    history.append({"role": "assistant", "content": full_resp})
                    
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: [ERROR] {str(e)}\n\n"
            
            # Note: The conversation update happens inside the generator which might be tricky if it fails mid-stream.
            # But for a basic implementation this works.
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Conversation-ID": conv_id,
                }
            )
        else:
            # Non-streaming
            text = model._client.generate(
                prompt=full_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )
            
            # Update history
            if len(history) >= MAX_HISTORY_MSGS:
                history.pop(0)
                if len(history) > 0: history.pop(0)
                
            history.append({"role": "user", "content": request.prompt})
            history.append({"role": "assistant", "content": text})
            
            return GenerateResponse(
                text=text,
                model_id=request.model_id,
                conversation_id=conv_id,
                message_count=len(history)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# --- Conversation Endpoints ---

@app.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations():
    """List active conversations"""
    results = []
    for cid, meta in _conversation_meta.items():
        results.append(ConversationInfo(
            id=cid,
            created_at=meta.get("created_at", ""),
            last_updated=meta.get("last_updated", ""),
            message_count=len(_conversations.get(cid, [])),
            model_id=meta.get("model_id", "unknown")
        ))
    return results


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _conversations[conversation_id]


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id in _conversations:
        del _conversations[conversation_id]
    if conversation_id in _conversation_meta:
        del _conversation_meta[conversation_id]
    return {"success": True}


@app.post("/conversations/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    """Reset a conversation history"""
    if conversation_id in _conversations:
        _conversations[conversation_id] = []
        return {"success": True}
    raise HTTPException(status_code=404, detail="Conversation not found")


@app.post("/unload", response_model=UnloadResponse)
async def unload_model_post(request: UnloadRequest):
    return await unload_model(request.model_id) # Reuse logic


@app.delete("/unload/{model_id}", response_model=UnloadResponse)
async def unload_model(model_id: str):
    global _models, _model_configs
    if model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    
    try:
        model = _models[model_id]
        model.unload()
        del _models[model_id]
        if model_id in _model_configs:
            del _model_configs[model_id]
        return UnloadResponse(success=True, model_id=model_id, message="Unloaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "127.0.0.1", port: int = 11434):
    """Run the daemon server"""
    import uvicorn
    print(f"{Colors.GREEN}{Colors.BOLD}Oprel Daemon v0.3.0{Colors.RESET}")
    print(f"  Listening on: {Colors.CYAN}http://{host}:{port}{Colors.RESET}")
    print(f"  Press {Colors.YELLOW}Ctrl+C{Colors.RESET} to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    run_server()
