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
import json
import time as time_module
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from oprel.core.config import Config
from oprel.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

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
    name: str
    size_gb: float = 0.0
    quantization: Optional[str] = None
    backend: str = "llama.cpp"
    loaded: bool = False
    status: str = "cached" # cached, loaded

class MetricsResponse(BaseModel):
    """System and generation metrics"""
    cpu_usage: float
    ram_total_gb: float
    ram_used_gb: float
    gpu_name: Optional[str] = None
    gpu_usage: Optional[float] = None
    vram_total_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None
    generation_speed: float = 0.0


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
_model_last_used: Dict[str, float] = {}  # model_id -> timestamp of last use
_conversations: Dict[str, List[Dict[str, str]]] = {} # conversation_id -> list of messages
_conversation_meta: Dict[str, Dict] = {} # conversation_id -> metadata
_last_gen_speed: float = 0.0 # tokens per second
MAX_CONVERSATIONS = 100
MAX_HISTORY_MSGS = 50
IDLE_TIMEOUT_SECONDS = 15 * 60  # 15 minutes
_cleanup_task: Optional[asyncio.Task] = None

# PID file for tracking backend processes
_PID_FILE = CONFIG.cache_dir / "daemon.pid"
_BACKEND_PIDS_FILE = CONFIG.cache_dir / "backend_pids.txt"


def _write_daemon_pid():
    """Write daemon PID to file for tracking."""
    try:
        _PID_FILE.write_text(str(os.getpid()))
    except Exception as e:
        logger.debug(f"Could not write PID file: {e}")


def _remove_daemon_pid():
    """Remove daemon PID file."""
    try:
        if _PID_FILE.exists():
            _PID_FILE.unlink()
    except Exception:
        pass


def _track_backend_pid(pid: int):
    """Add a backend PID to the tracking file."""
    try:
        existing = set()
        if _BACKEND_PIDS_FILE.exists():
            existing = set(int(p) for p in _BACKEND_PIDS_FILE.read_text().strip().split('\n') if p.strip())
        existing.add(pid)
        _BACKEND_PIDS_FILE.write_text('\n'.join(str(p) for p in existing))
    except Exception as e:
        logger.debug(f"Could not track backend PID {pid}: {e}")


def _untrack_backend_pid(pid: int):
    """Remove a backend PID from the tracking file."""
    try:
        if not _BACKEND_PIDS_FILE.exists():
            return
        existing = set(int(p) for p in _BACKEND_PIDS_FILE.read_text().strip().split('\n') if p.strip())
        existing.discard(pid)
        if existing:
            _BACKEND_PIDS_FILE.write_text('\n'.join(str(p) for p in existing))
        else:
            _BACKEND_PIDS_FILE.unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Could not untrack backend PID {pid}: {e}")


def _kill_orphaned_backends():
    """Kill any orphaned oprel-backend processes from a previous daemon instance."""
    import psutil
    
    killed = 0
    
    # Method 1: Kill backends tracked in PID file
    try:
        if _BACKEND_PIDS_FILE.exists():
            pids = [int(p) for p in _BACKEND_PIDS_FILE.read_text().strip().split('\n') if p.strip()]
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    proc_name = proc.name().lower()
                    if 'oprel-backend' in proc_name or 'llama' in proc_name:
                        logger.info(f"Killing orphaned backend (PID: {pid})")
                        proc.kill()
                        proc.wait(timeout=3)
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
            _BACKEND_PIDS_FILE.unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Error cleaning tracked PIDs: {e}")
    
    # Method 2: Scan for any oprel-backend processes
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info.get('name', '').lower()
                if 'oprel-backend' in proc_name:
                    logger.info(f"Killing orphaned backend process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
                    proc.wait(timeout=3)
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
    except Exception as e:
        logger.debug(f"Error scanning for orphaned backends: {e}")
    
    if killed > 0:
        logger.info(f"Cleaned up {killed} orphaned backend process(es)")


# --- Helper Functions ---

def _cleanup_models():
    """Unload all models on shutdown"""
    global _models, _model_last_used
    for model_id, model in list(_models.items()):
        try:
            _force_unload_model(model_id, model)
            print(f"Unloaded model: {model_id}")
        except Exception as e:
            print(f"Error unloading {model_id}: {e}")
    _models.clear()
    _model_configs.clear()
    _model_last_used.clear()
    
    # Clean PID files
    _remove_daemon_pid()
    try:
        _BACKEND_PIDS_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _force_unload_model(model_id: str, model=None):
    """
    Force unload a model, ensuring the backend process is fully terminated.
    This prevents orphaned oprel-backend.exe processes.
    """
    if model is None:
        model = _models.get(model_id)
    if model is None:
        return
    
    backend_pid = None
    
    try:
        # Get backend PID before stopping
        if hasattr(model, '_process') and model._process:
            if model._process.process:
                backend_pid = model._process.process.pid
            
            logger.info(f"Stopping backend process for {model_id} (PID: {backend_pid})")
            model._process.stop(force=False)
            
            # Double-check: force kill if still alive
            if backend_pid:
                try:
                    import psutil
                    proc = psutil.Process(backend_pid)
                    if proc.is_running():
                        logger.warning(f"Backend PID {backend_pid} still alive after stop, force killing")
                        proc.kill()
                        proc.wait(timeout=3)
                except Exception:
                    pass
                _untrack_backend_pid(backend_pid)
            
            model._process = None
        
        # Stop monitor
        if hasattr(model, '_monitor') and model._monitor:
            model._monitor.stop()
            model._monitor = None
        
        # Cleanup PyTorch backend
        if hasattr(model, '_pytorch_backend') and model._pytorch_backend:
            model._pytorch_backend.unload()
            model._pytorch_backend = None
        
        model._loaded = False
        
        # Force GC
        import gc
        gc.collect()
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, Exception):
            pass
        
        logger.info(f"Model {model_id} fully unloaded")
        
    except Exception as e:
        logger.error(f"Error force-unloading model {model_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _unload_idle_model(model_id: str):
    """
    Unload idle model and force-clear GPU memory.
    This is more aggressive than regular unload() - it terminates the backend process.
    """
    global _models, _model_configs, _model_last_used
    
    if model_id not in _models:
        return
    
    try:
        model = _models[model_id]
        logger.info(f"Unloading idle model: {model_id} (forcing backend termination)")
        
        _force_unload_model(model_id, model)
        
        # Remove from caches
        _models.pop(model_id, None)
        _model_configs.pop(model_id, None)
        _model_last_used.pop(model_id, None)
        
        logger.info(f"✓ Idle model {model_id} unloaded, GPU memory freed")
        
    except Exception as e:
        logger.error(f"Error unloading idle model {model_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def _monitor_idle_models():
    """Background task to monitor and unload idle models"""
    global _models, _model_last_used
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = time_module.time()
            models_to_unload = []
            
            # Find idle models
            for model_id in list(_models.keys()):
                last_used = _model_last_used.get(model_id, 0)
                idle_time = current_time - last_used
                
                if idle_time > IDLE_TIMEOUT_SECONDS:
                    models_to_unload.append(model_id)
                    logger.info(
                        f"Model {model_id} has been idle for "
                        f"{idle_time / 60:.1f} minutes (>15 min threshold)"
                    )
            
            # Unload idle models
            for model_id in models_to_unload:
                _unload_idle_model(model_id)
                
        except asyncio.CancelledError:
            logger.info("Idle model monitoring task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in idle model monitor: {e}")


def _mark_model_used(model_id: str):
    """Update the last used timestamp for a model"""
    global _model_last_used
    _model_last_used[model_id] = time_module.time()


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
    from ..utils.chat_templates import format_chat_prompt
    
    # Build full conversation history including new message
    conversation_history = []
    
    # Add existing history
    conversation_history.extend(history)
    
    # Use the comprehensive chat template system
    return format_chat_prompt(
        model_id=model_id,
        user_message=new_user_msg,
        system_prompt=system_prompt,
        conversation_history=conversation_history
    )


# --- Startup/Shutdown ---

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nReceived shutdown signal, cleaning up...")
    _cleanup_models()
    sys.exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global _cleanup_task
    
    print(f"{Colors.GREEN}Oprel daemon starting...{Colors.RESET}")
    print(f"Cache Dir: {CONFIG.cache_dir}")
    print(f"Idle model timeout: {IDLE_TIMEOUT_SECONDS / 60:.0f} minutes")
    
    # CRITICAL: Kill any orphaned backend processes from previous daemon instance
    # This prevents accumulation of oprel-backend.exe processes
    try:
        _kill_orphaned_backends()
    except Exception as e:
        logger.warning(f"Error cleaning orphaned backends on startup: {e}")
    
    # Write our PID for tracking
    _write_daemon_pid()
    
    # Start background task to monitor idle models
    _cleanup_task = asyncio.create_task(_monitor_idle_models())
    logger.info("Started idle model monitoring task")
    
    yield
    
    # Shutdown
    print("\nReceived shutdown signal, cleaning up...")
    _cleanup_models()
    
    if _cleanup_task:
        _cleanup_task.cancel()

app = FastAPI(lifespan=lifespan, title="Oprel Daemon")
app.add_middleware(GinStyleLoggingMiddleware)

# Serve Web UI
WEBUI_DIR = Path(__file__).parent.parent / "webui"
if WEBUI_DIR.exists():
    app.mount("/gui", StaticFiles(directory=str(WEBUI_DIR), html=True), name="gui")

@app.get("/")
async def root():
    """Redirect to GUI if available, otherwise return health info"""
    if WEBUI_DIR.exists():
        return Response(status_code=307, headers={"Location": "/gui/"})
    return {"status": "ok", "version": "0.3.3"}

@app.get("/system/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system hardware and generation metrics"""
    import psutil
    from oprel.telemetry.hardware import get_vram_usage, detect_gpu
    
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
    vram = get_vram_usage()
    gpu = detect_gpu()
    
    return MetricsResponse(
        cpu_usage=cpu,
        ram_total_gb=round(ram.total / (1024**3), 2),
        ram_used_gb=round(ram.used / (1024**3), 2),
        gpu_name=gpu["gpu_name"] if gpu else None,
        gpu_usage=vram["utilization_percent"] if vram else None,
        vram_total_mb=vram["total_mb"] if vram else None,
        vram_used_mb=vram["used_mb"] if vram else None,
        generation_speed=_last_gen_speed
    )


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
    """Load a model into the cache. Automatically unloads any previously loaded model first."""
    global _models, _model_configs, _model_last_used
    
    if request.model_id in _models:
        # Check if backend process is still alive
        model = _models[request.model_id]
        if hasattr(model, '_process') and model._process is not None:
            if not model._process.is_running():
                logger.warning(f"Backend process for {request.model_id} died, reloading...")
                # Clean up the dead model entry
                _force_unload_model(request.model_id, model)
                _models.pop(request.model_id, None)
                _model_configs.pop(request.model_id, None)
                _model_last_used.pop(request.model_id, None)
            else:
                return LoadResponse(
                    success=True,
                    model_id=request.model_id,
                    message="Model already loaded"
                )
        else:
            return LoadResponse(
                success=True,
                model_id=request.model_id,
                message="Model already loaded"
            )
    
    # CRITICAL: Unload ALL previously loaded models before loading a new one.
    # This prevents accumulation of orphaned oprel-backend.exe processes.
    # Only one model should be active at a time to avoid GPU memory exhaustion.
    models_to_unload = list(_models.keys())
    for old_model_id in models_to_unload:
        logger.info(f"Unloading previous model '{old_model_id}' before loading '{request.model_id}'")
        try:
            old_model = _models[old_model_id]
            _force_unload_model(old_model_id, old_model)
        except Exception as e:
            logger.warning(f"Error unloading previous model {old_model_id}: {e}")
        _models.pop(old_model_id, None)
        _model_configs.pop(old_model_id, None)
        _model_last_used.pop(old_model_id, None)
    
    try:
        from oprel.core.model import Model
        
        logger.info(f"Loading model: {request.model_id} (quant={request.quantization}, backend={request.backend})")
        
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
        
        # Track the backend PID for orphan cleanup
        if hasattr(model, '_process') and model._process and model._process.process:
            _track_backend_pid(model._process.process.pid)
        
        # Cache it
        _models[request.model_id] = model
        _model_configs[request.model_id] = {
            "quantization": request.quantization,
            "max_memory_mb": request.max_memory_mb,
            "backend": request.backend,
        }
        
        # Mark as just loaded (used)
        _mark_model_used(request.model_id)
        
        logger.info(f"Model loaded successfully: {request.model_id}")
        
        return LoadResponse(
            success=True,
            model_id=request.model_id,
            message="Model loaded successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to load model {request.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text (Conversational)"""
    from oprel.downloader.aliases import resolve_model_id
    
    # Resolve alias
    resolved_model_id = resolve_model_id(request.model_id)
    
    # Auto-load logic
    if resolved_model_id not in _models:
        load_req = LoadRequest(model_id=resolved_model_id)
        await load_model(load_req)
    
    model = _models[resolved_model_id]
    
    # Check if backend process is still alive
    if hasattr(model, '_process') and model._process is not None:
        if not model._process.is_running():
            logger.warning(f"Backend process for {resolved_model_id} died, reloading...")
            # Remove from cache and reload
            del _models[resolved_model_id]
            del _model_configs[resolved_model_id]
            load_req = LoadRequest(model_id=resolved_model_id)
            await load_model(load_req)
            model = _models[resolved_model_id]
    
    model._loaded = True # Fix reload bug
    
    if not hasattr(model, '_client') or model._client is None:
        raise HTTPException(status_code=500, detail="Model client not available")

    # Mark model as used (for idle timeout tracking)
    _mark_model_used(resolved_model_id)

    # --- Conversation Management ---
    conv_id = request.conversation_id
    if not conv_id:
        conv_id = str(uuid.uuid4())
        
    if request.reset_conversation or conv_id not in _conversations:
        _conversations[conv_id] = []
        _conversation_meta[conv_id] = {
            "created_at": str(datetime.now()),
            "model_id": resolved_model_id,
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
        resolved_model_id, 
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
                    start_gen_time = time_module.perf_counter()
                    token_count = 0
                    
                    for token in model._client.generate(
                        prompt=full_prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True,
                    ):
                        full_resp += token
                        token_count += 1
                        yield f"data: {token}\n\n"
                    
                    # Calculate speed
                    end_gen_time = time_module.perf_counter()
                    duration = end_gen_time - start_gen_time
                    if duration > 0:
                        global _last_gen_speed
                        _last_gen_speed = token_count / duration
                    
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
            start_gen_time = time_module.perf_counter()
            text = model._client.generate(
                prompt=full_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )
            
            # Calculate speed (approximate tokens by split)
            end_gen_time = time_module.perf_counter()
            duration = end_gen_time - start_gen_time
            if duration > 0:
                global _last_gen_speed
                # Rough token estimate
                token_est = len(text.split()) * 1.3 # 1 word ~ 1.3 tokens
                _last_gen_speed = token_est / duration
            
            # Update history
            if len(history) >= MAX_HISTORY_MSGS:
                history.pop(0)
                if len(history) > 0: history.pop(0)
                
            history.append({"role": "user", "content": request.prompt})
            history.append({"role": "assistant", "content": text})
            
            return GenerateResponse(
                text=text,
                model_id=resolved_model_id,
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
    global _models, _model_configs, _model_last_used
    if model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    
    try:
        model = _models[model_id]
        _force_unload_model(model_id, model)
        _models.pop(model_id, None)
        _model_configs.pop(model_id, None)
        _model_last_used.pop(model_id, None)
        return UnloadResponse(success=True, model_id=model_id, message="Unloaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# OpenAI & Ollama API Compatibility (Week 14)
# ============================================================================

class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    stream: bool = False


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    stream: bool = False


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    # Extract prompt from messages
    prompt = request.messages[-1].content if request.messages else ""
    
    # Build conversation history
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages[:-1]]
    
    # Create GenerateRequest
    gen_request = GenerateRequest(
        model_id=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature,
        stream=request.stream
    )
    
    # Set up conversation
    conv_id = str(uuid.uuid4())
    _conversations[conv_id] = conversation_history
    _conversation_meta[conv_id] = {
        "created_at": str(datetime.now()),
        "model_id": request.model,
        "last_updated": str(datetime.now())
    }
    gen_request.conversation_id = conv_id
    
    response = await generate_text(gen_request)
    
    if request.stream:
        # Convert daemon SSE format to OpenAI format
        async def openai_stream_wrapper():
            request_id = f"chatcmpl-{int(time_module.time() * 1000)}"
            buffer = ""
            async for chunk in response.body_iterator:
                chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                buffer += chunk_str
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if line.startswith("data: "):
                        token = line[6:]
                        if token and token != "[DONE]" and not token.startswith("[ERROR]"):
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time_module.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time_module.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            openai_stream_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conv_id,
            }
        )
    else:
        return {
            "id": f"chatcmpl-{int(time_module.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time_module.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response.text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.text.split()),
                "total_tokens": len(prompt.split()) + len(response.text.split())
            }
        }


@app.post("/v1/completions")
async def v1_completions(request: OpenAICompletionRequest):
    """OpenAI-compatible text completions endpoint"""
    gen_request = GenerateRequest(
        model_id=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature,
        stream=request.stream
    )
    
    response = await generate_text(gen_request)
    
    if request.stream:
        # Convert daemon SSE format to OpenAI format
        async def openai_stream_wrapper():
            request_id = f"cmpl-{int(time_module.time() * 1000)}"
            buffer = ""
            async for chunk in response.body_iterator:
                chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                buffer += chunk_str
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if line.startswith("data: "):
                        token = line[6:]
                        if token and token != "[DONE]" and not token.startswith("[ERROR]"):
                            chunk = {
                                "id": request_id,
                                "object": "text_completion",
                                "created": int(time_module.time()),
                                "model": request.model,
                                "choices": [{
                                    "text": token,
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time_module.time()),
                "model": request.model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            openai_stream_wrapper(),
            media_type="text/event-stream"
        )
    else:
        return {
            "id": f"cmpl-{int(time_module.time() * 1000)}",
            "object": "text_completion",
            "created": int(time_module.time()),
            "model": request.model,
            "choices": [{
                "text": response.text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response.text.split()),
                "total_tokens": len(request.prompt.split()) + len(response.text.split())
            }
        }


@app.get("/v1/models")
async def v1_list_models():
    """OpenAI-compatible model list endpoint"""
    from oprel.downloader.cache import list_cached_models
    from oprel.downloader.aliases import MODEL_ALIASES
    
    models = []
    
    try:
        cached = list_cached_models()
        for model_info in cached:
            model_name = model_info.get('name', '')
            if model_name:
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(model_info.get('modified', datetime.now()).timestamp()),
                    "owned_by": "oprel"
                })
    except Exception as e:
        logger.warning(f"Could not list cached models: {e}")
    
    try:
        for alias in MODEL_ALIASES.keys():
            if not any(m["id"] == alias for m in models):
                models.append({
                    "id": alias,
                    "object": "model",
                    "created": int(time_module.time()),
                    "owned_by": "oprel"
                })
    except Exception as e:
        logger.warning(f"Could not list aliases: {e}")
    
    return {"object": "list", "data": models}


@app.get("/v1/health")
async def v1_health():
    """OpenAI-style health check"""
    return await health_check()


@app.post("/api/chat")
async def api_chat(request: dict):
    """Ollama-compatible chat endpoint"""
    openai_req = OpenAIChatRequest(
        model=request.get("model", ""),
        messages=[OpenAIChatMessage(**msg) for msg in request.get("messages", [])],
        stream=request.get("stream", False),
        temperature=request.get("options", {}).get("temperature", 0.7),
        max_tokens=request.get("options", {}).get("num_predict", 512)
    )
    return await v1_chat_completions(openai_req)


@app.post("/api/generate")
async def api_generate(request: dict):
    """Ollama-compatible generate endpoint"""
    openai_req = OpenAICompletionRequest(
        model=request.get("model", ""),
        prompt=request.get("prompt", ""),
        stream=request.get("stream", False),
        temperature=request.get("options", {}).get("temperature", 0.7),
        max_tokens=request.get("options", {}).get("num_predict", 512)
    )
    return await v1_completions(openai_req)


@app.get("/api/tags")
async def api_tags():
    """Ollama-compatible model list endpoint"""
    models_response = await v1_list_models()
    return {
        "models": [
            {
                "name": model["id"],
                "modified_at": time_module.strftime("%Y-%m-%dT%H:%M:%SZ", time_module.gmtime(model["created"])),
                "size": 0,
                "digest": ""
            }
            for model in models_response["data"]
        ]
    }



class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request"""
    prompt: str
    model: Optional[str] = "dall-e-3" # defaults to this in openai lib, we'll map to default local model
    n: int = 1
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url" # url or b64_json
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    user: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[Dict[str, str]]


# --- Global ComfyUI State ---
_comfy_client = None
_comfy_generator = None


async def _ensure_comfyui():
    """Ensure ComfyUI backend is running and connected"""
    global _comfy_client, _comfy_generator
    
    # Lazy import to avoid dependency issues if not installed
    try:
        from oprel.runtime.backends.comfyui import ComfyUIClient, ComfyUIImageGenerator
        from oprel.downloader.comfyui_installer import get_comfyui_dir
    except ImportError:
        raise HTTPException(status_code=500, detail="Image generation dependencies not installed. Run 'oprel setup image' first.")

    if _comfy_client is None:
        _comfy_client = ComfyUIClient()
        
    if not _comfy_client.is_available():
        logger.info("Starting ComfyUI server from daemon...")
        
        # Check if installed
        comfy_dir = get_comfyui_dir()
        if not comfy_dir.exists():
            raise HTTPException(status_code=500, detail="ComfyUI not found. Run 'oprel setup image' first.")
            
        # Start backend
        from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
        backend = ComfyUIBackend(None, None)
        if not backend.start():
             raise HTTPException(status_code=500, detail="Failed to start ComfyUI backend")
             
        # Wait for valid connection
        max_retries = 10
        for _ in range(max_retries):
            await asyncio.sleep(1)
            if _comfy_client.is_available():
                break
        else:
            raise HTTPException(status_code=504, detail="ComfyUI backend started but not responding")

    if _comfy_generator is None:
        _comfy_generator = ComfyUIImageGenerator(_comfy_client)


@app.post("/v1/images/generations")
async def v1_images_generations(request: ImageGenerationRequest):
    """OpenAI-compatible image generation endpoint"""
    await _ensure_comfyui()
    
    # Map valid models
    # OpenAI sends "dall-e-2" or "dall-e-3". We map these to our local default if not specified properly.
    model_id = request.model
    if not model_id or model_id.startswith("dall-e"):
        # Default to a fast/good model if user didn't specify a local one
        # Ideally we'd look up what's installed.
        from oprel.downloader.comfyui_installer import list_installed_checkpoints
        installed = list_installed_checkpoints()
        if installed:
            # Pick first available, prefer flux or sdxl
            preferred = [m['name'] for m in installed if 'flux' in m['name'].lower()]
            if preferred:
                model_id = preferred[0]
            else:
                model_id = installed[0]['name']
        else:
            raise HTTPException(status_code=400, detail="No image models installed. Run 'oprel pull flux-1-schnell' or similar.")
    
    # Parse dimensions
    width, height = 1024, 1024
    if request.size:
        try:
            w, h = request.size.split("x")
            width, height = int(w), int(h)
        except:
            pass
            
    try:
        logger.info(f"Generating image with {model_id} for prompt: {request.prompt}")
        
        # Generate raw bytes (PNG)
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None,
            lambda: _comfy_generator.generate_txt2img(
                prompt=request.prompt,
                checkpoint=model_id,
                width=width,
                height=height,
                steps=4 if "turbo" in model_id or "schnell" in model_id else 20
            )
        )
        
        import base64
        
        response_data = []
        if request.response_format == "b64_json":
             b64_str = base64.b64encode(image_bytes).decode('utf-8')
             response_data.append({"b64_json": b64_str, "revised_prompt": request.prompt})
        else:
             # URL format. Since we are local, we can't easily give a public URL.
             # We could save to static dir and serve it, but for now fallback to b64 or a data URI concept?
             # Standard OpenAI clients expect a URL.
             # Let's save to a temporary public static folder if we want to support this fully, 
             # but for now let's just return b64_json anyway or a data uri if they really want a "url" string.
             # Actually, providing a base64 string in 'url' field sometimes works for some clients, but safer to use b64_json.
             # If they explicitly asked for URL, we'll try to serve it.
             
             # Create static dir if needed
             static_dir = CONFIG.cache_dir / "generated_images"
             static_dir.mkdir(exist_ok=True)
             
             filename = f"img_{int(time_module.time())}_{uuid.uuid4().hex[:8]}.png"
             file_path = static_dir / filename
             file_path.write_bytes(image_bytes)
             
             # We need to mount this dir to serve it.
             # For now, let's just return the local file path as the URL (clients running locally might resolve it)
             # or better, return a fake URL that we haven't implemented serving for yet :)
             # Let's stick to b64_json preference.
             if request.response_format == "url":
                 # We'll just return b64_json in the b64_json field and hope client checks it,
                 # or we validly fail.
                 # Actually, many tools verify the URL.
                 # Let's implement static serving later. For now, force usage of b64_json if possible.
                 pass
                 
             b64_str = base64.b64encode(image_bytes).decode('utf-8')
             response_data.append({"b64_json": b64_str, "revised_prompt": request.prompt})

        return ImageGenerationResponse(
            created=int(time_module.time()),
            data=response_data
        )

    except Exception as e:
        logger.error(f"Image generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
async def shutdown_server():
    """Gracefully shutdown the server"""
    import asyncio
    
    async def shutdown():
        # Give response time to be sent
        await asyncio.sleep(0.5)
        # Cleanup models
        _cleanup_models()
        # Exit the process
        import os
        os._exit(0)
    
    # Start shutdown in background
    asyncio.create_task(shutdown())
    return {"status": "shutting down"}


def run_server(host: str = "127.0.0.1", port: int = 11435):
    """Run the daemon server"""
    import uvicorn
    print(f"{Colors.GREEN}{Colors.BOLD}Oprel Daemon v0.3.3{Colors.RESET}")
    print(f"  Listening on: {Colors.CYAN}http://{host}:{port}{Colors.RESET}")
    print(f"  Press {Colors.YELLOW}Ctrl+C{Colors.RESET} to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    run_server()
