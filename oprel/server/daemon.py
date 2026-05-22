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
import threading
import httpx
from datetime import datetime
from typing import Dict, Optional, Any, List, Union
from contextlib import asynccontextmanager
from pathlib import Path
import importlib.resources as pkg_resources
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from oprel.core.config import Config
from oprel.utils.logging import get_logger
from oprel.server.download_manager import download_manager

# Initialize Config
CONFIG = Config()
CONFIG.ensure_dirs()

# Initialize logger
logger = get_logger(__name__)

# Import db AFTER ensuring directories
from oprel.server import db

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


class PullRequest(BaseModel):
    """Request to pull/download a model"""
    model_id: str
    quantization: Optional[str] = None  # Specific quantization to download
    
class EmbedRequest(BaseModel):
    """Request to generate embeddings"""
    model: str
    input: Union[str, List[str]]
    
class GenerateRequest(BaseModel):
    """Request to generate text"""
    model_id: str
    prompt: Any
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = False
    images: Optional[List[str]] = None
    
    # New fields for conversational API
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    reset_conversation: bool = False
    thinking: bool = False
    rag: bool = False  # New: Enable Retrieval-Augmented Generation


class GenerateResponse(BaseModel):
    """Response from text generation"""
    text: str
    model_id: str
    conversation_id: str
    message_count: int

class RenameConversationRequest(BaseModel):
    title: str

class UserSettings(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 4096
    system_instruction: Optional[str] = None



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


class UserProfile(BaseModel):
    name: str
    role: str
    initials: Optional[str] = None


class UnloadRequest(BaseModel):
    """Request to unload a model"""
    model_id: str


class DocumentInfo(BaseModel):
    """Information about an indexed document"""
    id: str
    filename: str
    size_bytes: int
    indexed_at: str
    chunks: int





# --- Global State ---

_models: Dict[str, Any] = {}  # model_id -> Model instance
_model_configs: Dict[str, dict] = {}  # model_id -> config info
_model_last_used: Dict[str, float] = {}  # model_id -> timestamp of last use
_last_gen_speed: float = 0.0 # tokens per second
IDLE_TIMEOUT_SECONDS = 15 * 60  # 15 minutes
_cleanup_task: Optional[asyncio.Task] = None

# PID file for tracking backend processes
_PID_FILE = CONFIG.cache_dir / "daemon.pid"
_BACKEND_PIDS_FILE = CONFIG.cache_dir / "backend_pids.txt"

# In-memory history for ephemeral (non-WebUI) conversations
# WebUI conversations use 'chat_' prefix and go to SQLite
_ephemeral_history: Dict[str, List[Dict[str, str]]] = {}


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
    """No-op, DB handles this"""
    pass


def _scan_cached_models() -> List[ModelInfo]:
    """Scan cache directory for available models.
    Returns ONE entry per GGUF file so that every downloaded quantization
    appears as a separate row (e.g. Q8_0 and Q2_K of the same model).
    """
    available = []
    # Track (repo_id, filename) pairs to avoid truly duplicate entries
    seen_files: set = set()

    # All known quantization patterns, checked in order (longer first to avoid partial matches)
    QUANT_PATTERNS = [
        "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
        "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K",
        "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_K",
        "Q2_K_XL", "Q2_K_L", "Q2_K_S", "Q2_K",
        "Q6_K", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
        "IQ1_M", "IQ1_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
        "IQ3_M", "IQ3_S", "IQ3_XS", "IQ4_NL", "IQ4_XS",
        "F32", "F16", "BF16",
    ]

    # First add loaded models (from RAM)
    for model_id, config in _model_configs.items():
        from oprel.downloader.aliases import get_model_category, get_best_alias_for_repo
        cat = get_model_category(model_id)

        model_id_lower = model_id.lower()
        is_unwanted = any(kw in model_id_lower for kw in ['embed', 'embedding', 'nomic-embed', 'bge-m3', 'flux', 'stable-diffusion', 'sdxl', 'pixart'])
        if is_unwanted or cat in ["embeddings", "text-to-image", "text-to-video"]:
            continue

        quant = config.get("quantization") or "Unknown"

        # If the loaded model has no quant info, defer to file-scan which will
        # detect the quant from the filename and mark it as loaded.
        if quant == "Unknown":
            continue

        file_key = (model_id, quant)
        if file_key in seen_files:
            continue
        seen_files.add(file_key)

        best_alias = get_best_alias_for_repo(model_id)
        display_name = best_alias or (model_id.split("/")[-1] if "/" in model_id else model_id)

        available.append(ModelInfo(
            model_id=model_id,
            quantization=quant,
            backend=config.get("backend", "llama.cpp"),
            loaded=True,
            name=display_name
        ))

    # Scan cache dir — one entry per GGUF file
    cache_dir = CONFIG.cache_dir
    if not cache_dir.exists():
        return available

    try:
        from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
        from oprel.downloader.aliases import get_model_category, get_best_alias_for_repo

        for gguf_file in cache_dir.rglob("*.gguf"):
            if not gguf_file.is_file() or gguf_file.stat().st_size == 0:
                continue

            filename = gguf_file.name
            filename_lower = filename.lower()

            # Skip side-car files
            if 'mmproj' in filename_lower or filename_lower.startswith('vision-') or filename_lower.startswith('clip-'):
                continue

            # Resolve repo_id
            repo_id = get_repo_id_from_filename(cache_dir, filename) \
                      or infer_repo_id_from_cache(cache_dir, filename) \
                      or filename

            # Skip embedding / image models
            if repo_id != filename:
                cat = get_model_category(repo_id)
                repo_id_lower = repo_id.lower()
                is_unwanted = any(kw in repo_id_lower for kw in [
                    'embed', 'embedding', 'nomic-embed', 'bge-m3',
                    'flux', 'stable-diffusion', 'sdxl', 'pixart'
                ])
                if is_unwanted or cat in ["embeddings", "text-to-image", "text-to-video"]:
                    continue

            # Dedup on exact file, not on repo
            file_key = (repo_id, filename)
            if file_key in seen_files:
                continue
            seen_files.add(file_key)

            # Detect quantization from filename
            name_upper = filename.upper()
            quant = next((q for q in QUANT_PATTERNS if q in name_upper), "Unknown")

            # Check whether this specific GGUF file is the currently-loaded model.
            # Priority: match by stored filename > match by quant string.
            is_loaded = False
            for loaded_id, cfg in _model_configs.items():
                ids_match = loaded_id in (repo_id, filename.replace(".gguf", ""))
                if not ids_match:
                    continue
                stored_filename = cfg.get("filename")
                loaded_quant = (cfg.get("quantization") or "").upper()
                if stored_filename:
                    # Exact filename match — only ONE file will match
                    is_loaded = (filename == stored_filename)
                elif loaded_quant:
                    # Fall back to quant string comparison
                    is_loaded = (loaded_quant == quant)
                # If neither stored, leave as False (can't determine)
                if is_loaded:
                    break

            size_gb = gguf_file.stat().st_size / (1024**3)

            # Display name: alias (preferred) or repo name
            best_alias = get_best_alias_for_repo(repo_id)
            if best_alias:
                display_name = best_alias
            elif repo_id != filename and "/" in repo_id:
                display_name = repo_id.split("/")[-1]
            else:
                display_name = filename.replace(".gguf", "")

            # Skip if this quant is already represented by a loaded model (to avoid duplicate rows)
            alreadyLoaded = any(
                cfg.get("quantization") == quant and (
                    model_id == repo_id or cfg.get("filename") == filename
                )
                for model_id, cfg in _model_configs.items()
            )
            if alreadyLoaded:
                continue

            available.append(ModelInfo(
                model_id=repo_id,
                quantization=quant if quant != "Unknown" else None,
                backend="llama.cpp",
                loaded=is_loaded,
                size_gb=round(size_gb, 2),
                name=display_name
            ))

    except Exception as e:
        logger.error(f"Error scanning cache: {e}")

    return available


def _build_chat_prompt(model_id: str, history: List[Dict[str, str]], system_prompt: Optional[str] = None, new_user_msg: str = "", thinking: bool = False) -> str:
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
        conversation_history=conversation_history,
        thinking=thinking
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
    
    # Migrate existing models to add metadata
    try:
        from oprel.downloader.migrate_metadata import migrate_existing_models
        migrate_existing_models()
    except Exception as e:
        logger.warning(f"Error during metadata migration: {e}")
    
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

class StripApiPrefixMiddleware(BaseHTTPMiddleware):
    """Strip /api prefix from incoming requests so React's /api/* calls
    reach the existing un-prefixed FastAPI routes."""
    async def dispatch(self, request: Request, call_next):
        if request.scope["path"].startswith("/api/"):
            stripped = request.scope["path"][4:]  # remove '/api'
            request.scope["path"] = stripped
            request.scope["raw_path"] = stripped.encode()
        return await call_next(request)


app = FastAPI(lifespan=lifespan, title="Oprel Daemon")
app.add_middleware(GinStyleLoggingMiddleware)
app.add_middleware(StripApiPrefixMiddleware)
from fastapi.middleware.cors import CORSMiddleware

# after you create your FastAPI/Starlette `app` object:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000","http://127.0.0.1:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve Web UI — use importlib.resources for reliable packaging access
def get_webui_dir():
    # Priority 1: Use importlib.resources for reliable packaging access (Modern approach)
    try:
        from importlib.resources import files
        path = files("oprel") / "webui-react" / "out"
        if path.joinpath("index.html").exists():
            return str(path)
    except (ImportError, Exception):
        pass

    # Priority 2: webui-react/out (The new React build - relative to this file)
    react_ui_paths = [
        Path(sys.prefix) / "oprel" / "webui-react" / "out",
        Path(__file__).parent.parent / "webui-react" / "out"
    ]
    
    for path in react_ui_paths:
        if path.exists() and (path / "index.html").exists():
            return str(path)
            
    # Priority 3: webui (Legacy)
    legacy_ui_paths = [
        Path(sys.prefix) / "oprel" / "webui",
        Path(__file__).parent.parent / "webui"
    ]
    
    for path in legacy_ui_paths:
        if path.exists():
            return str(path)
            
    return None

class UIStaticFiles(StaticFiles):
    """Custom StaticFiles handler that prevents caching for HTML content."""
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        # Prevent caching for the entry point (empty path) or any HTML-related path
        # This fixes the 'ChunkLoadError' when developers rebuild the WebUI
        # and the browser tries to load defunct hashes from a cached index.html.
        last_part = path.split("/")[-1] if path else ""
        if not path or path.endswith(".html") or "." not in last_part:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

WEBUI_DIR = get_webui_dir()
if WEBUI_DIR and Path(WEBUI_DIR).exists():
    app.mount("/gui", UIStaticFiles(directory=str(WEBUI_DIR), html=True), name="gui")

@app.get("/")
async def root():
    """Redirect to GUI if available, otherwise return health info"""
    if WEBUI_DIR and Path(WEBUI_DIR).exists():
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


# --- Knowledge Infrastructure (RAG) Endpoints ---

class IngestRequest(BaseModel):
    text: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.post("/knowledge/ingest")
async def ingest_knowledge(request: IngestRequest):
    """Ingest text or file into knowledge store"""
    from oprel.knowledge.sync_engine import SyncEngine
    engine = SyncEngine()
    
    try:
        if request.file_path:
            engine.ingest_file(Path(request.file_path))
            return {"success": True, "message": f"Ingested file: {request.file_path}"}
        elif request.text:
            engine.ingest_text(request.text, request.metadata)
            return {"success": True, "message": "Ingested raw text"}
        else:
            raise HTTPException(status_code=400, detail="Either 'text' or 'file_path' must be provided")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/search")
async def search_knowledge(q: str, top_k: int = 5):
    """Search knowledge store (Vector + BM25)"""
    from oprel.knowledge.knowledge_store import KnowledgeStore
    
    # Use direct internal embedding to avoid HTTP recursion
    async def internal_embed(text, model=None):
        res = await get_embeddings(EmbedRequest(input=text, model=model or "nomic-embed-text"))
        return res.get("embedding")
        
    store = KnowledgeStore(embed_func=internal_embed)
    results = await store.search(q, top_k=top_k)
    return results

@app.post("/knowledge/reset")
async def reset_knowledge():
    """Clear all knowledge indices"""
    from oprel.knowledge.config import KNOWLEDGE_DIR
    import shutil
    try:
        if KNOWLEDGE_DIR.exists():
            shutil.rmtree(KNOWLEDGE_DIR)
            KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": "Knowledge store reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        models_loaded=len(_models),
        active_conversations=db.get_active_conversation_count()
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded and cached models"""
    return _scan_cached_models()


@app.get("/registry/models")
async def list_registry_models():
    """List all officially supported models from the registry"""
    from oprel.downloader.aliases import OFFICIAL_REPOS, CATEGORY_INFO
    return {
        "categories": CATEGORY_INFO,
        "models": OFFICIAL_REPOS
    }


@app.get("/models/info/{model_id:path}")
async def get_model_detailed_info(model_id: str):
    """
    Get detailed model information including parameters and available quantizations.
    
    Args:
        model_id: Model alias or repo ID (URL encoded)
        
    Returns:
        Model info with parameters, quantizations, and size calculations
    """
    from oprel.downloader.aliases import resolve_model_id
    from oprel.utils.model_info import get_model_info
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
    
    try:
        # Resolve model ID (handle both aliases and local filenames)
        repo_id = model_id
        
        # First, check if this is a local filename that needs repo ID resolution
        if model_id.endswith('.gguf'):
            # Try to get repo_id from metadata first
            resolved_repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id)
            
            # If no metadata, try to infer from cache structure
            if not resolved_repo_id:
                resolved_repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, model_id)
            
            if resolved_repo_id:
                logger.info(f"Resolved local filename '{model_id}' -> '{resolved_repo_id}'")
                repo_id = resolved_repo_id
            else:
                # Fallback: treat as filename (backward compatibility)
                logger.warning(f"Could not resolve repo_id for local file: {model_id}")
        else:
            # Handle aliases and regular repo IDs
            repo_id = resolve_model_id(model_id)
        
        # Get comprehensive model info
        info = get_model_info(repo_id, alias=model_id)
        
        return info
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {str(e)}")


@app.get("/models/{model_id:path}/local-quants")
async def get_local_quantizations(model_id: str):
    """
    Get locally available quantizations for a model.
    
    Args:
        model_id: Model alias or repo ID (URL encoded)
        
    Returns:
        List of locally available quantization files
    """
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
    from pathlib import Path
    
    try:
        # Resolve model ID (handle both aliases and local filenames)
        repo_id = model_id
        
        # First, check if this is a local filename that needs repo ID resolution
        if model_id.endswith('.gguf'):
            # Try to get repo_id from metadata first
            resolved_repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id)
            
            # If no metadata, try to infer from cache structure
            if not resolved_repo_id:
                resolved_repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, model_id)
            
            if resolved_repo_id:
                logger.debug(f"Resolved local filename '{model_id}' -> '{resolved_repo_id}'")
                repo_id = resolved_repo_id
            else:
                # Fallback: treat as filename (backward compatibility)
                logger.warning(f"Could not resolve repo_id for local file: {model_id}")
        else:
            # Handle aliases and regular repo IDs
            repo_id = resolve_model_id(model_id)
        
        # Convert repo_id to cache directory format: models--Author--Name
        cache_name = "models--" + repo_id.replace("/", "--")
        cache_dir = CONFIG.cache_dir / cache_name
        
        local_quants = []
        
        if cache_dir.exists():
            # Check snapshots directory
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        # Find GGUF files
                        gguf_files = list(snapshot.glob("*.gguf"))
                        for file in gguf_files:
                            # Extract quantization from filename
                            name_upper = file.name.upper()
                            for quant in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]:
                                if quant in name_upper:
                                    if quant not in local_quants:
                                        local_quants.append(quant)
                                    break
        
        return {
            "model_id": model_id,
            "repo_id": repo_id,
            "local_quantizations": local_quants,
            "has_local": len(local_quants) > 0
        }
    except Exception as e:
        logger.error(f"Failed to get local quantizations for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch local quantizations: {str(e)}")


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    """Load a model into the cache. Automatically unloads any previously loaded model first."""
    global _models, _model_configs, _model_last_used
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
    
    # Canonicalize the model ID
    original_id = request.model_id
    
    # First, check if this is a local filename that needs repo ID resolution
    if request.model_id.endswith('.gguf'):
        # Try to get repo_id from metadata first
        repo_id = get_repo_id_from_filename(CONFIG.cache_dir, request.model_id)
        
        # If no metadata, try to infer from cache structure
        if not repo_id:
            repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, request.model_id)
        
        if repo_id:
            logger.info(f"Resolved local filename '{request.model_id}' -> '{repo_id}'")
            request.model_id = repo_id
        else:
            # Fallback: treat as filename (backward compatibility)
            logger.warning(f"Could not resolve repo_id for local file: {request.model_id}")
    else:
        # Handle aliases and regular repo IDs
        request.model_id = resolve_model_id(request.model_id)
    
    if request.model_id in _models:
        # Check if backend process is still alive
        model = _models[request.model_id]
        process_alive = False
        if hasattr(model, '_process') and model._process is not None:
            process_alive = model._process.is_running()

        if not process_alive:
            # Dead process – clean up and fall through to reload
            logger.warning(f"Backend process for {request.model_id} died, reloading...")
            _force_unload_model(request.model_id, model)
            _models.pop(request.model_id, None)
            _model_configs.pop(request.model_id, None)
            _model_last_used.pop(request.model_id, None)
        else:
            # Process is alive – check if requested quantization differs from loaded one.
            # If the same quant is already loaded, return early (no-op).
            # If a DIFFERENT quant is requested, fall through to unload + reload.
            loaded_quant = (_model_configs.get(request.model_id, {}).get("quantization") or "").upper()
            requested_quant = (request.quantization or "").upper()

            if not requested_quant or loaded_quant == requested_quant:
                logger.info(f"Model {request.model_id} [{loaded_quant or 'auto'}] already loaded – skipping reload")
                return LoadResponse(
                    success=True,
                    model_id=request.model_id,
                    message="Model already loaded"
                )
            else:
                # Different quant requested – unload current first
                logger.info(
                    f"Switching quant for {request.model_id}: {loaded_quant} → {requested_quant}. Unloading current."
                )
                _force_unload_model(request.model_id, model)
                _models.pop(request.model_id, None)
                _model_configs.pop(request.model_id, None)
                _model_last_used.pop(request.model_id, None)
    
    # If model already loaded, just return success
    if request.model_id in _models:
        _mark_model_used(request.model_id)
        logger.debug(f"Model {request.model_id} already loaded, skipping unload logic")
        return LoadResponse(
            success=True,
            model_id=request.model_id,
            message="Model already loaded"
        )

    # UNLOAD OTHER MODELS (Heuristic: Embedding models can coexist with one LLM)
    from oprel.downloader.aliases import get_model_category
    
    def _is_embedding_model(m_id):
        # Check by category alias
        if get_model_category(m_id) == "embeddings":
            return True
        # Fallback to string name match
        return "embed" in m_id.lower() or "bge-" in m_id.lower()
        
    is_embedding = _is_embedding_model(original_id) or _is_embedding_model(request.model_id)
    
    if not is_embedding:
        # We are loading a main model: Unload other main models, keep embedders
        models_to_unload = list(_models.keys())
        for old_model_id in models_to_unload:
            if _is_embedding_model(old_model_id):
                continue # Keep embedders
                
            logger.info(f"Unloading previous LLM '{old_model_id}' before loading '{request.model_id}'")
            try:
                old_model = _models[old_model_id]
                _force_unload_model(old_model_id, old_model)
                _models.pop(old_model_id, None)
                _model_configs.pop(old_model_id, None)
                _model_last_used.pop(old_model_id, None)
            except Exception as e:
                logger.warning(f"Error unloading previous model {old_model_id}: {e}")
    else:
        # We are loading an embedder: Don't unload anything, just load it
        logger.debug(f"Loading embedding model '{request.model_id}' alongside existing models")
    
    # --- Check if this is an external provider ---
    from oprel.server import db
    p_id = request.model_id
    if "::" in p_id:
        p_id = p_id.split("::", 1)[0]
    elif ":" in p_id:
        p_id = p_id.split(":", 1)[0]
    
    provider = db.get_provider(p_id)
    if provider:
        logger.info(f"Model ID '{request.model_id}' matches external provider '{provider['name']}' - skipping local load")
        return LoadResponse(
            success=True,
            model_id=request.model_id,
            message=f"Model is provided by external provider: {provider['name']}"
        )

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
        
        # Resolve which quantization was actually loaded
        # If not specified, detect from the model's internal state
        actual_quant = request.quantization
        actual_filename = None
        if not actual_quant:
            # Try to get from model's loaded file path
            if hasattr(model, '_model_path') and model._model_path:
                p = str(model._model_path)
                actual_filename = p.split('\\')[-1].split('/')[-1]
            elif hasattr(model, 'model_path') and model.model_path:
                p = str(model.model_path)
                actual_filename = p.split('\\')[-1].split('/')[-1]
            elif hasattr(model, '_process') and model._process:
                if hasattr(model._process, 'model_path') and model._process.model_path:
                    p = str(model._process.model_path)
                    actual_filename = p.split('\\')[-1].split('/')[-1]
            
            if actual_filename:
                # Detect quant from filename
                _QUANT_PATTERNS = [
                    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
                    "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K",
                    "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_K",
                    "Q2_K_XL", "Q2_K_L", "Q2_K_S", "Q2_K",
                    "Q6_K", "Q8_0", "Q4_0", "Q4_1", "F32", "F16", "BF16",
                ]
                fn_upper = actual_filename.upper()
                actual_quant = next((q for q in _QUANT_PATTERNS if q in fn_upper), None)
        
        _model_configs[request.model_id] = {
            "quantization": actual_quant,
            "filename": actual_filename,
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


@app.post("/pull")
async def pull_model_endpoint(request: PullRequest):
    """Download a model with specific quantization. Returns immediately with download_id for progress tracking."""
    from oprel.downloader.aliases import resolve_model_id
    from oprel.telemetry.recommender import recommend_quantization
    from oprel.core.config import Config
    import concurrent.futures
    import base64
    
    try:
        model_id = resolve_model_id(request.model_id)
        
        # Use specified quantization or recommend one
        if request.quantization:
            quantization = request.quantization
            logger.info(f"Downloading model {model_id} with specified quantization: {quantization}")
        else:
            quantization = recommend_quantization()
            logger.info(f"Downloading model {model_id} with recommended quantization: {quantization}")
        
        # Generate unique download ID (base64 encoded to avoid URL issues)
        raw_id = f"{model_id}:{quantization}:{uuid.uuid4().hex[:8]}"
        download_id = base64.urlsafe_b64encode(raw_id.encode()).decode().rstrip('=')
        
        # Register download with raw_id as key
        download_manager.start_download(raw_id, model_id, quantization)
        
        # Start download in background thread
        def download_task():
            from oprel.downloader.hub import download_model_with_progress
            config = Config()
            started = time_module.time()
            
            try:
                # Custom progress callback
                def progress_callback(downloaded, total):
                    download_manager.update_progress(raw_id, downloaded, total)
                
                download_model_with_progress(
                    model_id,
                    quantization=quantization,
                    cache_dir=config.cache_dir,
                    progress_callback=progress_callback
                )
                download_manager.complete_download(raw_id)
                logger.info(f"Download completed: {raw_id}")
                # Persist to DB
                progress = download_manager.get_progress(raw_id)
                db.save_download_log(
                    model_id=model_id,
                    model_name=request.model_id,  # alias
                    quantization=quantization,
                    status="completed",
                    size_bytes=progress.total_bytes if progress else 0,
                    duration_seconds=round(time_module.time() - started, 2),
                )
            except Exception as e:
                logger.error(f"Download failed: {raw_id} - {e}")
                download_manager.fail_download(raw_id, str(e))
                # Persist error to DB
                db.save_download_log(
                    model_id=model_id,
                    model_name=request.model_id,
                    quantization=quantization,
                    status="error",
                    duration_seconds=round(time_module.time() - started, 2),
                    error=str(e),
                )
        
        # Submit to thread pool
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        executor.submit(download_task)
        
        return {
            "success": True,
            "model_id": request.model_id,
            "quantization": quantization,
            "download_id": download_id,
            "message": "Download started. Use /downloads/progress?id={download_id} to track progress."
        }
    except Exception as e:
        logger.error(f"Failed to start download for {request.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(e)}")


@app.get("/downloads/progress")
async def stream_download_progress(id: str):
    """Stream download progress via Server-Sent Events (SSE)"""
    import base64
    
    # Decode the base64 download_id
    try:
        # Add padding if needed
        padding = 4 - (len(id) % 4)
        if padding != 4:
            id += '=' * padding
        raw_id = base64.urlsafe_b64decode(id.encode()).decode()
    except Exception as e:
        logger.error(f"Failed to decode download_id: {e}")
        raise HTTPException(status_code=400, detail="Invalid download_id")
    
    async def event_generator():
        try:
            while True:
                progress = download_manager.get_progress(raw_id)
                
                if not progress:
                    yield f"data: {json.dumps({'error': 'Download not found'})}\n\n"
                    break
                
                # Send progress update
                data = {
                    "model_id": progress.model_id,
                    "quantization": progress.quantization,
                    "status": progress.status,
                    "progress": round(progress.progress, 2),
                    "downloaded": progress.downloaded_bytes,
                    "total": progress.total_bytes,
                    "speed": round(progress.speed_bps, 2),
                    "eta": round(progress.eta_seconds, 1),
                    "error": progress.error
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop streaming if completed or failed
                if progress.status in ["completed", "error"]:
                    break
                
                # Update every 500ms
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            logger.info(f"Client disconnected from progress stream: {raw_id}")
        except Exception as e:
            logger.error(f"Error streaming progress: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/downloads")
async def list_downloads():
    """List all active and recent downloads"""
    downloads = download_manager.get_all_downloads()
    return {
        "downloads": [
            {
                "download_id": download_id,
                "model_id": progress.model_id,
                "quantization": progress.quantization,
                "status": progress.status,
                "progress": round(progress.progress, 2),
                "downloaded": progress.downloaded_bytes,
                "total": progress.total_bytes,
                "speed": round(progress.speed_bps, 2),
                "eta": round(progress.eta_seconds, 1),
                "error": progress.error
            }
            for download_id, progress in downloads.items()
        ]
    }

@app.get("/download-logs")
async def list_download_logs(limit: int = 100):
    """Return persistent download history from DB"""
    return {"logs": db.list_download_logs(limit=limit)}


# Add endpoint to get download by ID for reconnection
@app.get("/downloads/{download_id}")
async def get_download(download_id: str):
    """Get a specific download by ID"""
    import base64
    
    # Decode the base64 download_id
    try:
        padding = 4 - (len(download_id) % 4)
        if padding != 4:
            download_id += '=' * padding
        raw_id = base64.urlsafe_b64decode(download_id.encode()).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid download_id")
    
    progress = download_manager.get_progress(raw_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Download not found")
    
    return {
        "download_id": download_id,
        "model_id": progress.model_id,
        "quantization": progress.quantization,
        "status": progress.status,
        "progress": round(progress.progress, 2),
        "downloaded": progress.downloaded_bytes,
        "total": progress.total_bytes,
        "speed": round(progress.speed_bps, 2),
        "eta": round(progress.eta_seconds, 1),
        "error": progress.error
    }


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text (Conversational)"""
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
    
    # Resolve model ID (handle both aliases and local filenames)
    resolved_model_id = request.model_id
    
    # First, check if this is a local filename that needs repo ID resolution
    if request.model_id.endswith('.gguf'):
        # Try to get repo_id from metadata first
        repo_id = get_repo_id_from_filename(CONFIG.cache_dir, request.model_id)
        
        # If no metadata, try to infer from cache structure
        if not repo_id:
            repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, request.model_id)
        
        if repo_id:
            logger.info(f"Resolved local filename '{request.model_id}' -> '{repo_id}'")
            resolved_model_id = repo_id
        else:
            # Fallback: treat as filename (backward compatibility)
            logger.warning(f"Could not resolve repo_id for local file: {request.model_id}")
    else:
        # Handle aliases and regular repo IDs
        resolved_model_id = resolve_model_id(request.model_id)
    
    # Auto-load logic
    if resolved_model_id not in _models:
        # Check if it's an external provider first
        p_id = resolved_model_id
        if ":" in p_id: p_id = p_id.split(":", 1)[0]
        
        provider = db.get_provider(p_id)
        if not provider:
            # Only try to load locally if not a provider
            load_req = LoadRequest(model_id=resolved_model_id)
            await load_model(load_req)
        else:
            # It's a provider - we don't 'load' it into _models, but we can't use local _client either
            # If called via /generate (GenerateRequest), we must use the provider proxy logic
            # but GenerateRequest doesn't have a messages list. 
            # We'll let provider_chat_proxy handle the prompt if we must, 
            # though it's better to use v1_chat_completions for providers.
            pass

    if resolved_model_id in _models:
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
    else:
        # Check if it's a provider handled by proxy
        p_id = resolved_model_id
        if "::" in p_id: 
            p_id = p_id.split("::", 1)[0]
        elif ":" in p_id:
            p_id = p_id.split(":", 1)[0]
        
        provider = db.get_provider(p_id)
        if provider:
            # Map GenerateRequest to ProviderChatRequest
            # Since we only have a prompt string, we create a single user message
            from oprel.server.daemon import ProviderChatRequest, provider_chat_proxy
            m_name = resolved_model_id.split("::", 1)[1] if "::" in resolved_model_id else (resolved_model_id.split(":", 1)[1] if ":" in resolved_model_id else None)
            if not m_name:
                enabled = provider.get("enabled_model_ids", [])
                m_name = enabled[0] if enabled else resolved_model_id
            
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            proxy_body = ProviderChatRequest(
                model=m_name,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                conversation_id=request.conversation_id
            )
            return await provider_chat_proxy(p_id, proxy_body)
        
        raise HTTPException(status_code=404, detail=f"Model '{resolved_model_id}' not found or not loaded")

    # --- Extract text and images from multimodal prompt ---
    # request.prompt may be a str or a list (OpenAI multimodal format)
    raw_prompt = request.prompt
    images = request.images  # May already be set from v1_chat_completions
    
    if isinstance(raw_prompt, list):
        # Multimodal: extract text + base64 images
        text_parts = []
        if images is None:
            images = []
        for item in raw_prompt:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        try:
                            _, b64 = url.split(",", 1)
                            images.append(b64)
                        except Exception:
                            pass
        text_prompt = " ".join(text_parts)
    else:
        text_prompt = raw_prompt

    # --- Conversation Management ---
    # Only WebUI conversations (via /gui/ route) should persist to DB
    # CLI and API conversations should use ephemeral memory storage
    conv_id = request.conversation_id
    is_persistent = False
    
    # Check if request is from WebUI by examining the conversation_id prefix
    # WebUI always uses 'chat_' prefix for persistent conversations
    if conv_id and conv_id.startswith("chat_"):
        is_persistent = True
    elif not conv_id:
        # No ID provided - generate ephemeral ID for CLI/API usage
        # WebUI will always provide a conversation_id
        conv_id = f"ephemeral_{uuid.uuid4().hex[:12]}"
        is_persistent = False
        
    if request.reset_conversation:
        if is_persistent:
            db.reset_conversation(conv_id)
        else:
            _ephemeral_history.pop(conv_id, None)
            
    if is_persistent:
        history = db.get_conversation_messages(conv_id)
    else:
        history = _ephemeral_history.get(conv_id, [])
    
    # Build text-only chat prompt for the template system
    sys_prompt = request.system_prompt
    
    # --- Mode-aware parameter adjustments ---
    if request.thinking:
        # Thinking mode needs budget
        if request.max_tokens and request.max_tokens < 8192:
             request.max_tokens = 8192
             logger.debug("Thinking mode: bumping max_tokens to 8192")
    else:
        # Fast mode: cap max_tokens for real speed gain
        if request.max_tokens and request.max_tokens > 2048:
            request.max_tokens = 2048
            logger.debug("Fast mode: capping max_tokens to 2048")
    
    # --- RAG: Knowledge Retrieval ---
    context_text = ""
    if request.rag:
        try:
            from oprel.knowledge.knowledge_store import KnowledgeStore
            
            # Use direct internal embedding to avoid HTTP recursion
            async def internal_embed(text, model=None):
                from oprel.downloader.aliases import resolve_model_id
                resolved_embed_model = resolve_model_id(model or "nomic-embed-text")
                res = await get_embeddings(EmbedRequest(input=text, model=resolved_embed_model))
                return res.get("embedding")
                
            from oprel.knowledge.config import TOP_K
            store = KnowledgeStore(embed_func=internal_embed)
            try:
                search_results = await store.search(text_prompt, top_k=TOP_K)
            except Exception as se:
                logger.error(f"RAG search error: {se}", exc_info=True)
                search_results = []
            
            if search_results:
                context_parts = []
                for i, res in enumerate(search_results):
                    source = res.get('metadata', {}).get('filename', 'Unknown source')
                    context_parts.append(f"Source [{i+1}] ({source}):\n{res['text']}")
                
                context_text = "\n\n".join(context_parts)
                logger.info(f"RAG: Found {len(search_results)} relevant chunks")
                
                # INJECT CONTEXT INTO USER PROMPT (Better for small models)
                text_prompt = (
                    f"CONTEXT FROM LOCAL KNOWLEDGE BASE:\n"
                    f"----------------------------------------\n"
                    f"{context_text}\n"
                    f"----------------------------------------\n\n"
                    f"QUESTION: {text_prompt}\n\n"
                    f"INSTRUCTION: Use ONLY the provided context above to answer. Cite source labels [1], [2], etc. "
                    f"If the answer isn't firmly supported by the context, state that you don't have enough information."
                )
                
                logger.info(f"RAG: Injected {len(search_results)} chunks into prompt")
                    
        except Exception as e:
            logger.error(f"RAG search failed: {e}")

    full_prompt = _build_chat_prompt(
        resolved_model_id, 
        history, 
        sys_prompt, 
        text_prompt,  # Always pass plain text to chat template
        thinking=request.thinking
    )
    
    # Inject retrieved context into 'thinking' block if applicable
    if request.thinking and context_text:
        # Prepend context reasoning to prompt if desired? 
        # For now, relying on the system prompt injection above.
        pass
    
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
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repeat_penalty=request.repeat_penalty,
                        stream=True,
                        images=images if images else None
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
                    
                    # Store to DB or Memory
                    if is_persistent:
                        db.add_message(conv_id, "user", raw_prompt)
                        db.add_message(conv_id, "assistant", full_resp)
                    else:
                        if conv_id not in _ephemeral_history:
                            _ephemeral_history[conv_id] = []
                        _ephemeral_history[conv_id].append({"role": "user", "content": text_prompt})
                        _ephemeral_history[conv_id].append({"role": "assistant", "content": full_resp})
                        # Cap memory for ephemeral
                        if len(_ephemeral_history[conv_id]) > 40:
                            _ephemeral_history[conv_id] = _ephemeral_history[conv_id][-40:]
                    
                    yield "data: [DONE]\n\n"
                    
                    # Log to analytics DB
                    prompt_tokens_est = len(text_prompt.split()) * 1.3
                    db.add_inference_log(
                        model_id=resolved_model_id,
                        prompt_tokens=int(prompt_tokens_est),
                        completion_tokens=token_count,
                        latency_ms=duration * 1000,
                        tps=token_count / duration if duration > 0 else 0
                    )
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
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                stream=False,
                images=images if images else None
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
            if is_persistent:
                db.add_message(conv_id, "user", raw_prompt)
                db.add_message(conv_id, "assistant", text)
            else:
                if conv_id not in _ephemeral_history:
                    _ephemeral_history[conv_id] = []
                _ephemeral_history[conv_id].append({"role": "user", "content": text_prompt})
                _ephemeral_history[conv_id].append({"role": "assistant", "content": text})
                if len(_ephemeral_history[conv_id]) > 40:
                    _ephemeral_history[conv_id] = _ephemeral_history[conv_id][-40:]
            
            # Log to analytics DB
            prompt_tokens_est = len(text_prompt.split()) * 1.3
            completion_tokens_est = len(text.split()) * 1.3
            db.add_inference_log(
                model_id=resolved_model_id,
                prompt_tokens=int(prompt_tokens_est),
                completion_tokens=int(completion_tokens_est),
                latency_ms=duration * 1000,
                tps=token_est / duration if duration > 0 else 0
            )

            # To get count we would query, but just returning len(history)+2 is approx
            return GenerateResponse(
                text=text,
                model_id=resolved_model_id,
                conversation_id=conv_id,
                message_count=len(history) + 2
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/embedding")
@app.post("/v1/embeddings")
async def get_embeddings(request: EmbedRequest):
    """
    Generate embeddings for input text.
    Ollama and OpenAI compatible.
    """
    # Use load_model logic to get/start the embedding model
    # Resolve and load
    from oprel.downloader.aliases import resolve_model_id
    resolved_id = resolve_model_id(request.model)
    
    load_req = LoadRequest(model_id=resolved_id)
    await load_model(load_req)
    
    if resolved_id not in _models:
        logger.error(f"Model '{resolved_id}' not found in _models after loading. Available: {list(_models.keys())}")
        raise HTTPException(status_code=500, detail=f"Model '{resolved_id}' failed to stay in cache")
        
    model = _models[resolved_id]
    
    # Wait for the embedding model's backend process to be ready.
    # load_model() starts the process asynchronously; if the first embedding
    # request arrives before the HTTP server inside the process is accepting
    # connections, we get a 500.  A short probe loop prevents this.
    if hasattr(model, '_process') and model._process:
        backend_port = model._process.port
        _deadline = time_module.time() + 15  # 15-second maximum wait
        _ready = False
        while time_module.time() < _deadline:
            try:
                async with httpx.AsyncClient(timeout=1.0) as _hc:
                    _probe = await _hc.get(f"http://127.0.0.1:{backend_port}/health")
                    if _probe.status_code < 500:
                        _ready = True
                        break
            except Exception:
                pass
            await asyncio.sleep(0.25)
        
        if not _ready:
            raise HTTPException(status_code=503, detail=f"Embedding backend on port {backend_port} did not become ready in time")
    
    # Internal _client is the ModelProcess/llama-server
    # It has its own /embedding endpoint
    
    is_single = isinstance(request.input, str)
    inputs = [request.input] if is_single else request.input
    embeddings = []
    backend_port = model._process.port

    async def _embed_chunk(hc: httpx.AsyncClient, chunk: str) -> list:
        """Embed a single chunk of text via the llama-server backend."""
        # Try modern OpenAI-compatible endpoint first (/v1/embeddings),
        # then fall back to the legacy endpoint (/embedding).
        # Newer llama-server builds return 501 for /embedding.
        last_exc: Exception | None = None
        for endpoint, payload in [
            (
                f"http://127.0.0.1:{backend_port}/v1/embeddings",
                {"input": chunk, "model": "nomic-embed-text"},
            ),
            (
                f"http://127.0.0.1:{backend_port}/embedding",
                {"content": chunk},
            ),
        ]:
            try:
                resp = await hc.post(endpoint, json=payload, timeout=30.0)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (404, 501):
                    last_exc = exc
                    continue   # try next endpoint
                raise
        else:
            raise last_exc  # both endpoints failed

        res = resp.json()

        # llama-server returns at least 3 different shapes depending on version:
        #   1. {"embedding": [0.1, ...]}              — standard single vector
        #   2. [{"index": 0, "embedding": [[...]]}]   — list of objects (newer builds)
        #   3. {"data": [{"embedding": [...]}]}        — OpenAI-compat wrapper
        if isinstance(res, dict):
            if "embedding" in res:
                vec = res["embedding"]
                if vec and isinstance(vec[0], list):   # nested [[...]] → flatten
                    vec = vec[0]
                return vec
            elif "data" in res:
                return res["data"][0]["embedding"]
            else:
                raise ValueError(f"Unrecognised dict response from backend: {list(res.keys())}")
        elif isinstance(res, list):
            first = res[0]
            vec = first["embedding"]
            if vec and isinstance(vec[0], list):
                vec = vec[0]
            return vec
        else:
            raise ValueError(f"Unexpected response type from backend: {type(res)}")

    async def _embed_text(hc: httpx.AsyncClient, text: str) -> list:
        """
        Embed text, automatically chunking if it is too long for the model.
        Chunks are mean-pooled and L2-normalised so the result is a single vector.

        Uses a conservative 150-word chunk size (~225 tokens) so it fits safely
        even inside a 512-token context window.  If a chunk still fails the size
        is halved (binary search), down to a floor of 32 words.
        """
        # Fast path — try the full text first.
        try:
            return await _embed_chunk(hc, text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 500:
                raise
            logger.info(
                f"Embedding: text too long ({len(text.split())} words) — switching to chunked mode"
            )

        # Chunk into word windows that safely fit the model's context window.
        # Start conservative (150 words ≈ 225 tokens) and halve on failure.
        chunk_size = 150
        overlap    = 20
        floor      = 32

        while chunk_size >= floor:
            words  = text.split()
            chunks = []
            start  = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunks.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start += chunk_size - overlap

            # Try embedding every chunk at this size.
            try:
                chunk_vecs = [await _embed_chunk(hc, c) for c in chunks]
                break  # success — fall through to pooling
            except httpx.HTTPStatusError as exc2:
                if exc2.response.status_code != 500 or chunk_size <= floor:
                    raise
                chunk_size = max(chunk_size // 2, floor)
                logger.debug(f"Embedding: chunk still too large, retrying with {chunk_size} words")
        else:
            raise RuntimeError(
                f"Could not embed text: backend returned 500 even for {floor}-word chunks"
            )

        # Mean-pool chunk vectors into one document vector, then L2-normalise.
        dim    = len(chunk_vecs[0])
        pooled = [sum(v[i] for v in chunk_vecs) / len(chunk_vecs) for i in range(dim)]
        mag    = sum(x * x for x in pooled) ** 0.5
        return [x / mag for x in pooled] if mag > 0 else pooled

    try:
        async with httpx.AsyncClient(timeout=60.0) as hc:
            for text in inputs:
                vec = await _embed_text(hc, text)
                embeddings.append(vec)

        if is_single:
            return {"embedding": embeddings[0]}
        else:
            return {"embeddings": embeddings}

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# --- Conversation Endpoints ---

@app.get("/conversations")
async def list_conversations():
    """List active conversations"""
    return db.list_conversations()


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    messages = db.get_conversation_messages(conversation_id)
    if not messages:
        return []
    return messages


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    db.delete_conversation(conversation_id)
    return {"success": True}


@app.post("/conversations/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    db.reset_conversation(conversation_id)
    return {"success": True}

# --- Analytics Endpoints ---

@app.get("/analytics/summary")
async def get_analytics_summary(days: int = 7):
    """Get usage analytics summary"""
    return db.get_inference_summary(days)


@app.put("/conversations/{conversation_id}/title")
async def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    """Rename a conversation"""
    db.rename_conversation(conversation_id, request.title)
    return {"success": True}


@app.post("/unload", response_model=UnloadResponse)
async def unload_model_post(request: UnloadRequest):
    return await unload_model(request.model_id) # Reuse logic


@app.delete("/unload/{model_id:path}", response_model=UnloadResponse)
async def unload_model(model_id: str):
    global _models, _model_configs, _model_last_used
    from oprel.downloader.aliases import resolve_model_id
    
    # Canonicalize
    model_id = resolve_model_id(model_id)
    
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


# --- Indexing & Knowledge Endpoints ---

@app.post("/index/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document into the knowledge base."""
    from oprel.knowledge.sync_engine import SyncEngine
    from oprel.knowledge.knowledge_store import KnowledgeStore
    import shutil
    
    # Ensure knowledge directory exists
    knowledge_dir = CONFIG.cache_dir / "knowledge_files"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = knowledge_dir / file.filename
    
    try:
        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Initialize internal embedding function (same as in /generate)
        async def internal_embed(text, model=None):
            from oprel.downloader.aliases import resolve_model_id
            resolved_embed_model = resolve_model_id(model or "nomic-embed-text")
            res = await get_embeddings(EmbedRequest(input=text, model=resolved_embed_model))
            return res.get("embedding")
            
        # Index document
        store = KnowledgeStore(embed_func=internal_embed)
        engine = SyncEngine(store)
        
        await engine.ingest_file(file_path)
        
        return {
            "success": True, 
            "filename": file.filename,
            "message": "Document indexed successfully"
        }
    except Exception as e:
        logger.error(f"Failed to ingest file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List indexed documents in the knowledge base."""
    from oprel.knowledge.knowledge_store import KnowledgeStore
    
    # Initialize store to access BM25 docs (metadata)
    store = KnowledgeStore()
    
    # Deduplicate by filename from BM25 metadata
    docs_map = {}
    for doc in store.bm25_docs:
        meta = doc.get("metadata", {})
        fname = meta.get("filename", "Unknown")
        if fname not in docs_map:
            docs_map[fname] = {
                "id": doc["id"],
                "filename": fname,
                "size_bytes": 0, # Should fetch from disk
                "indexed_at": meta.get("timestamp", ""),
                "chunks": 0
            }
        docs_map[fname]["chunks"] += 1
        
    return list(docs_map.values())


@app.delete("/index/documents/{filename}")
async def delete_indexed_document(filename: str):
    """Remove a document from the index (Not fully implemented in KnowledgeStore yet)"""
    # For now just return a placeholder or implement in KnowledgeStore
    return {"success": False, "message": "Deletion via API not yet implemented"}


@app.delete("/models/{model_id:path}/quant/{quantization}")
async def delete_model_quant(model_id: str, quantization: str):
    """
    Delete a specific quantization of a model from local disk.
    Removes the GGUF file, metadata sidecar, and unloads if active.
    """
    from oprel.downloader.aliases import resolve_model_id, get_best_alias_for_repo
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    # Resolve alias -> repo_id
    repo_id = resolve_model_id(model_id)

    # Find the GGUF file in cache
    cache_dir = CONFIG.cache_dir
    deleted_files: list[str] = []
    quant_upper = quantization.upper()

    try:
        for gguf_file in cache_dir.rglob("*.gguf"):
            if not gguf_file.is_file():
                continue

            filename = gguf_file.name
            fname_lower = filename.lower()

            # Skip side-car files
            if 'mmproj' in fname_lower or fname_lower.startswith('vision-') or fname_lower.startswith('clip-'):
                continue

            # Check quant match in filename
            if quant_upper not in filename.upper():
                continue

            # Verify it belongs to this repo
            file_repo_id = get_repo_id_from_filename(cache_dir, filename)
            if not file_repo_id:
                file_repo_id = infer_repo_id_from_cache(cache_dir, filename)
            if file_repo_id and file_repo_id != repo_id:
                continue  # different model

            # Delete the GGUF file
            gguf_file.unlink(missing_ok=True)
            deleted_files.append(filename)
            logger.info(f"Deleted model file: {gguf_file}")

            # Delete metadata sidecar
            meta_file = cache_dir / ".metadata" / f"{filename}.json"
            if meta_file.exists():
                meta_file.unlink()
                logger.info(f"Deleted metadata: {meta_file}")

        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"No downloaded file found for {model_id} / {quantization}"
            )

        # Unload from RAM if currently loaded with this quant
        global _models, _model_configs, _model_last_used
        for loaded_id in list(_models.keys()):
            cfg = _model_configs.get(loaded_id, {})
            if loaded_id in (model_id, repo_id) and cfg.get("quantization", "").upper() == quant_upper:
                logger.info(f"Unloading active model {loaded_id} after file deletion")
                _force_unload_model(loaded_id)
                _models.pop(loaded_id, None)
                _model_configs.pop(loaded_id, None)
                _model_last_used.pop(loaded_id, None)

        return {"success": True, "deleted": deleted_files, "model_id": model_id, "quantization": quantization}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id} / {quantization}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user", response_model=Optional[UserProfile])
async def get_user_profile():
    """Get the current user profile"""
    user = db.get_user()
    return user

@app.post("/user", response_model=UserProfile)
async def update_user_profile(user: UserProfile):
    """Create or update the user profile"""
    result = db.set_user(user.name, user.role)
    return result

@app.get("/user/settings", response_model=UserSettings)
async def get_user_settings():
    """Get the current user settings from DB"""
    settings = db.get_user_settings()
    if not settings:
        return UserSettings()
    return UserSettings(**settings)

@app.post("/user/settings", response_model=UserSettings)
async def update_user_settings(settings: UserSettings):
    """Update user settings in DB"""
    db.set_user_settings(settings.dict())
    return settings


# ============================================================================
# OpenAI & Ollama API Compatibility (Week 14)
# ============================================================================

class OpenAIChatMessage(BaseModel):
    role: str
    content: Any


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: Optional[int] = 8192
    stream: bool = False
    conversation_id: Optional[str] = None
    thinking: bool = False
    rag: bool = False  # Enable Retrieval-Augmented Generation


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: Optional[int] = 8192
    stream: bool = False


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: OpenAIChatRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint"""
    # Detect if request is from WebUI by checking Referer header
    referer = http_request.headers.get("referer", "")
    is_webui_request = "/gui/" in referer or referer.endswith("/gui")

    # --- Check for external provider ID ---
    p_id = request.model
    m_name = None
    if "::" in p_id:
        pts = p_id.split("::", 1)
        p_id, m_name = pts[0], pts[1]
    elif ":" in p_id:
        pts = p_id.split(":", 1)
        p_id, m_name = pts[0], pts[1]
    
    provider = db.get_provider(p_id)
    if provider:
        # Redirect directly to provider proxy using structured messages
        from oprel.server.daemon import ProviderChatRequest, provider_chat_proxy
        
        # Use provided model name or first enabled one
        if not m_name:
            enabled = provider.get("enabled_model_ids", [])
            m_name = enabled[0] if enabled else request.model
            
        proxy_body = ProviderChatRequest(
            model=m_name,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            conversation_id=request.conversation_id,
            rag=request.rag
        )
        resp = await provider_chat_proxy(p_id, proxy_body)
        if hasattr(resp, "text"):
            # It's a GenerateResponse, convert to OpenAI format
            return {
                "id": f"chatcmpl-{int(time_module.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time_module.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(resp.text.split()),
                    "total_tokens": len(resp.text.split())
                }
            }
        return resp
    
    # Extract the last user message - may be str or multimodal list
    prompt = request.messages[-1].content if request.messages else ""
    
    system_prompt = None
    conversation_history = []
    
    # Separate system prompt from history
    for msg in request.messages[:-1]:
        if msg.role == 'system':
            system_prompt = msg.content if isinstance(msg.content, str) else ""
        else:
            conversation_history.append({"role": msg.role, "content": msg.content})
    
    # Create GenerateRequest - prompt can be str or list (multimodal)
    gen_request = GenerateRequest(
        model_id=request.model,
        prompt=prompt,  # Pass as-is; generate_text() handles multimodal extraction
        max_tokens=request.max_tokens or 8192,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stream=request.stream,
        system_prompt=system_prompt,
        thinking=request.thinking,
        rag=request.rag
    )
    
    # For vision requests (multimodal content), ensure max_tokens is at least 8192
    if isinstance(prompt, list):
        has_image = any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in prompt
        )
        if has_image and gen_request.max_tokens < 8192:
            gen_request.max_tokens = 8192
            logger.info("Vision request detected — bumping max_tokens to 8192")
    
    # Set up conversation - only persist for WebUI requests
    conv_id = request.conversation_id
    if is_webui_request:
        # WebUI request - persist to database
        if not conv_id:
            from oprel.utils.chat_templates import _get_content_text
            title_text = _get_content_text(prompt)
            conv_id = db.create_conversation(request.model, title=title_text[:30] + "..." if len(title_text) > 30 else title_text)
            
            # Save conversation history to DB
            for msg in conversation_history:
                db.add_message(conv_id, msg["role"], msg["content"])
        elif conv_id.startswith("temp-"):
            # New conversation starting with temp ID - convert to real
            from oprel.utils.chat_templates import _get_content_text
            title_text = _get_content_text(prompt)
            conv_id = db.create_conversation(request.model, title=title_text[:30] + "..." if len(title_text) > 30 else title_text)
            for msg in conversation_history:
                db.add_message(conv_id, msg["role"], msg["content"])
        # If it starts with 'ephemeral-', we DON'T create a DB entry (prevents duplicates)
    else:
        # CLI/API request - use ephemeral conversation ID
        if not conv_id:
            conv_id = f"ephemeral_{uuid.uuid4().hex[:12]}"

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
            
            # Final buffer flush
            if buffer.startswith("data: "):
                token = buffer[6:]
                if token and token != "[DONE]" and not token.startswith("[ERROR]"):
                    yield f"data: {json.dumps({
                        'id': request_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time_module.time()),
                        'model': request.model,
                        'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]
                    })}\n\n"
            
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
        stream=request.stream,
        rag=getattr(request, 'rag', False)
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
            
            # Final buffer flush
            if buffer.startswith("data: "):
                token = buffer[6:]
                if token and token != "[DONE]" and not token.startswith("[ERROR]"):
                    yield f"data: {json.dumps({
                        'id': request_id,
                        'object': 'text_completion',
                        'created': int(time_module.time()),
                        'model': request.model,
                        'choices': [{'text': token, 'index': 0, 'finish_reason': None}]
                    })}\n\n"
            
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
    """OpenAI-compatible model list endpoint with categorization tags.
    
    Uses alias names as canonical IDs. Locally-cached GGUF files are resolved back
    to their repo_id -> alias so they don't appear as a second duplicate entry.
    """
    from oprel.downloader.aliases import OFFICIAL_REPOS, get_model_category, get_best_alias_for_repo
    from oprel.downloader.cache import list_cached_models
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    models = []

    # Map internal categories to requested WebUI tags
    category_to_tags = {
        "text-generation": ["text llm", "text"],
        "coding": ["coding"],
        "reasoning": ["reasoning"],
        "vision": ["vision llms", "text+vision"]
    }

    # Categories to skip
    skip_categories = ["embeddings", "text-to-image", "text-to-video"]

    # Build flat alias -> repo_id and repo_id -> alias maps for fast lookup
    alias_to_repo: dict = {}
    repo_to_alias: dict = {}
    for _cat, _alias_dict in OFFICIAL_REPOS.items():
        for _alias, _repo_id in _alias_dict.items():
            alias_to_repo[_alias] = _repo_id
            repo_to_alias[_repo_id] = _alias

    try:
        # --- Step A: Resolve every cached GGUF file -> alias ---
        # This tells us which aliases are actually downloaded locally.
        cached = list_cached_models()
        downloaded_aliases: set = set()     # aliases confirmed on disk
        unregistered_cached: list = []      # (raw_id, model_info) for models with no alias

        for model_info in cached:
            filename = model_info.get('name', '')
            if not filename:
                continue

            # Skip mmproj / vision encoder side-files
            fname_lower = filename.lower()
            if ('mmproj' in fname_lower
                    or fname_lower.startswith('vision-')
                    or fname_lower.startswith('clip-')):
                continue

            # Resolve GGUF filename -> HuggingFace repo_id
            repo_id = get_repo_id_from_filename(CONFIG.cache_dir, filename)
            if not repo_id:
                repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, filename)

            if repo_id:
                best_alias = get_best_alias_for_repo(repo_id)
                if best_alias:
                    # Known alias: mark as downloaded and stop here
                    downloaded_aliases.add(best_alias)
                else:
                    # Repo exists on disk but is not in the registry
                    unregistered_cached.append((repo_id, model_info))
            else:
                # Completely unknown file (manually placed, etc.)
                unregistered_cached.append((filename, model_info))

        # Also treat models currently loaded in RAM as downloaded
        for loaded_id in list(_models.keys()):
            best_alias = get_best_alias_for_repo(loaded_id)
            if best_alias:
                downloaded_aliases.add(best_alias)
            elif loaded_id in alias_to_repo:
                # loaded_id is already an alias
                downloaded_aliases.add(loaded_id)

        # --- Step B: Emit one entry per registry alias ---
        for category, alias_dict in OFFICIAL_REPOS.items():
            if category in skip_categories:
                continue

            tags = category_to_tags.get(category, ["text"])

            for alias, repo_id in alias_dict.items():
                is_downloaded = alias in downloaded_aliases
                is_loaded = alias in _models or repo_id in _models

                models.append({
                    "id": alias,
                    "object": "model",
                    "created": int(time_module.time()),
                    "owned_by": "oprel",
                    "tags": tags,
                    "category": category,
                    "loaded": is_loaded,
                    "downloaded": is_downloaded,
                })

        # --- Step C: Add unregistered local models (not covered by any alias) ---
        all_registry_repo_ids = set(alias_to_repo.values())
        seen_unregistered: set = set()

        for raw_id, model_info in unregistered_cached:
            # Skip if this repo is already represented by a registry alias
            if raw_id in all_registry_repo_ids:
                continue
            if raw_id in seen_unregistered:
                continue
            seen_unregistered.add(raw_id)

            cat = get_model_category(raw_id)

            raw_id_lower = raw_id.lower()
            is_embedding = any(kw in raw_id_lower for kw in ['embed', 'embedding', 'nomic-embed', 'bge-m3', 'minilm'])
            is_image_gen = any(kw in raw_id_lower for kw in ['flux', 'stable-diffusion', 'sdxl', 'sana', 'pixart', 'playground'])

            if is_embedding or is_image_gen or cat in skip_categories:
                continue

            tags = category_to_tags.get(cat, ["text"])
            display_name = raw_id.split("/")[-1] if "/" in raw_id else raw_id

            models.append({
                "id": raw_id,
                "object": "model",
                "created": int(model_info.get('modified', datetime.now()).timestamp()),
                "owned_by": "oprel",
                "tags": tags,
                "category": cat or "text-generation",
                "loaded": raw_id in _models,
                "downloaded": True,
                "name": display_name,
            })

        # --- Step D: Add models from external providers ---
        providers = db.list_providers()
        for p in providers:
            if not p.get("enabled", True):
                continue
            
            p_id = p["id"]
            p_type = p["type"]
            enabled_models = p.get("enabled_model_ids", [])
            
            # If no specific models enabled, show the provider itself as a model
            if not enabled_models:
                models.append({
                    "id": p_id,
                    "object": "model",
                    "created": int(time_module.time()),
                    "owned_by": p_id,
                    "tags": ["external", p_type],
                    "category": "external",
                    "loaded": True,
                    "downloaded": True,
                    "name": f"{p['name']} (Provider)"
                })
            else:
                for target_m_id in enabled_models:
                    # Provide a unique composite ID (Standard :: format)
                    composite_id = f"{p_id}::{target_m_id}"
                    models.append({
                        "id": composite_id,
                        "object": "model",
                        "created": int(time_module.time()),
                        "owned_by": p_id,
                        "tags": ["external", p_type],
                        "category": "external",
                        "loaded": True,
                        "downloaded": True,
                        "name": f"{target_m_id} ({p['name']})"
                    })

    except Exception as e:
        logger.warning(f"Could not list models for V1 API: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return {
        "object": "list",
        "data": models
    }


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


class ProviderChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: bool = True  # Default to True for compatibility with existing UI
    conversation_id: Optional[str] = None
    rag: bool = False  # Enable Retrieval-Augmented Generation


# ─────────────────────────────────────────────────────────────────────────────
# External AI Provider Config Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class ProviderUpsertRequest(BaseModel):
    id: str
    name: str
    type: str          # "openai" | "gemini" | "openai-compatible" | "nvidia" | "groq" | "openrouter"
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    enabled_model_ids: List[str] = []
    available_model_ids: List[str] = []
    last_fetched: Optional[str] = None


@app.get("/providers")
async def list_providers_route():
    """List all configured external AI providers."""
    return db.list_providers()


@app.get("/providers/{provider_id}")
async def get_provider_route(provider_id: str):
    p = db.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="Provider not found")
    return p


@app.post("/providers/{provider_id}")
async def upsert_provider_route(provider_id: str, body: ProviderUpsertRequest):
    """Create or update a provider configuration."""
    data = body.dict()
    data["id"] = provider_id  # ensure ID matches path
    result = db.upsert_provider(data)
    return result


@app.delete("/providers/{provider_id}")
async def delete_provider_route(provider_id: str):
    db.delete_provider(provider_id)
    return {"success": True, "id": provider_id}


@app.get("/providers/{provider_id}/models")
async def fetch_provider_models_proxy(provider_id: str):
    """Proxy to fetch models from an external provider (bypasses CORS)."""
    p = db.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    api_key = p.get("api_key")
    base_url = p.get("base_url")
    p_type = p.get("type", "openai")

    # Inferred base URL if empty
    presets = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    
    url = base_url or presets.get(p_type, "")
    if not url and p_type != "gemini":
        raise HTTPException(status_code=400, detail="Base URL is missing")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if p_type == "gemini":
                res = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
                res.raise_for_status()
                data = res.json()
                models = [m["name"].replace("models/", "") for m in data.get("models", []) 
                          if "generateContent" in m.get("supportedGenerationMethods", [])]
                return sorted(models)
            else:
                res = await client.get(f"{url}/models", headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://oprel.ai",
                    "X-Title": "OPREL"
                })
                res.raise_for_status()
                data = res.json()
                models = [m["id"] for m in data.get("data", [])]
                return sorted(models)
        except Exception as e:
            logger.error(f"Failed to fetch models for {provider_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Provider error: {str(e)}")


@app.post("/providers/{provider_id}/chat")
async def provider_chat_proxy(provider_id: str, body: ProviderChatRequest):
    """Proxy streaming chat to an external provider (bypasses CORS)."""
    p = db.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    api_key = p.get("api_key")
    base_url = p.get("base_url")
    p_type = p.get("type", "openai")

    # --- 1. Manage Conversation Persistence ---
    # If no ID provided, create a new one to ensure persistence
    effective_conv_id = body.conversation_id
    if not effective_conv_id:
        # Create a real DB conversation
        title = "New Chat"
        if body.messages:
            first_msg = body.messages[0].get("content", "")
            if isinstance(first_msg, str) and first_msg:
                title = first_msg[:60] + ("..." if len(first_msg) > 60 else "")
                
        effective_conv_id = db.create_conversation(model_id=body.model, title=title)

    presets = {
        "openai": "https://api.openai.com/v1",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    url = base_url or presets.get(p_type, "")

    # --- 0. RAG RETRIEVAL (Pre-processing) ---
    context_text = ""
    # Inject context into last user message if RAG is enabled
    if body.rag and body.messages:
        last_user_msg = None
        for m in reversed(body.messages):
            if m["role"] == "user":
                last_user_msg = m
                break
        
        if last_user_msg:
            query = str(last_user_msg["content"])
            last_user_msg["_original_content"] = query  # Preserve for DB
            try:
                from oprel.knowledge.knowledge_store import KnowledgeStore
                async def internal_embed(text, model=None):
                    from oprel.downloader.aliases import resolve_model_id
                    res = await get_embeddings(EmbedRequest(input=text, model=resolve_model_id(model or "nomic-embed-text")))
                    return res.get("embedding")
                
                try:
                    from oprel.knowledge.config import TOP_K
                except ImportError:
                    TOP_K = 5
                    
                store = KnowledgeStore(embed_func=internal_embed)
                search_results = await store.search(query, top_k=TOP_K)
                
                if search_results:
                    # ── Token-budget-aware injection ──────────────────────────────
                    # Rough token estimate: 1 token ≈ 4 chars.
                    # Reserve 1500 tokens for the model's reply.
                    # Provider max context defaults to 4096; use body.max_tokens as a
                    # proxy for how much the user expects to receive.
                    CHARS_PER_TOKEN = 4
                    REPLY_RESERVE   = max(body.max_tokens or 1024, 1024)  # tokens for reply
                    CTX_LIMIT       = 4096   # conservative provider context budget (tokens)
                    context_budget  = (CTX_LIMIT - REPLY_RESERVE) * CHARS_PER_TOKEN

                    # How many chars are already used by existing messages?
                    existing_chars = sum(
                        len(str(m.get("content", "")))
                        for m in body.messages
                    )
                    # Overhead for the wrapper text (~80 tokens)
                    wrapper_overhead = 80 * CHARS_PER_TOKEN
                    available_chars = context_budget - existing_chars - wrapper_overhead

                    context_parts = []
                    used_chars = 0
                    for i, res in enumerate(search_results):
                        source   = res.get("metadata", {}).get("filename", "Unknown source")
                        chunk    = f"Source [{i+1}] ({source}):\n{res['text']}"
                        chunk_chars = len(chunk)

                        if used_chars + chunk_chars > available_chars:
                            # Try to include a truncated version of this chunk
                            remaining = available_chars - used_chars
                            if remaining > 200:   # only worth adding if > ~50 tokens
                                truncated = chunk[:remaining - 4] + " ..."
                                context_parts.append(truncated)
                            break

                        context_parts.append(chunk)
                        used_chars += chunk_chars + 2  # +2 for "\n\n" separator

                    if context_parts:
                        context_text = "\n\n".join(context_parts)
                        last_user_msg["content"] = (
                            f"CONTEXT FROM LOCAL KNOWLEDGE BASE:\n"
                            f"----------------------------------------\n"
                            f"{context_text}\n"
                            f"----------------------------------------\n\n"
                            f"QUESTION: {query}\n\n"
                            f"INSTRUCTION: Use ONLY the provided context above to answer. Cite source labels [1], [2], etc. "
                            f"If the answer isn't firmly supported by the context, state that you don't have enough information."
                        )
                        logger.info(f"Provider RAG: Injected {len(context_parts)}/{len(search_results)} chunks "
                                    f"({used_chars // CHARS_PER_TOKEN} est. tokens)")
            except Exception as e:
                logger.error(f"Provider RAG search failed: {e}")


    # ── 1. Save USER message to DB ───────────────────
    user_msg = body.messages[-1] if body.messages else None
    if user_msg:
        # We save the ORIGINAL query to DB, not the injected one, 
        # to keep the chat history clean for the user.
        db.add_message(effective_conv_id, user_msg["role"], user_msg.get("_original_content", user_msg["content"]))

    # ── 2. Handle Non-Streaming Request ──────────────
    if not body.stream:
        async with httpx.AsyncClient(timeout=60.0) as client:
            full_response = ""
            if p_type == "gemini":
                model_name = body.model if body.model.startswith("models/") else f"models/{body.model}"
                system_msg = next((m for m in body.messages if m["role"] == "system"), None)
                contents = []
                use_system_instruction = "gemma" not in body.model.lower()

                for i, m in enumerate(body.messages):
                    if m["role"] == "system": continue
                    role = "model" if m["role"] == "assistant" else "user"
                    content_text = str(m["content"])
                    if not use_system_instruction and system_msg and i == 1:
                        content_text = f"{system_msg['content']}\n\n{content_text}"
                    contents.append({"role": role, "parts": [{"text": content_text}]})
                
                gemini_body = {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": body.max_tokens or 4096,
                        "temperature": body.temperature if body.temperature is not None else 0.7,
                    }
                }
                if use_system_instruction and system_msg:
                    gemini_body["systemInstruction"] = {"parts": [{"text": str(system_msg["content"])}]}

                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}",
                    json=gemini_body
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"Gemini API Error: {resp.text}")
                
                data = resp.json()
                try:
                    full_response = data["candidates"][0]["content"]["parts"][0]["text"]
                except:
                    full_response = "Error parsing Gemini response"
            else:
                # OpenAI / NVIDIA / Groq
                # Strip internal-only keys before sending to provider
                clean_messages = [
                    {k: v for k, v in m.items() if not k.startswith("_")}
                    for m in body.messages
                ]
                resp = await client.post(
                    f"{url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": body.model,
                        "messages": clean_messages,
                        "stream": False,
                        "max_tokens": body.max_tokens,
                        "temperature": body.temperature,
                    }
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"Provider {p_type} Error: {resp.text}")
                
                data = resp.json()
                full_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Save Answer and Log Analytics
            if full_response.strip():
                db.add_message(effective_conv_id, "assistant", full_response)
                db.add_inference_log(
                    model_id=body.model,
                    prompt_tokens=len(str(body.messages[-1]["content"])) // 4,
                    completion_tokens=len(full_response) // 4,
                    latency_ms=100.0, tps=0.0
                )
            
            # Return compatible format for v1_chat_completions and generate_text
            # It will be a dict that has a .text field in v1_chat_completions? NO.
            # Wait, v1_chat_completions thinks it's a GenerateResponse.
            # But generate_text returns a GenerateResponse object.
            
            # For FastAPI to handle it correctly, I should return a GenerateResponse-like object
            from oprel.server.daemon import GenerateResponse
            return GenerateResponse(
                text=full_response,
                model_id=body.model,
                conversation_id=effective_conv_id,
                message_count=len(body.messages) + 1
            )

    # ── 3. Handle Streaming Request ──────────────────
    async def stream_generator(conv_id):
        nonlocal api_key
        full_response = ""
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            if p_type == "gemini":
                model_name = body.model if body.model.startswith("models/") else f"models/{body.model}"
                system_msg = next((m for m in body.messages if m["role"] == "system"), None)
                contents = []
                use_system_instruction = "gemma" not in body.model.lower()

                for i, m in enumerate(body.messages):
                    if m["role"] == "system": continue
                    role = "model" if m["role"] == "assistant" else "user"
                    content_text = str(m["content"])
                    if not use_system_instruction and system_msg and i == 1:
                        content_text = f"{system_msg['content']}\n\n{content_text}"
                    contents.append({"role": role, "parts": [{"text": content_text}]})
                
                gemini_body = {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": body.max_tokens or 4096,
                        "temperature": body.temperature if body.temperature is not None else 0.7,
                    }
                }
                if use_system_instruction and system_msg:
                    gemini_body["systemInstruction"] = {"parts": [{"text": str(system_msg["content"])}]}

                try:
                    async with client.stream(
                        "POST", 
                        f"https://generativelanguage.googleapis.com/v1beta/{model_name}:streamGenerateContent?key={api_key}&alt=sse",
                        json=gemini_body
                    ) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            error_msg = f"Gemini API Error {resp.status_code}: {err_body.decode()}"
                            yield f"data: {json.dumps({'error': error_msg})}\n\n"
                            return

                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    json_data = json.loads(line[6:])
                                    token = json_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                    if token:
                                        full_response += token
                                except: pass
                            yield line + "\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"
            else:
                # OpenAI / NVIDIA / Groq
                # Strip internal-only keys (e.g. _original_content) before sending to provider
                clean_messages = [
                    {k: v for k, v in m.items() if not k.startswith("_")}
                    for m in body.messages
                ]
                async with client.stream(
                    "POST", f"{url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": body.model, "messages": clean_messages, "stream": True,
                        "max_tokens": body.max_tokens, "temperature": body.temperature,
                    }
                ) as resp:
                    # Surface provider errors immediately instead of silently streaming nothing
                    if resp.status_code not in (200, 206):
                        err_body = await resp.aread()
                        error_msg = f"Provider {p_type} error {resp.status_code}: {err_body.decode()}"
                        logger.error(f"Streaming provider error: {error_msg}")
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if token: full_response += token
                            except: pass
                        yield line + "\n"
        
        # Save assistant msg to history
        if full_response.strip():
            db.add_message(conv_id, "assistant", full_response)
            db.add_inference_log(
                model_id=body.model,
                prompt_tokens=len(str(body.messages[-1]["content"])) // 4,
                completion_tokens=len(full_response) // 4,
                latency_ms=100.0, tps=0.0
            )

    response = StreamingResponse(stream_generator(effective_conv_id), media_type="text/event-stream")
    response.headers["X-Conversation-ID"] = effective_conv_id
    return response



def run_server(host: str = "127.0.0.1", port: int = 11435):
    """Run the daemon server"""
    import uvicorn
    print(f"{Colors.GREEN}{Colors.BOLD}Oprel Daemon v0.3.3{Colors.RESET}")
    print(f"  Listening on: {Colors.CYAN}http://{host}:{port}{Colors.RESET}")
    print(f"  Press {Colors.YELLOW}Ctrl+C{Colors.RESET} to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    run_server()
