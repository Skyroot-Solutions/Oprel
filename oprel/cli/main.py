"""
Command-line interface for Oprel SDK
"""

import sys
import argparse
from pathlib import Path
import uuid

from oprel import Model, __version__
from oprel.downloader.cache import (
    list_cached_models,
    get_cache_size,
    clear_cache,
    delete_model,
)
from oprel.telemetry.hardware import get_hardware_info
from oprel.utils.logging import set_log_level, get_logger

logger = get_logger(__name__)


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat mode"""
    from ..utils.chat_templates import format_chat_prompt
    print(f"Oprel Chat v{__version__}")
    print(f"Model: {args.model}")
    print("Type 'exit', 'quit' or Ctrl+D to end.")
    print("Type '/reset' to clear conversation history.\n")

    # Determine server mode
    use_server = not getattr(args, 'no_server', False)
    
    # Generate conversation ID for tracking history on server
    conversation_id = str(uuid.uuid4())
    if use_server:
        print(f"Conversation ID: {conversation_id}")
        
    system_prompt = getattr(args, 'system', None)
    if system_prompt:
        print(f"System: {system_prompt}")

    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
            use_server=use_server,
            allow_low_quality=getattr(args, 'allow_low_quality', False),
        ) as model:
            print("\nModel loaded. Ready to chat!\n")

            # Interactive loop across platforms
            import sys

            # Local conversation history for non-server (direct) mode
            conversation_history = []

            while True:
                try:
                    # Handle input properly (Python input() uses readline if available)
                    try:
                        prompt = input(">>> ")
                    except EOFError:
                        print("\nExiting...")
                        break

                    if prompt.lower() in ["exit", "quit"]:
                        break

                    if prompt.strip() == "/reset":
                        if use_server:
                            # Generate new conversation ID to reset history
                            conversation_id = str(uuid.uuid4())
                            print(f"Conversation reset. New ID: {conversation_id}\n")
                        else:
                            conversation_history = []
                            print("Conversation history cleared (local mode).\n")
                        continue

                    if not prompt.strip():
                        continue

                    # Prepare formatted prompt using chat templates for direct mode
                    if use_server:
                        formatted_prompt = prompt
                    else:
                        formatted_prompt = format_chat_prompt(
                            model_id=args.model,
                            user_message=prompt,
                            system_prompt=system_prompt,
                            conversation_history=conversation_history,
                        )

                    # Server mode or direct mode both support streaming
                    if args.stream:
                        assistant_accum = ""
                        print("AI: ", end="", flush=True)
                        for token in model.generate(
                            formatted_prompt,
                            stream=True,
                            conversation_id=conversation_id if use_server else None,
                            system_prompt=system_prompt if use_server else None,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                        ):
                            print(token, end="", flush=True)
                            assistant_accum += token
                        print()

                        # Update conversation history in local mode
                        if not use_server:
                            conversation_history.append({"role": "user", "content": prompt})
                            conversation_history.append({"role": "assistant", "content": assistant_accum})
                            if len(conversation_history) > 40:
                                conversation_history = conversation_history[-40:]

                    else:
                        response = model.generate(
                            formatted_prompt,
                            conversation_id=conversation_id if use_server else None,
                            system_prompt=system_prompt if use_server else None,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                        )
                        print(response)
                        print()

                        if not use_server:
                            conversation_history.append({"role": "user", "content": prompt})
                            conversation_history.append({"role": "assistant", "content": response})
                            if len(conversation_history) > 40:
                                conversation_history = conversation_history[-40:]

                    # Clear one-time system prompt after first use
                    system_prompt = None

                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'exit' to quit.")
                    continue

        return 0

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return 1


def cmd_run_embed(args):
    """Interactive embedding mode"""
    from oprel.client_api import Client
    import json
    
    print(f"Oprel Embed v{__version__}")
    print(f"Model: {args.model}")
    print("Type 'exit' or 'quit' to end.\n")
    
    client = Client()
    
    try:
        while True:
            try:
                text = input(">>> ")
            except EOFError:
                print("\nExiting...")
                break
            
            if text.lower() in ["exit", "quit"]:
                break
            
            if not text.strip():
                continue
            
            try:
                # Generate embedding
                embeddings = client.embed(
                    texts=[text],
                    model=args.model,
                    normalize=True
                )
                
                # Ensure it's a list
                if isinstance(embeddings[0], float):
                    embeddings = [embeddings]
                
                embedding = embeddings[0]
                
                # Display results
                print(f"✓ Dimensions: {len(embedding)}")
                print(f"  Vector (first 5): {embedding[:5]}")
                print(f"  Magnitude: {sum(x*x for x in embedding)**0.5:.4f}\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Single-shot text generation"""
    from ..utils.chat_templates import format_chat_prompt
    
    # Determine server mode
    use_server = not getattr(args, 'no_server', False)

    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
            use_server=use_server,
            allow_low_quality=getattr(args, 'allow_low_quality', False),
        ) as model:
            # Format prompt using chat templates
            formatted_prompt = format_chat_prompt(
                model_id=args.model,
                user_message=args.prompt,
                system_prompt=None,
                conversation_history=[]
            )
            
            response = model.generate(
                formatted_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=args.stream,
            )

            if args.stream:
                for token in response:
                    print(token, end="", flush=True)
                print()
            else:
                print(response)

        return 0

    except Exception as e:
        logger.error(f"Generate error: {e}")
        return 1
        logger.error(f"Generation error: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information"""
    hw_info = get_hardware_info()

    print("System Information:")
    print(f"  OS: {hw_info['os']} ({hw_info['arch']})")
    print(f"  CPU Cores: {hw_info['cpu_count']} physical, {hw_info['cpu_threads']} threads")
    print(
        f"  RAM: {hw_info['ram_total_gb']:.1f} GB total, {hw_info['ram_available_gb']:.1f} GB available"
    )

    if "gpu_type" in hw_info:
        print(f"  GPU: {hw_info['gpu_name']} ({hw_info['gpu_type'].upper()})")
        print(f"  VRAM: {hw_info['vram_total_gb']:.1f} GB")
    else:
        print("  GPU: None detected")

    return 0


def cmd_cache_list(args: argparse.Namespace) -> int:
    """List cached models (GGUF and ComfyUI)"""
    from pathlib import Path
    from oprel.downloader.comfyui_installer import list_installed_checkpoints
    from datetime import datetime
    
    # GGUF models
    gguf_models = list_cached_models()
    total_size = get_cache_size()
    
    # ComfyUI models
    comfyui_dir = Path.home() / ".cache" / "oprel" / "models" / "comfyui" / "models" / "checkpoints"
    comfyui_models = []
    if comfyui_dir.exists():
        for ckpt in comfyui_dir.glob("*.safetensors"):
            stat = ckpt.stat()
            comfyui_models.append({
                "name": ckpt.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime)
            })
    
    if not gguf_models and not comfyui_models:
        print("No models in cache.")
        return 0
    
    print(f"📦 Cached Models (Total: {total_size:.1f} MB)\n")
    
    if gguf_models:
        print(f"🔤 GGUF Models ({len(gguf_models)}):\n")
        for model in gguf_models:
            print(f"  {model['name']}")
            print(f"    Size: {model['size_mb']:.1f} MB")
            print(f"    Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    if comfyui_models:
        print(f"🎨 ComfyUI Models ({len(comfyui_models)}):\n")
        for model in comfyui_models:
            print(f"  {model['name']}")
            print(f"    Size: {model['size_mb']:.1f} MB")
            print(f"    Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    print(f"💡 To delete a model: oprel cache delete <model_name>")
    return 0


def cmd_cache_clear(args: argparse.Namespace) -> int:
    """Clear model cache"""
    if not args.yes:
        response = input("This will delete all cached models. Continue? [y/N] ")
        if response.lower() != "y":
            print("Cancelled.")
            return 0

    try:
        count = clear_cache(confirm=True)
        print(f"Cleared cache ({count} files deleted)")
        return 0
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return 1


def cmd_cache_delete(args: argparse.Namespace) -> int:
    """Delete specific model from cache (GGUF or ComfyUI models)"""
    from oprel.downloader.comfyui_installer import list_installed_checkpoints
    from pathlib import Path
    
    model_name = args.model_name
    
    # Try GGUF models first
    if delete_model(model_name):
        print(f"✓ Deleted GGUF model: {model_name}")
        return 0
    
    # Try ComfyUI checkpoints
    comfyui_dir = Path.home() / ".cache" / "oprel" / "models" / "comfyui" / "models" / "checkpoints"
    if comfyui_dir.exists():
        checkpoints = list(comfyui_dir.glob("*.safetensors"))
        for ckpt in checkpoints:
            if model_name in ckpt.name or ckpt.name == model_name:
                try:
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    ckpt.unlink()
                    print(f"✓ Deleted ComfyUI model: {ckpt.name} ({size_mb:.1f}MB)")
                    return 0
                except Exception as e:
                    print(f"❌ Error deleting {ckpt.name}: {e}")
                    return 1
    
    print(f"❌ Model not found: {model_name}")
    print("\nTip: Use 'oprel cache list' to see all cached models")
    return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the oprel daemon server"""
    try:
        import socket
        import platform
        import psutil
        from oprel.server.daemon import run_server
        
        # Check if port is already in use
        port = args.port
        host = args.host
        
        # Find process using the port
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    pid = conn.pid
                    if pid:
                        try:
                            process = psutil.Process(pid)
                            process_name = process.name()
                            print(f"Port {port} is already in use by process {pid} ({process_name})")
                            print(f"Stopping previous server...")
                            
                            # Kill the process
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                                print(f"Previous server stopped successfully")
                            except psutil.TimeoutExpired:
                                print(f"Process didn't stop, forcing...")
                                process.kill()
                                process.wait()
                                print(f"Previous server killed")
                            
                            # Give the port time to be released
                            import time
                            time.sleep(1)
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            print(f"Warning: Could not stop process {pid}: {e}")
                    break
        except Exception as e:
            # If we can't check/stop, just try to start anyway
            logger.debug(f"Could not check for existing server: {e}")
        
        print(f"Starting Oprel daemon server...")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print()
        
        run_server(host=host, port=port)
        return 0
        
    except ImportError as e:
        logger.error(
            "Server dependencies not installed. "
            "Install with: pip install oprel[server]"
        )
        logger.error(f"Details: {e}")
        return 1
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Fast inference using direct mode (auto-cleanup after use)"""
    import sys
    from ..utils.chat_templates import format_chat_prompt
    
    try:
        # Use DIRECT mode to ensure cleanup after use
        # This prevents memory leaks by stopping the process when done
        model = Model(
            args.model,
            quantization=args.quantization,
            use_server=False,  # Changed: Direct mode for auto-cleanup
            allow_low_quality=getattr(args, 'allow_low_quality', False),
        )
        
        # Load model
        model.load()
        
        # Flush stderr to ensure all log messages are written before output
        sys.stderr.flush()
        
        # Add separator between logs and response
        print()
        
        # If no prompt provided, enter interactive mode
        if args.prompt is None:
            result = _run_interactive(model, args)
            # Cleanup after interactive session
            model.unload()
            return result
        
        # One-shot mode: generate single response with proper chat formatting
        formatted_prompt = format_chat_prompt(
            model_id=args.model,
            user_message=args.prompt,
            system_prompt=None,
            conversation_history=[]
        )
        
        if args.stream:
            for token in model.generate(
                formatted_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
            ):
                print(token, end="", flush=True)
            print()
        else:
            response = model.generate(
                formatted_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(response)
        
        # CRITICAL: Unload model to free GPU/RAM
        # This stops the backend process and releases all memory
        logger.debug("Cleaning up: Unloading model to free GPU/RAM...")
        model.unload()
        logger.debug("✓ Memory freed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Run error: {e}")
        # Ensure cleanup even on error
        try:
            if 'model' in locals():
                model.unload()
        except:
            pass
        return 1


def _run_interactive(model: Model, args: argparse.Namespace) -> int:
    """Interactive chat mode for oprel run (like ollama run)"""
    import sys
    import time
    import threading
    from ..utils.chat_templates import format_chat_prompt
    
    print(f">>> Model loaded: {args.model}")
    print(">>> Send a message (/? for help)")
    print(">>> Auto-cleanup: Model will unload after 15 minutes of inactivity")
    print()
    
    # Generate conversation ID for tracking history on server
    conversation_id = str(uuid.uuid4())
    system_prompt = getattr(args, 'system', None)

    # Maintain a local conversation history (for direct interactive mode)
    # Each entry: {'role': 'user'|'assistant', 'content': '...'}
    conversation_history = []
    
    # Idle timeout tracking (15 minutes)
    IDLE_TIMEOUT_SECONDS = 15 * 60
    last_activity_time = time.time()
    idle_check_lock = threading.Lock()
    should_exit = threading.Event()
    
    def idle_monitor():
        """Background thread to monitor idle time and trigger cleanup"""
        nonlocal last_activity_time
        while not should_exit.is_set():
            time.sleep(60)  # Check every minute
            
            with idle_check_lock:
                idle_time = time.time() - last_activity_time
                
            if idle_time > IDLE_TIMEOUT_SECONDS:
                print("\n\n⏱️  Idle timeout reached (15 minutes)")
                print("Unloading model to free memory...")
                print("Bye!\n")
                should_exit.set()
                # Force exit the input loop
                import os
                if sys.platform == "win32":
                    import msvcrt
                    # Send Ctrl+C to interrupt input() on Windows
                    os.kill(os.getpid(), 2)  # SIGINT
                return
    
    # Start idle monitor thread
    monitor_thread = threading.Thread(target=idle_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        while not should_exit.is_set():
            try:
                # Get user input
                try:
                    user_input = input(">>> ")
                    
                    # Update activity timestamp
                    with idle_check_lock:
                        last_activity_time = time.time()
                        
                except EOFError:
                    print("\nBye!")
                    break
                
                # Check if we should exit due to idle timeout
                if should_exit.is_set():
                    break
                
                # Handle special commands
                if user_input.strip() in ["/exit", "/bye", "/quit"]:
                    print("Bye!")
                    break
                
                if user_input.strip() == "/?":
                    print("Available commands:")
                    print("  /exit, /bye, /quit - Exit the chat")
                    print("  /reset            - Clear conversation history")
                    print("  /?                - Show this help")
                    print()
                    continue
                
                if user_input.strip() == "/reset":
                    conversation_id = str(uuid.uuid4())
                    conversation_history.clear()
                    print("Conversation history cleared.\n")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Generate response - use chat templates for consistent formatting
                try:
                    # Format prompt using chat templates so models receive proper roles
                    formatted_prompt = format_chat_prompt(
                        model_id=args.model,
                        user_message=user_input,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                    )

                    if args.stream:
                        # Stream tokens and accumulate assistant response
                        assistant_accum = ""
                        print("AI: ", end="", flush=True)
                        for token in model.generate(
                            formatted_prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            stream=True,
                            conversation_id=conversation_id,
                            system_prompt=None,
                        ):
                            print(token, end="", flush=True)
                            assistant_accum += token
                        print("\n")
                        # Append user and assistant to local history
                        conversation_history.append({"role": "user", "content": user_input})
                        conversation_history.append({"role": "assistant", "content": assistant_accum})
                        # Truncate history to recent 20 turns to limit prompt size
                        if len(conversation_history) > 40:
                            conversation_history = conversation_history[-40:]
                        system_prompt = None

                    else:
                        response = model.generate(
                            formatted_prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            conversation_id=conversation_id,
                            system_prompt=None,
                        )
                        # Append to history and display
                        conversation_history.append({"role": "user", "content": user_input})
                        conversation_history.append({"role": "assistant", "content": response})
                        if len(conversation_history) > 40:
                            conversation_history = conversation_history[-40:]
                        system_prompt = None
                        print(response)
                        print()
                    
                    # Update activity timestamp after successful generation
                    with idle_check_lock:
                        last_activity_time = time.time()
                
                except KeyboardInterrupt:
                    print("\n")
                    continue
                except Exception as e:
                    print(f"\nError: {e}\n")
                    continue
            
            except KeyboardInterrupt:
                print("\n\nUse /exit to quit\n")
                continue
    
    finally:
        # Signal monitor thread to stop
        should_exit.set()
        
        # Explicitly unload the model and force-stop backend to free GPU memory
        try:
            print("\nCleaning up...")
            
            # Force-stop backend process (critical for GPU memory cleanup)
            if hasattr(model, '_process') and model._process:
                print("Stopping backend process...")
                model._process.stop(force=False)
                model._process = None
            
            # Stop monitor if exists
            if hasattr(model, '_monitor') and model._monitor:
                model._monitor.stop()
                model._monitor = None
            
            # Cleanup PyTorch backend if used
            if hasattr(model, '_pytorch_backend') and model._pytorch_backend:
                model._pytorch_backend.unload()
                model._pytorch_backend = None
            
            # Mark as unloaded
            model._loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("✓ GPU memory cleared")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Could not clear CUDA cache: {e}")
            
            print("✓ Cleanup complete")
                
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
    
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    import requests
    
    server_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.get(f"{server_url}/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        
        if not models:
            print("No models currently loaded in server.")
            return 0
        
        print(f"Models loaded in server ({len(models)}):\n")
        for model in models:
            status = "loaded" if model.get("loaded") else "unloaded"
            quant = model.get("quantization") or "auto"
            print(f"  {model['model_id']}")
            print(f"    Backend: {model.get('backend', 'llama.cpp')}")
            print(f"    Quantization: {quant}")
            print(f"    Status: {status}")
            print()
        
        return 0
        
    except requests.ConnectionError:
        print(f"Cannot connect to server at {server_url}")
        print("Start server with: oprel serve")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the oprel daemon server and all backend processes"""
    import requests
    import psutil
    import time
    
    server_url = f"http://{args.host}:{args.port}"
    port = args.port
    stopped_server = False
    stopped_backends = False
    
    # Step 1: Try graceful shutdown via API (this will unload all models and exit)
    try:
        print("Requesting graceful shutdown...")
        response = requests.post(f"{server_url}/shutdown", timeout=5)
        if response.status_code == 200:
            print("  ✓ Shutdown signal sent")
            # Wait a moment for the server to clean up
            time.sleep(2)
            
            # Check if it actually stopped
            if not _is_port_in_use(port):
                print("  ✓ Daemon stopped gracefully")
                stopped_server = True
                # If graceful shutdown worked, backend processes should be cleaned up too
                stopped_backends = True
                print("\n✓ All Oprel processes stopped successfully")
                return 0
    except Exception as e:
        logger.debug(f"Graceful shutdown failed: {e}")
    
    # Step 2: If graceful shutdown failed, unload models manually
    try:
        response = requests.get(f"{server_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            loaded_models = [m for m in models if m.get("loaded", False)]
            if loaded_models:
                print(f"Unloading {len(loaded_models)} model(s)...")
                for model in loaded_models:
                    model_id = model["model_id"]
                    import urllib.parse
                    encoded_id = urllib.parse.quote(model_id, safe="")
                    try:
                        unload_response = requests.delete(f"{server_url}/unload/{encoded_id}", timeout=30)
                        if unload_response.status_code == 200:
                            print(f"  ✓ Unloaded: {model_id}")
                        else:
                            print(f"  ✗ Failed to unload: {model_id}")
                    except Exception as e:
                        logger.debug(f"Failed to unload {model_id}: {e}")
    except Exception as e:
        logger.debug(f"Could not communicate with server to unload models: {e}")
    
    # Step 3: Find and kill the daemon server process
    try:
        server_pid = None
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                server_pid = conn.pid
                break
        
        if server_pid:
            try:
                process = psutil.Process(server_pid)
                process_name = process.name()
                print(f"Stopping Oprel daemon (PID: {server_pid})...")
                
                # Terminate gracefully
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print(f"  ✓ Daemon stopped")
                    stopped_server = True
                except psutil.TimeoutExpired:
                    print(f"  Process didn't stop gracefully, forcing...")
                    process.kill()
                    process.wait()
                    print(f"  ✓ Daemon killed")
                    stopped_server = True
                    
                # Give the port time to be released
                time.sleep(0.5)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Could not stop daemon process {server_pid}: {e}")
        else:
            if not stopped_server:
                print("Oprel daemon is not running")
            
    except Exception as e:
        logger.debug(f"Error finding/stopping daemon: {e}")
    
    # Step 4: Kill any orphaned backend processes
    # These might be left over if the daemon crashed or was forcefully killed
    try:
        killed_backends = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_info = proc.info
                proc_name = proc_info.get('name', '').lower()
                cmdline = proc_info.get('cmdline', [])
                
                # Look for oprel-backend processes (our renamed binaries)
                # or fallback to llama-server if running older version
                is_backend = False
                
                # Primary check: look for our branded "oprel-backend" process
                if proc_name and 'oprel-backend' in proc_name:
                    is_backend = True
                # Fallback: look for llama-server from our binary directory
                elif proc_name and any(x in proc_name for x in ['llama-server', 'llama_server', 'llama-cpp']):
                    # Additional check: only kill if it's from oprel's binary directory
                    # This avoids killing user's own llama-server instances
                    if cmdline:
                        cmdline_str = ' '.join(cmdline).lower()
                        from oprel.core.config import Config
                        config = Config()
                        binary_dir_str = str(config.binary_dir).lower()
                        if binary_dir_str in cmdline_str:
                            is_backend = True
                
                if is_backend:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                        killed_backends.append((proc_info['pid'], proc_info['name']))
                    except psutil.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        killed_backends.append((proc_info['pid'], proc_info['name']))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if killed_backends:
            print(f"Stopped {len(killed_backends)} backend process(es):")
            for pid, name in killed_backends:
                print(f"  ✓ {name} (PID: {pid})")
            stopped_backends = True
            
    except Exception as e:
        logger.debug(f"Error cleaning up backend processes: {e}")
    
    # Summary
    if stopped_server or stopped_backends:
        print("\n✓ All Oprel processes stopped successfully")
        return 0
    else:
        print("\nNo Oprel processes were running")
        return 0


def _is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    try:
        import psutil
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return True
        return False
    except:
        return False


def cmd_list_models(args: argparse.Namespace) -> int:
    """List all available model aliases"""
    from oprel.downloader.aliases import (
        list_models_by_category,
        get_categories,
        get_category_info,
        MODEL_ALIASES
    )
    
    # Get category filter if provided
    category_filter = getattr(args, 'category', None)
    
    try:
        # Get models (filtered or all)
        categorized_models = list_models_by_category(category_filter)
        
        # Count total models
        total_count = sum(len(models) for models in categorized_models.values())
        
        # Print header
        if category_filter:
            cat_info = get_category_info(category_filter)
            print(f"\n{cat_info['icon']} {cat_info['name']} Models ({total_count} total)")
            print(f"   {cat_info['description']}\n")
        else:
            print(f"\nAvailable Models ({total_count} total across {len(categorized_models)} categories)\n")
        
        # Display models by category
        for category, models in categorized_models.items():
            cat_info = get_category_info(category)
            
            if not category_filter:
                print(f"{cat_info['icon']} {cat_info['name']} ({len(models)} models)")
                print(f"   {cat_info['description']}")
            
            # Sort models by alias
            for alias in sorted(models.keys()):
                repo_id = models[alias]
                source = repo_id.split("/")[0]
                model_name = repo_id.split("/")[-1]
                print(f"   {alias:25} → {source}/{model_name[:40]}")
            print()
        
        # Show available categories if listing all
        if not category_filter:
            print("Filter by category:")
            print("   oprel list-models --category <category>")
            print(f"   Categories: {', '.join(get_categories())}")
        
        print("\nUsage: oprel run <alias> \"your prompt\"")
        return 0
        
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable categories: {', '.join(get_categories())}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for model aliases"""
    from oprel.downloader.aliases import search_aliases, MODEL_ALIASES
    
    matches = search_aliases(args.query)
    
    if not matches:
        print(f"No models found matching '{args.query}'")
        return 1
    
    print(f"Models matching '{args.query}':\n")
    for alias in matches:
        gguf_id = MODEL_ALIASES.get(alias, "")
        print(f"  {alias:20} -> {gguf_id}")
    
    print(f"\nUsage: oprel run {matches[0]} \"your prompt\"")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Download a model (text or image) without running it"""
    from oprel.downloader.comfyui_installer import download_checkpoint
    from oprel.downloader.hub import download_model as download_gguf
    from oprel.downloader.aliases import MODEL_ALIASES, resolve_model_id
    
    model_id = args.model
    logger.info(f"Pulling model: {model_id}")
    
    # Check if it's an alias or known repo
    repo_id = None
    filename = None
    
    if model_id in MODEL_ALIASES:
        spec = MODEL_ALIASES[model_id]
        if ":" in spec:
            repo_id, filename = spec.split(":", 1)
        else:
            # Assume text model (GGUF)
            print(f"Downloading text model: {spec}...")
            try:
                download_gguf(spec)
                print(f"✓ Successfully pulled {model_id}")
                return 0
            except Exception as e:
                print(f"Error downloading text model: {e}")
                return 1
    else:
        # Check reverse lookup for image models
        for alias, spec in MODEL_ALIASES.items():
            if ":" in spec:
                r, f = spec.split(":", 1)
                # Check match on repo part
                if model_id == r:
                    repo_id = r
                    filename = f
                    break
        
        # If still not found, check if it looks like a repo ID
        if not repo_id and "/" in model_id:
            # Assume image/video model download attempt if explicitly using pull
            # OR could be a GGUF text model.
            # Let's try GGUF first as default behavior for `oprel` usually implies text models.
            # BUT if it fails, maybe try checkpoint download?
            # Actually, `oprel pull` for arbitrary image models is valuable.
            pass

    if repo_id and filename:
        print(f"Downloading image/video model: {repo_id}/{filename}...")
        try:
            download_checkpoint(repo_id, filename)
            print(f"✓ Successfully pulled {filename}")
            return 0
        except Exception as e:
            print(f"Error downloading image model: {e}")
            return 1
    
    # Fallback: Treat as GGUF text model download
    print(f"Downloading model: {model_id}...")
    try:
        download_gguf(model_id)
        return 0
    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1


def cmd_vision(args: argparse.Namespace) -> int:
    """
    Vision command: Ask questions about images using VLM models.
    Supports: qwen-vl, llava, minicpm-v, moondream, etc.
    """
    from oprel.core.model import Model
    from oprel.runtime.backends.multimodal import format_vision_prompt, get_vision_model_config
    
    try:
        # Validate images exist
        for img_path in args.images:
            if not Path(img_path).exists():
                print(f"Error: Image not found: {img_path}")
                return 1
        
        # Get vision model config
        config = get_vision_model_config(args.model)
        logger.info(f"Vision model architecture: {config['architecture']}")
        
        # Check image count
        if len(args.images) > config['max_images']:
            print(f"Warning: {args.model} supports max {config['max_images']} images, using first {config['max_images']}")
            args.images = args.images[:config['max_images']]
        
        # Load vision model (use direct mode to avoid daemon complexity)
        print(f"Loading vision model: {args.model}")
        model = Model(args.model, use_server=False)  # Direct mode for vision
        model.load()
        
        # Format prompt with images
        vision_data = format_vision_prompt(
            text_prompt=args.prompt,
            image_paths=args.images,
            model_architecture=config['architecture']
        )
        
        print(f"\nAnalyzing {vision_data['num_images']} image(s)...\n")
        
        # Generate response with image data
        response = model.generate(
            vision_data['prompt'],
            max_tokens=args.max_tokens or 512,
            temperature=args.temperature or 0.7,
            stream=not args.no_stream,
            images=vision_data['images'],  # Pass base64-encoded images to backend
        )
        
        if args.no_stream:
            print(response)
        else:
            for chunk in response:
                print(chunk, end='', flush=True)
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Vision command failed: {e}", exc_info=True)
        return 1


def cmd_embed(args):
    """Generate text embeddings"""
    from oprel.client_api import Client
    import json
    
    client = Client()
    
    # Collect texts to embed
    texts = []
    
    if args.batch:
        # Read from file
        batch_file = Path(args.batch)
        if not batch_file.exists():
            logger.error(f"Batch file not found: {args.batch}")
            return 1
        
        with open(batch_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(texts)} texts from {args.batch}")
    elif args.text:
        # Single text from positional argument
        texts = [args.text]
    else:
        logger.error("Either provide text as argument or use --batch flag")
        print("Usage:")
        print('  oprel embed <model> "your text here"')
        print("  oprel embed <model> --batch file.txt")
        return 1
    
    if not texts:
        logger.error("No texts to embed")
        return 1
    
    try:
        # Generate embeddings
        logger.info(f"Generating embeddings with model: {args.model}")
        
        embeddings = client.embed(
            texts=texts,
            model=args.model,
            normalize=not args.no_normalize
        )
        
        # Ensure embeddings is a list of lists
        if isinstance(embeddings[0], float):
            # Single embedding was returned, wrap it
            embeddings = [embeddings]
        
        # Output results
        if args.output:
            # Save to file
            output_path = Path(args.output)
            output_data = {
                "model": args.model,
                "embeddings": embeddings,
                "texts": texts if not args.no_texts else None,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✓ Saved {len(embeddings)} embeddings to {args.output}")
            print(f"Dimensions: {len(embeddings[0])}")
            print(f"Count: {len(embeddings)}")
        else:
            # Print to stdout
            if args.format == "json":
                result = {
                    "model": args.model,
                    "embeddings": embeddings,
                    "dimensions": len(embeddings[0]) if embeddings else 0
                }
                print(json.dumps(result, indent=2))
            elif args.format == "jsonl":
                for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                    line = {"text": text, "embedding": embedding}
                    print(json.dumps(line))
            else:
                # Simple format
                for i, embedding in enumerate(embeddings):
                    if len(texts) > 1:
                        print(f"\n[{i}] {texts[i][:50]}...")
                    print(f"Dimensions: {len(embedding)}")
                    print(f"Vector (first 5): {embedding[:5]}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return 1



def cmd_gen_image(args: argparse.Namespace) -> int:
    """
    Image generation command: Create images from text prompts.
    
    Uses the new snapshot_download system to properly handle
    diffusers pipelines and merged checkpoints.
    """
    from oprel.runtime.backends.comfyui import ComfyUIClient, ComfyUIImageGenerator
    from oprel.downloader.comfyui_installer import (
        ensure_comfyui_ready,
        download_checkpoint,
        list_installed_checkpoints,
        get_comfyui_dir
    )
    from pathlib import Path
    import time
    
    try:
        # 1. Ensure ComfyUI is installed
        comfyui_dir = get_comfyui_dir()
        if not comfyui_dir.exists():
            print("❌ Image generation not set up!")
            print()
            print("Please run: oprel setup image")
            print()
            print("This will install ComfyUI + dependencies")
            return 1
        
        # 2. Parse model name
        model_id = args.model
        logger.info(f"Image generation with model: {model_id}")
        
        # 3. Check if model is already installed
        installed_models = list_installed_checkpoints()
        model_found = None
        
        # Try exact match first
        for model in installed_models:
            if model['name'] == model_id or model_id in model['name'].lower():
                model_found = model
                logger.info(f"Found installed model: {model['name']} ({model['type']})")
                break
        
        # If not found, download it
        if not model_found:
            print(f"📥 Downloading model: {model_id}")
            print("This may take a while...")
            
            try:
                # Use new snapshot_download system
                downloaded_path = download_checkpoint(
                    repo_id=model_id,
                    filename=None  # Let it auto-detect
                )
                
                print(f"✓ Downloaded to: {downloaded_path}")
                
                # Re-scan for the model
                installed_models = list_installed_checkpoints()
                for model in installed_models:
                    if downloaded_path.name in model['name']:
                        model_found = model
                        break
                
            except Exception as e:
                print(f"❌ Download failed: {e}")
                logger.error(f"Download error: {e}", exc_info=True)
                return 1
        
        if not model_found:
            print(f"❌ Model '{model_id}' not available")
            print()
            print("Available models:")
            for model in installed_models[:10]:
                print(f"  - {model['name']} ({model['type']})")
            return 1
        
        # 4. Start ComfyUI if not running
        client = ComfyUIClient()
        
        if not client.is_available():
            print("🚀 Starting ComfyUI server...")
            
            # Use ComfyUIBackend to start the server
            from oprel.runtime.backends.comfyui_process import ComfyUIBackend
            backend = ComfyUIBackend(None, None)
            
            if not backend.start():
                print("❌ Failed to start ComfyUI")
                print()
                print("Try starting manually:")
                print(f"  cd {comfyui_dir}")
                print("  python main.py")
                return 1
            
            # Wait for server to be ready
            print("⏳ Waiting for server...")
            time.sleep(3)
            
            # Re-check connection
            if not client.is_available():
                print("❌ ComfyUI started but not responding")
                return 1
        
        # 5. Generate image
        generator = ComfyUIImageGenerator(client)
        
        # Get model name for generator
        model_name = model_found['name']
        print(f"🎨 Generating with {model_name}...")
        print(f"   Prompt: {args.prompt}")
        
        start_time = time.time()
        
        try:
            image_bytes = generator.generate_txt2img(
                prompt=args.prompt,
                checkpoint=model_name,  # REQUIRED
                negative_prompt=args.negative or "",
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.guidance or 7.5,
            )
        except ValueError as e:
            # Model format error
            print(f"❌ {e}")
            return 1
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            logger.error(f"Generation error: {e}", exc_info=True)
            return 1
        
        elapsed = time.time() - start_time
        
        # 7. Save image
        if args.output:
            output_path = Path(args.output)
        else:
            # Auto-generate filename
            import re
            safe_prompt = re.sub(r'[^\w\s-]', '', args.prompt)[:30]
            safe_prompt = re.sub(r'[-\s]+', '-', safe_prompt)
            timestamp = int(time.time())
            output_path = Path(f"oprel_{safe_prompt}_{timestamp}.png")
        
        output_path.write_bytes(image_bytes)
        
        print(f"✓ Generated in {elapsed:.1f}s")
        print(f"✓ Saved to: {output_path.absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠ Cancelled")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Image generation failed: {e}", exc_info=True)
        return 1


def cmd_gen_video(args: argparse.Namespace) -> int:
    """Video generation command - Coming Soon"""
    print("🎬 Video Generation - Coming Soon!")
    print()
    print("Video generation will be available in a future release.")
    print()
    print("Available models:")
    print("  • animatediff-motion - AnimateDiff motion module")
    print("  • svd - Stable Video Diffusion")
    print("  • svd-xt - Stable Video Diffusion XT (longer videos)")
    print()
    print("Stay tuned for:")
    print("  ✨ Text-to-video generation")
    print("  ✨ Image-to-video animation")
    print("  ✨ Custom video workflows")
    print()
    return 0




def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="oprel",
        description="Oprel SDK - Local-first AI runtime",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"oprel {__version__}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("model", help="Model ID (e.g., TheBloke/Llama-2-7B-GGUF)")
    chat_parser.add_argument("--quantization", help="Quantization level (Q4_K_M, Q8_0, etc.)")
    chat_parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    chat_parser.add_argument("--stream", action="store_true", default=True, help="Stream responses")
    chat_parser.add_argument("--system", help="System prompt")
    chat_parser.add_argument(
        "--no-server",
        action="store_true",
        help="Force direct mode (don't use persistent server)"
    )
    chat_parser.add_argument("--allow-low-quality", action="store_true", help="Allow low-quality quantizations like Q2_K")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("model", help="Model ID")
    gen_parser.add_argument("prompt", help="Input prompt")
    gen_parser.add_argument("--quantization", help="Quantization level")
    gen_parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    gen_parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--stream", action="store_true", help="Stream response")
    gen_parser.add_argument(
        "--no-server",
        action="store_true",
        help="Force direct mode (don't use persistent server)"
    )

    # Serve command (NEW)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the oprel daemon server for persistent model caching"
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port to listen on (default: 11434)"
    )

    # Run command - Interactive modes
    run_parser = subparsers.add_parser(
        "run",
        help="Interactive modes (text generation, embeddings, etc.)"
    )
    run_subparsers = run_parser.add_subparsers(dest="run_command", help="Mode to run")
    
    # run text: Text generation (default behavior)
    run_text_parser = run_subparsers.add_parser(
        "text",
        help="Interactive text generation"
    )
    run_text_parser.add_argument("model", help="Model ID")
    run_text_parser.add_argument("prompt", nargs="?", default=None, help="Input prompt (omit for interactive mode)")
    
    # run embed: Interactive embedding
    run_embed_parser = run_subparsers.add_parser(
        "embed",
        help="Interactive embedding mode"
    )
    run_embed_parser.add_argument(
        "model",
        help="Embedding model (e.g., nomic-embed-text, bge-m3)"
    )
    
    # Models command (NEW) - list loaded models in server
    models_parser = subparsers.add_parser(
        "models",
        help="List models loaded in the server"
    )
    models_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    models_parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Server port (default: 11434)"
    )

    # Stop command (NEW) - stop server
    stop_parser = subparsers.add_parser(
        "stop",
        help="Request server to unload all models"
    )
    stop_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    stop_parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Server port (default: 11434)"
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup additional features")
    setup_subparsers = setup_parser.add_subparsers(dest="setup_command")
    setup_subparsers.add_parser("image", help="Install ComfyUI + CUDA for image generation")

    # List-models command (NEW)
    list_models_parser = subparsers.add_parser("list-models", help="List all available model aliases")
    list_models_parser.add_argument(
        "--category",
        choices=["text-generation", "coding", "reasoning", "text-to-video", "text-to-image", "vision", "embeddings"],
        help="Filter models by category"
    )

    # Search command (NEW)
    search_parser = subparsers.add_parser("search", help="Search for models by name")
    search_parser.add_argument("query", help="Search term (e.g., 'llama', 'qwen')")

    # Pull command (NEW)
    pull_parser = subparsers.add_parser("pull", help="Download a model (text or image)")
    pull_parser.add_argument("model", help="Model ID, alias, or HF repo")


    # ===================================================================
    # MULTIMODAL COMMANDS
    # ===================================================================
    
    # Vision command: Image→Text (VLMs like qwen-vl, llava, moondream)
    vision_parser = subparsers.add_parser(
        "vision",
        help="Ask questions about images using vision models (qwen-vl, llava, etc.)"
    )
    vision_parser.add_argument("model", help="Vision model alias (e.g., qwen3-vl-7b, llava-v1.6)")
    vision_parser.add_argument("prompt", help="Question or instruction about the image(s)")
    vision_parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Path(s) to image file(s) to analyze"
    )
    vision_parser.add_argument("--max-tokens", type=int, help="Max tokens in response")
    vision_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    vision_parser.add_argument("--no-stream", action="store_true", help="Disable streaming")

    # Image generation command
    genimg_parser = subparsers.add_parser(
        "gen-image",
        help="Generate images from text prompts (ComfyUI)"
    )
    genimg_parser.add_argument(
        "model",
        help="Model name (e.g., flux-1-schnell, sdxl-turbo, sd-1.5) - REQUIRED"
    )
    genimg_parser.add_argument(
        "prompt",
        help="Text description of the image to generate"
    )
    genimg_parser.add_argument(
        "--negative",
        help="Negative prompt (what to avoid)"
    )
    genimg_parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    genimg_parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    genimg_parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of sampling steps (default: 28, turbo models use 4)"
    )
    genimg_parser.add_argument(
        "--guidance",
        type=float,
        help="Guidance scale/CFG (default: auto-detect based on model)"
    )
    genimg_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: auto-generated)"
    )

    # Video generation command: Text→Video (WAN, Mochi, CogVideoX, etc.)
    genvid_parser = subparsers.add_parser(
        "gen-video",
        help="Generate videos from text prompts (wan, mochi, cogvideox, etc.)"
    )
    genvid_parser.add_argument("model", help="Video model alias (e.g., wan2.2-5b, mochi-1-10b)")
    genvid_parser.add_argument("prompt", help="Text description of video to generate")
    genvid_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: auto-generated .mp4)"
    )
    genvid_parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames (default: 60)"
    )
    genvid_parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    genvid_parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width (default: 512)"
    )
    genvid_parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (default: 512)"
    )
    genvid_parser.add_argument(
        "--negative",
        help="Negative prompt (what to avoid)"
    )

    # Embed command: Generate text embeddings
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate text embeddings for semantic search and RAG"
    )
    embed_parser.add_argument(
        "model",
        help="Embedding model (e.g., nomic-embed-text, bge-m3, all-minilm-l6-v2)"
    )
    embed_parser.add_argument(
        "text",
        nargs="?",  # Optional for batch mode
        help="Text to embed (or use --batch for multiple texts)"
    )
    embed_parser.add_argument(
        "--batch",
        "-b",
        help="File containing texts to embed (one per line)"
    )
    embed_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file (default: print to stdout)"
    )
    embed_parser.add_argument(
        "--format",
        choices=["json", "jsonl", "simple"],
        default="simple",
        help="Output format (default: simple)"
    )
    embed_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize embeddings"
    )
    embed_parser.add_argument(
        "--no-texts",
        action="store_true",
        help="Don't include original texts in output file"
    )

    # Cache commands
    cache_parser = subparsers.add_parser("cache", help="Manage model cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")

    cache_subparsers.add_parser("list", help="List cached models")

    clear_parser = cache_subparsers.add_parser("clear", help="Clear all cached models")
    clear_parser.add_argument("--yes", action="store_true", help="Skip confirmation")

    delete_parser = cache_subparsers.add_parser("delete", help="Delete specific model")
    delete_parser.add_argument("model_name", help="Model filename to delete")

    # Parse arguments
    args = parser.parse_args()

    # Set log level
    if args.verbose:
        set_log_level("DEBUG")
    elif args.quiet:
        set_log_level("CRITICAL")

    # Handle run command special case for streaming
    if args.command == "run" and getattr(args, 'no_stream', False):
        args.stream = False

    # Route to command handlers
    if args.command == "chat":
        return cmd_chat(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "run":
        if args.run_command == "embed":
            return cmd_run_embed(args)
        elif args.run_command == "text" or args.run_command is None:
            # Default to text generation for backwards compatibility
            return cmd_run(args)
        else:
            run_parser.print_help()
            return 1
    elif args.command == "models":
        return cmd_models(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "setup":
        from oprel.cli.setup_commands import cmd_setup_image
        if args.setup_command == "image":
            return cmd_setup_image(args)
        else:
            setup_parser.print_help()
            return 1
    elif args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "pull":
        return cmd_pull(args)
    # Multimodal commands
    elif args.command == "vision":
        return cmd_vision(args)
    elif args.command == "gen-image":
        return cmd_gen_image(args)
    elif args.command == "gen-video":
        return cmd_gen_video(args)
    elif args.command == "embed":
        return cmd_embed(args)
    elif args.command == "cache":
        if args.cache_command == "list":
            return cmd_cache_list(args)
        elif args.cache_command == "clear":
            return cmd_cache_clear(args)
        elif args.cache_command == "delete":
            return cmd_cache_delete(args)
        else:
            cache_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
