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
        ) as model:
            print("\nModel loaded. Ready to chat!\n")
            
            # Interactive loop across platforms
            import sys
            
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
                            # Send a dummy request with reset flag or just generate new ID?
                            # Generating new ID is cleaner client-side approach but to reset server state for same ID:
                            # We can just call generate with reset_conversation=True and empty prompt?
                            # Or just make a new ID. New ID is easier.
                            conversation_id = str(uuid.uuid4())
                            print(f"Conversation reset. New ID: {conversation_id}\n")
                        else:
                            print("Reset available in server mode only.\n")
                        continue

                    if not prompt.strip():
                        continue

                    print("AI: ", end="", flush=True)
                    
                    # For first turn, send system prompt
                    # We send system prompt on every request? No, usually once.
                    # But server holds state. So send it once or let server handle.
                    # Our daemon updates system prompt if provided.
                    
                    if args.stream:
                        for token in model.generate(
                            prompt,
                            stream=True,
                            conversation_id=conversation_id,
                            system_prompt=system_prompt,
                        ):
                             # Clear system prompt after first use so it doesn't get re-appended
                             # (Server daemon handles this via history, but explicit is safer)
                            system_prompt = None 
                            print(token, end="", flush=True)
                        print()
                    else:
                        response = model.generate(
                            prompt,
                            conversation_id=conversation_id,
                            system_prompt=system_prompt,
                        )
                        system_prompt = None
                        print(response)

                    print()

                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'exit' to quit.")
                    continue

        return 0

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Single-shot text generation"""
    # Determine server mode
    use_server = not getattr(args, 'no_server', False)

    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
            use_server=use_server,
        ) as model:
            response = model.generate(
                args.prompt,
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
    """List cached models"""
    models = list_cached_models()
    total_size = get_cache_size()

    if not models:
        print("No models in cache.")
        return 0

    print(f"Cached Models ({len(models)} total, {total_size:.1f} MB):\n")

    for model in models:
        print(f"  {model['name']}")
        print(f"    Size: {model['size_mb']:.1f} MB")
        print(f"    Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

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
    """Delete specific model from cache"""
    if delete_model(args.model_name):
        print(f"Deleted: {args.model_name}")
        return 0
    else:
        print(f"Model not found: {args.model_name}")
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
    """Fast inference using server mode (like ollama run)"""
    import sys
    
    try:
        # Always use server mode for the run command
        model = Model(
            args.model,
            quantization=args.quantization,
            use_server=True,
        )
        
        # Load model (will auto-start server if needed)
        model.load()
        
        # Flush stderr to ensure all log messages are written before output
        sys.stderr.flush()
        
        # Add separator between logs and response
        print()
        
        # If no prompt provided, enter interactive mode
        if args.prompt is None:
            return _run_interactive(model, args)
        
        # One-shot mode: generate single response
        if args.stream:
            for token in model.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
            ):
                print(token, end="", flush=True)
            print()
        else:
            response = model.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(response)
        
        # Don't unload - keep model in server cache for next run
        return 0
        
    except Exception as e:
        logger.error(f"Run error: {e}")
        return 1


def _run_interactive(model: Model, args: argparse.Namespace) -> int:
    """Interactive chat mode for oprel run (like ollama run)"""
    import sys
    
    print(f">>> Model loaded: {args.model}")
    print(">>> Send a message (/? for help)")
    print()
    
    # Generate conversation ID for tracking history on server
    conversation_id = str(uuid.uuid4())
    system_prompt = getattr(args, 'system', None)
    
    while True:
        try:
            # Get user input
            try:
                user_input = input(">>> ")
            except EOFError:
                print("\nBye!")
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
                print("Conversation history cleared.\n")
                continue
            
            if not user_input.strip():
                continue
            
            # Generate response
            try:
                if args.stream:
                    for token in model.generate(
                        user_input,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stream=True,
                        conversation_id=conversation_id,
                        system_prompt=system_prompt,
                    ):
                        print(token, end="", flush=True)
                        system_prompt = None  # Clear after first use
                    print("\n")
                else:
                    response = model.generate(
                        user_input,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        conversation_id=conversation_id,
                        system_prompt=system_prompt,
                    )
                    system_prompt = None
                    print(response)
                    print()
            
            except KeyboardInterrupt:
                print("\n")
                continue
            except Exception as e:
                print(f"\nError: {e}\n")
                continue
        
        except KeyboardInterrupt:
            print("\n\nUse /exit to quit\n")
            continue
    
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List models loaded in the server"""
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
    """Stop the oprel daemon server"""
    import requests
    
    server_url = f"http://{args.host}:{args.port}"
    
    # First, list and unload all models
    try:
        response = requests.get(f"{server_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            for model in models:
                model_id = model["model_id"]
                import urllib.parse
                encoded_id = urllib.parse.quote(model_id, safe="")
                try:
                    requests.delete(f"{server_url}/unload/{encoded_id}", timeout=30)
                    print(f"Unloaded: {model_id}")
                except:
                    pass
    except:
        pass
    
    print(f"Server shutdown requested.")
    print("Note: The server process may need to be stopped manually (Ctrl+C)")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    """List all available model aliases"""
    from oprel.downloader.aliases import list_available_aliases
    
    aliases = list_available_aliases()
    
    print(f"Available Models ({len(aliases)} aliases):\n")
    
    # Group by family
    families = {}
    for alias, gguf_id in aliases.items():
        if alias.startswith("llama"):
            family = "Llama (Meta)"
        elif alias.startswith("gemma"):
            family = "Gemma (Google)"
        elif alias.startswith("qwen"):
            family = "Qwen (Alibaba)"
        elif alias.startswith("mistral") or alias.startswith("mixtral"):
            family = "Mistral AI"
        elif alias.startswith("phi"):
            family = "Phi (Microsoft)"
        elif alias.startswith("deepseek"):
            family = "DeepSeek"
        else:
            family = "Other"
        
        if family not in families:
            families[family] = []
        families[family].append((alias, gguf_id))
    
    for family, models in sorted(families.items()):
        print(f"{family}:")
        for alias, gguf_id in sorted(models):
            source = gguf_id.split("/")[0]
            print(f"  {alias:20} -> {source}")
        print()
    
    print("Usage: oprel run <alias> \"your prompt\"")
    return 0


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

    # Run command (NEW) - like ollama run
    run_parser = subparsers.add_parser(
        "run",
        help="Fast inference using server mode (models stay loaded)"
    )
    run_parser.add_argument("model", help="Model ID")
    run_parser.add_argument("prompt", nargs="?", default=None, help="Input prompt (omit for interactive mode)")
    run_parser.add_argument("--quantization", help="Quantization level")
    run_parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    run_parser.add_argument("--stream", action="store_true", default=True, help="Stream response (default)")
    run_parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    run_parser.add_argument("--system", help="System prompt for chat")

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

    # List-models command (NEW)
    subparsers.add_parser("list-models", help="List all available model aliases")

    # Search command (NEW)
    search_parser = subparsers.add_parser("search", help="Search for models by name")
    search_parser.add_argument("query", help="Search term (e.g., 'llama', 'qwen')")

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
        return cmd_run(args)
    elif args.command == "models":
        return cmd_models(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "search":
        return cmd_search(args)
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
