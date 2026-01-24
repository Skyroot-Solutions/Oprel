"""
Command-line interface for Oprel SDK
"""

import sys
import argparse
from pathlib import Path

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
    print("Type 'exit' or 'quit' to end the conversation.\n")

    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
        ) as model:
            print("Model loaded. Ready to chat!\n")

            while True:
                try:
                    prompt = input("You: ")
                    if prompt.lower() in ["exit", "quit"]:
                        break

                    if not prompt.strip():
                        continue

                    print("Assistant: ", end="", flush=True)

                    if args.stream:
                        for token in model.generate(prompt, stream=True):
                            print(token, end="", flush=True)
                        print()
                    else:
                        response = model.generate(prompt)
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
    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
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
    chat_parser.add_argument("--stream", action="store_true", help="Stream responses")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("model", help="Model ID")
    gen_parser.add_argument("prompt", help="Input prompt")
    gen_parser.add_argument("--quantization", help="Quantization level")
    gen_parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    gen_parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--stream", action="store_true", help="Stream response")

    # Info command
    subparsers.add_parser("info", help="Show system information")

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

    # Route to command handlers
    if args.command == "chat":
        return cmd_chat(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "info":
        return cmd_info(args)
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
