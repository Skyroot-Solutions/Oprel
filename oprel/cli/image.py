"""Image generation commands for Oprel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from oprel.core.config import Config
from oprel.runtime.binaries.installer import ensure_binary
from oprel.runtime.image_generation import ImageGenerationParams, generate_image_file
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_gen_image(args: argparse.Namespace) -> int:
    """Generate an image using stable-diffusion.cpp."""
    try:
        output_path = Path(args.output).expanduser().resolve() if args.output else None
        params = ImageGenerationParams(
            model=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg_scale=args.guidance or 7.0,
            seed=getattr(args, "seed", -1),
            sampler=getattr(args, "sampler", None),
            output_path=output_path,
        )

        print(f"Generating with stable-diffusion.cpp GGUF model: {args.model}")
        print(f"Prompt: {args.prompt}")
        start = time.time()
        generated_path = generate_image_file(params)
        elapsed = time.time() - start

        print(f"Saved to: {generated_path}")
        print(f"Completed in {elapsed:.1f}s")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled")
        return 1
    except Exception as exc:
        logger.error("Image generation failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1


def cmd_setup_image(args: argparse.Namespace) -> int:
    """Download the stable-diffusion.cpp binary for this machine."""
    try:
        config = Config()
        binary_path = ensure_binary(
            backend="stable-diffusion.cpp",
            version=config.image_binary_version,
            binary_dir=config.binary_dir,
            config=config,
        )
        print(f"stable-diffusion.cpp is ready: {binary_path}")
        return 0
    except Exception as exc:
        logger.error("Image setup failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1
