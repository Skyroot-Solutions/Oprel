"""
Image generation commands for Oprel CLI.
"""
import argparse
import time
import sys
import subprocess
from pathlib import Path
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


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
            from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
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


def cmd_setup_image(args):
    """Setup image generation (install ComfyUI + CUDA dependencies)"""
    from oprel.downloader.comfyui_installer import ensure_comfyui_ready
    from oprel.telemetry.hardware import detect_gpu
    
    print("🎨 Setting up Image Generation...")
    print()
    
    # Step 1: Check GPU
    gpu_info = detect_gpu()
    if not gpu_info or gpu_info.get("gpu_type") != "cuda":
        print("⚠ Warning: No NVIDIA GPU detected!")
        print("Image generation will work but will be slower on CPU.")
        print()
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return 0
    else:
        vram_gb = gpu_info.get("vram_total_gb", 0)
        print(f"✓ Detected: {gpu_info.get('gpu_name')} ({vram_gb:.1f}GB VRAM)")
        print()
    
    # Step 2: Install PyTorch with CUDA
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ PyTorch with CUDA already installed (version {torch.__version__})")
        else:
            print("⚠ PyTorch installed but CUDA not available")
            print("Reinstalling PyTorch with CUDA support...")
            print()
            print("Installing: pytorch torchvision --index-url https://download.pytorch.org/whl/cu121")
            subprocess.check_call([
                "pip", "install", "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ])
            print("✓ PyTorch with CUDA installed")
    except ImportError:
        print("Installing PyTorch with CUDA...")
        print()
        subprocess.check_call([
            "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print("✓ PyTorch with CUDA installed")
    
    print()
    
    # Step 3: Install ComfyUI
    print("Installing ComfyUI...")
    ensure_comfyui_ready()
    print("✓ ComfyUI installed successfully")
    
    print()
    print("✅ Image generation setup complete!")
    print()
    print("Try it out:")
    print("  oprel gen-image sd-1.5 'a beautiful landscape'")
    print("  oprel gen-image sdxl-turbo 'a cute cat'")
    
    return 0
