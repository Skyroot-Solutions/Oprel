"""Setup command implementation"""

def cmd_setup_image(args):
    """Setup image generation (install ComfyUI + CUDA dependencies)"""
    from oprel.downloader.comfyui_installer import ensure_comfyui_ready
    from oprel.telemetry.hardware import detect_gpu
    import subprocess
    
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
