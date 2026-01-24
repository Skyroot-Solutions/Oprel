"""
Basic usage examples for Oprel SDK
"""

from oprel import Model

# Example 1: Simple text generation
def simple_generation():
    """Basic text generation with auto-configuration"""
    model = Model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    
    response = model.generate(
        "What is the capital of France?",
        max_tokens=100
    )
    
    print(response)
    model.unload()


# Example 2: Using context manager
def context_manager_example():
    """Recommended: Use context manager for automatic cleanup"""
    with Model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF") as model:
        response = model.generate("Explain quantum computing in simple terms")
        print(response)
    # Model is automatically unloaded


# Example 3: Streaming responses
def streaming_example():
    """Stream tokens as they're generated"""
    with Model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF") as model:
        stream = model.generate(
            "Write a short poem about Python",
            stream=True
        )
        
        for token in stream:
            print(token, end="", flush=True)
        print()


# Example 4: Custom quantization
def custom_quantization():
    """Manually specify quantization level"""
    model = Model(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        quantization="Q8_0",  # Higher quality, more memory
        max_memory_mb=16384   # 16GB limit
    )
    
    response = model.generate("What is machine learning?")
    print(response)
    model.unload()


# Example 5: Pre-loading for faster first response
def preload_example():
    """Pre-load model to avoid latency on first request"""
    model = Model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    
    print("Loading model...")
    model.load()  # Download and start process now
    print("Model ready!")
    
    # First generation is now instant
    response = model.generate("Hello!")
    print(response)
    
    model.unload()


# Example 6: Error handling
def error_handling_example():
    """Handle memory errors gracefully"""
    from oprel import MemoryError as OprelMemoryError
    
    model = Model(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",  # Large model
        max_memory_mb=4096  # Only 4GB allowed
    )
    
    try:
        response = model.generate("Write a very long essay...")
    except OprelMemoryError as e:
        print(f"Memory limit exceeded: {e}")
        print("Try a smaller model or increase max_memory_mb")
    finally:
        model.unload()


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("Example 1: Simple Generation")
    print("=" * 60)
    simple_generation()
    
    print("\n" + "=" * 60)
    print("Example 2: Context Manager (Recommended)")
    print("=" * 60)
    context_manager_example()
    
    print("\n" + "=" * 60)
    print("Example 3: Streaming")
    print("=" * 60)
    streaming_example()

    print("\n" + "=" * 60)
    print("Example 4: Custom Quantization")
    print("=" * 60)
    custom_quantization()

    print("\n" + "=" * 60)
    print("Example 5: Pre-loading")
    print("=" * 60)
    preload_example()

    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)
    error_handling_example()