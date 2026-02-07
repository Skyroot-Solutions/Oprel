"""
Example: Using Oprel Server with Ollama SDK  

This demonstrates how to use the Oprel server with the Ollama Python SDK.
The server is 100% compatible with Ollama's API format.
"""

try:
    import ollama
except ImportError:
    print("Installing ollama package...")
    import subprocess
    subprocess.check_call(["pip", "install", "ollama"])
    import ollama

# Configure client to use local Oprel server
client = ollama.Client(host='http://localhost:11434')

print("🚀 Oprel Server Example - Ollama SDK\n")
print("Make sure you've started the server with: oprel serve\n")

# Example 1: Simple chat (non-streaming)
print("=" * 60)
print("Example 1: Simple Chat (Non-Streaming)")
print("=" * 60)

response = client.chat(
    model='qwen3-3b',
    messages=[
        {'role': 'user', 'content': 'Why is the ocean salty?'}
    ]
)

print(f"\nResponse: {response['message']['content']}")

# Example 2: Streaming chat
print("\n" + "=" * 60)
print("Example 2: Streaming Chat")
print("=" * 60)

print("\nStreaming response: ", end="", flush=True)
stream = client.chat(
    model='qwen3-3b',
    messages=[
        {'role': 'user', 'content': 'Tell me a short joke'}
    ],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print()

# Example 3: Text generation (non-chat format)
print("\n" + "=" * 60)
print("Example 3: Text Generation (Generate API)")
print("=" * 60)

response = client.generate(
    model='qwen3-3b',
    prompt='Complete this sentence: The best thing about programming is'
)

print(f"\nResponse: {response['response']}")

# Example 4: List available models
print("\n" + "=" * 60)
print("Example 4: List Available Models")
print("=" * 60)

models = client.list()
print(f"\nAvailable models: {len(models['models'])}")
for model in models['models'][:5]:  # Show first 5
    print(f"  - {model['name']}")
if len(models['models']) > 5:
    print("  ...")

print("\n✅ All examples completed successfully!")
print("\nThe Oprel server works with:")
print("  ✓ OpenAI SDK (Python, Node.js, etc.)")
print("  ✓ Ollama SDK (Python, JavaScript, Go, etc.)")
print("  ✓ Any HTTP client (curl, fetch, axios, etc.)")
print("\nThis means you can switch between Ollama and Oprel with zero code changes!")
