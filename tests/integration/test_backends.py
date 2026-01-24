# The llama.cpp URLs in registry.py might be outdated
# Go to: https://github.com/ggerganov/llama.cpp/releases
# Update the URLs in oprel/runtime/binaries/registry.py

# Test binary download:
from oprel.runtime.binaries.installer import ensure_binary
from pathlib import Path

binary = ensure_binary('llama.cpp', 'b3901', Path.home() / '.cache/oprel/bin')
print(f"Binary: {binary}")