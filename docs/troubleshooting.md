# Troubleshooting Guide

This guide covers common issues and their solutions.

## Installation Issues

### pip install fails

**Problem:**
```bash
ERROR: Could not find a version that satisfies the requirement oprel
```

**Solution:**
1. Update pip:
```bash
python -m pip install --upgrade pip
```

2. Check Python version (requires 3.9+):
```bash
python --version
```

3. Try with explicit version:
```bash
pip install oprel==0.3.0
```

### Missing dependencies

**Problem:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
Install with server dependencies:
```bash
pip install oprel[server]
```

For all dependencies:
```bash
pip install oprel[all]
```

## Model Loading Issues

### Model not found error

**Problem:**
```python
ModelNotFoundError: Model 'qwencoder' not found
```

**Solutions:**

1. Check internet connection (models download from HuggingFace)
2. List available aliases:
```bash
oprel list-models
```

3. Use full HuggingFace path:
```python
Model("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")
```

4. Check firewall settings (allow HuggingFace access)

### Download fails or hangs

**Problem:**
Download stalls at certain percentage or fails with timeout.

**Solutions:**

1. Check available disk space (need 4-8GB per model)
```bash
# Windows
Get-PSDrive

# Linux/macOS
df -h
```

2. Clear partial downloads:
```bash
oprel cache clear
```

3. Set custom cache directory with more space:
```python
from oprel.core.config import Config

config = Config(cache_dir="/path/with/space")
model = Model("qwencoder", config=config)
```

4. Use environment variable:
```bash
export OPREL_CACHE_DIR=/data/models
```

## Memory Issues

### Out of memory error

**Problem:**
```
MemoryError: Insufficient memory to load model
```

**Solutions:**

1. Use lighter quantization:
```python
# Instead of Q4_K_M (4-5GB)
model = Model("qwencoder", quantization="Q2_K")  # 2-3GB
```

2. Close other applications

3. Use smaller model:
```python
Model("phi3.5")  # 3.8B parameters vs 7B
```

4. Set explicit memory limit:
```python
model = Model("qwencoder", max_memory_mb=3072)
```

5. Reduce context size:
```python
from oprel.core.config import Config

config = Config(ctx_size=2048)  # Default is 4096
model = Model("qwencoder", config=config)
```

### System freezes during load

**Problem:**
Computer becomes unresponsive when loading model.

**Solutions:**

1. Set conservative memory limit:
```python
import psutil

available_mb = psutil.virtual_memory().available / (1024 * 1024)
safe_limit = int(available_mb * 0.7)  # Use 70% of available
model = Model("qwencoder", max_memory_mb=safe_limit)
```

2. Use direct mode to avoid daemon:
```python
model = Model("qwencoder", use_server=False)
```

3. Monitor memory before loading:
```python
import psutil

mem = psutil.virtual_memory()
print(f"Available: {mem.available / (1024**3):.2f} GB")
```

## Backend Issues

### Backend process crashes

**Problem:**
```
BackendError: Backend process terminated unexpectedly
```

**Solutions:**

1. Check backend logs:
```bash
oprel serve --log-level DEBUG
```

2. Server mode auto-recovers, but in direct mode:
```python
model = Model("qwencoder", use_server=False)
try:
    response = model.generate("prompt")
except BackendError:
    model.unload()
    model.load()  # Restart backend
    response = model.generate("prompt")
```

3. Reduce load with lighter quantization or smaller model

4. Check for conflicting ports:
```bash
# Windows
netstat -ano | findstr :11434

# Linux/macOS
lsof -i :11434
```

### llama-server not starting

**Problem:**
Backend process fails to start.

**Solutions:**

1. Check if binaries are installed:
```bash
oprel cache list
```

2. Reinstall binaries:
```bash
python -m oprel.runtime.binaries.installer
```

3. Check permissions:
```bash
# Linux/macOS
chmod +x ~/.oprel/binaries/llama-server
```

4. Check antivirus (may block executable)

### Hidden CMD windows appearing (Windows)

**Problem:**
CMD windows flash or appear when running models.

**Solution:**
This should be fixed in current version. If still occurring:

1. Verify you have latest version:
```bash
pip install --upgrade oprel
```

2. Check implementation:
```python
import subprocess
# Should use subprocess.CREATE_NO_WINDOW flag
```

## Server Issues

### Server won't start

**Problem:**
```bash
oprel serve
# Error: Port already in use
```

**Solutions:**

1. Check if already running:
```bash
# Windows
Get-Process | Where-Object { $_.ProcessName -like "*uvicorn*" }

# Linux/macOS
ps aux | grep uvicorn
```

2. Use different port:
```bash
oprel serve --port 8080
```

3. Kill existing server:
```bash
# Windows
Stop-Process -Name uvicorn -Force

# Linux/macOS
pkill uvicorn
```

4. Check port availability:
```bash
# Windows
Test-NetConnection -ComputerName localhost -Port 11434

# Linux/macOS
nc -zv localhost 11434
```

### Can't connect to server

**Problem:**
```
ConnectionError: Unable to connect to server at localhost:11434
```

**Solutions:**

1. Verify server is running:
```bash
curl http://localhost:11434/health
```

2. Check firewall settings

3. Use direct mode instead:
```python
model = Model("qwencoder", use_server=False)
```

4. Start server explicitly:
```bash
oprel serve
```

## Generation Issues

### Generation very slow

**Problem:**
Each generation takes 60+ seconds.

**Solutions:**

1. Use server mode (default):
```python
model = Model("qwencoder", use_server=True)  # Fast after first load
```

2. Enable GPU acceleration:
```python
# Check if GPU detected
from oprel.telemetry.hardware import detect_gpu

gpu_info = detect_gpu()
print(gpu_info)
```

3. Reduce max_tokens:
```python
response = model.generate("prompt", max_tokens=256)  # vs 512 default
```

4. Increase batch size:
```python
from oprel.core.config import Config

config = Config(batch_size=1024)  # Default 512
model = Model("qwencoder", config=config)
```

### Empty or nonsensical responses

**Problem:**
Model generates gibberish or empty text.

**Solutions:**

1. Check prompt formatting:
```python
# Good
response = model.generate("What is Python?")

# May not work well
response = model.generate("")
```

2. Adjust temperature:
```python
# Too high (1.5+) = random
# Too low (0.0-0.1) = repetitive
response = model.generate("prompt", temperature=0.7)  # Sweet spot
```

3. Verify model downloaded correctly:
```bash
oprel cache list
# Delete and redownload if suspicious
oprel cache delete qwencoder
```

4. Try different model:
```python
model = Model("llama3.1")  # vs qwencoder
```

### Streaming not working

**Problem:**
```python
for chunk in model.generate("prompt", stream=True):
    print(chunk)  # All at once, not streaming
```

**Solutions:**

1. Use Ollama API for proper streaming:
```python
from oprel import generate

stream = generate(model='qwencoder', prompt='prompt', stream=True)
for chunk in stream:
    print(chunk.response, end='', flush=True)
```

2. Ensure flush:
```python
import sys

for chunk in stream:
    print(chunk, end='', flush=True)
    sys.stdout.flush()
```

## Conversation Issues

### Context not retained

**Problem:**
Model doesn't remember previous messages.

**Solutions:**

1. Use same conversation_id:
```python
conv_id = "my-chat"

response1 = model.generate("My name is Alice", conversation_id=conv_id)
response2 = model.generate("What's my name?", conversation_id=conv_id)
```

2. Don't reset conversation accidentally:
```python
# Wrong - resets each time
response = model.generate("prompt", conversation_id="chat", reset_conversation=True)

# Right
response = model.generate("prompt", conversation_id="chat")
```

3. Use server mode (required for conversation memory):
```python
model = Model("qwencoder", use_server=True)
```

### Conversation too long / context overflow

**Problem:**
```
Error: Context length exceeded
```

**Solutions:**

1. Reset conversation periodically:
```python
if message_count > 20:
    response = model.generate(
        "New topic",
        conversation_id=conv_id,
        reset_conversation=True
    )
```

2. Increase context size:
```python
from oprel.core.config import Config

config = Config(ctx_size=8192)  # Default 4096
model = Model("qwencoder", config=config)
```

3. Use new conversation ID:
```python
conv_id_2 = f"chat-{uuid.uuid4()}"
```

## CLI Issues

### oprel command not found

**Problem:**
```bash
oprel run qwencoder
# Command 'oprel' not found
```

**Solutions:**

1. Ensure installation completed:
```bash
pip install oprel
```

2. Check if in PATH:
```bash
# Windows
python -m oprel.cli.main run qwencoder

# Linux/macOS
python3 -m oprel.cli.main run qwencoder
```

3. Add to PATH (if needed):
```bash
# Add Python scripts directory to PATH
export PATH="$PATH:$HOME/.local/bin"
```

4. Use full module path:
```bash
python -m oprel.cli.main --help
```

### Interactive mode commands not working

**Problem:**
```
>>> /exit
# Nothing happens
```

**Solutions:**

1. Ensure exact spelling:
- `/exit`, `/bye`, `/quit` (not `exit` or `/Exit`)
- `/reset` (not `/clear`)
- `/?` (not `/help`)

2. Use proper command:
```bash
# Interactive mode
oprel run qwencoder

# Not chat mode (different interface)
oprel chat qwencoder
```

## Platform-Specific Issues

### Windows

**Problem:** PowerShell execution policy error

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Problem:** Path too long error

**Solution:**
Enable long paths in Windows:
```
Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1
```

### macOS

**Problem:** "llama-server" cannot be opened because the developer cannot be verified

**Solution:**
```bash
xattr -d com.apple.quarantine ~/.oprel/binaries/llama-server
```

**Problem:** Metal GPU not detected

**Solution:**
Ensure you have compatible Mac (M1/M2/M3):
```python
from oprel.telemetry.hardware import detect_gpu
print(detect_gpu())
```

### Linux

**Problem:** Permission denied for binaries

**Solution:**
```bash
chmod +x ~/.oprel/binaries/llama-server
```

**Problem:** CUDA not found

**Solution:**
Install CUDA toolkit:
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Verify
nvidia-smi
```

## Testing Issues

### Tests fail to run

**Problem:**
```bash
pytest tests/
# Import errors or test failures
```

**Solutions:**

1. Install test dependencies:
```bash
pip install pytest pytest-cov
```

2. Install oprel in editable mode:
```bash
pip install -e .
```

3. Run specific test file:
```bash
pytest tests/unit/test_client_api.py -v
```

4. Check Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from oprel import Model
model = Model("qwencoder")
```

### Monitor Processes

```bash
# Windows PowerShell
while ($true) {
    Get-Process | Where-Object { $_.ProcessName -like "*llama*" }
    Start-Sleep -Seconds 2
}

# Linux/macOS
watch -n 2 'ps aux | grep llama'
```

### Check Memory Usage

```python
import psutil

def monitor_memory():
    process = psutil.Process()
    print(f"Memory: {process.memory_info().rss / (1024**3):.2f} GB")

# Call periodically during generation
```

### Inspect Model Cache

```bash
# List cached models
oprel cache list

# Check cache directory
# Windows
dir %USERPROFILE%\.cache\oprel

# Linux/macOS
ls -lh ~/.cache/oprel
```

### Test HTTP Connectivity

```python
import requests

try:
    response = requests.get("http://localhost:11434/health", timeout=5)
    print(f"Server responding: {response.status_code}")
except Exception as e:
    print(f"Server not accessible: {e}")
```

## Getting More Help

If your issue isn't covered here:

1. Check existing issues: https://github.com/ragultv/oprel-SDK/issues

2. Enable debug logging and capture error:
```python
import logging
logging.basicConfig(level=logging.DEBUG, filename='oprel_debug.log')
```

3. Create new issue with:
   - Oprel version (`pip show oprel`)
   - Python version (`python --version`)
   - OS and version
   - Complete error traceback
   - Debug logs
   - Steps to reproduce

4. Check documentation:
   - [API Reference](api_reference.md)
   - [Architecture](architecture.md)
   - [Quick Start](quickstart.md)

## Common Error Messages

### "Connection refused"
- Server not running: `oprel serve`
- Wrong port: Check `OPREL_SERVER_PORT`

### "Model file corrupted"
- Redownload: `oprel cache delete <model>`

### "Unsupported quantization"
- Use valid level: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0

### "Backend initialization failed"
- Check memory available
- Try lighter quantization
- Verify binary permissions

### "Timeout waiting for backend"
- Increase timeout in config
- Check system resources
- Try direct mode

## Prevention Tips

1. Always use try-except for error handling
2. Set reasonable memory limits
3. Use server mode for repeated requests
4. Monitor disk space for cache
5. Keep dependencies updated
6. Test with small prompts first
7. Use appropriate quantization for your RAM
8. Enable logging in production

## Performance Checklist

If performance is poor, check:

- [ ] Using server mode (not direct mode)
- [ ] GPU detected and enabled
- [ ] Appropriate quantization (Q4_K_M for most)
- [ ] Sufficient RAM available
- [ ] Model fully cached (not redownloading)
- [ ] No other heavy processes running
- [ ] Context size not excessive
- [ ] Batch size optimized for hardware
