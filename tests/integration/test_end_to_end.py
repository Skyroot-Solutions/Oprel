# Try with a tiny model first (faster download)
from oprel import Model

with Model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF") as model:
    response = model.generate("Hello!", max_tokens=20)
    print(response)
    
# If this works, YOU'RE DONE. Ship it.