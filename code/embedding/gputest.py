import modal
app = modal.App("vec2vec-gputest",
                image=modal.Image.debian_slim().pip_install("torch==2.5.1"))

@app.function(gpu="A10G", timeout=300)
def a10g():
    import torch, subprocess
    return f"A10G ok cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}"

@app.function(gpu="T4", timeout=300)
def t4():
    import torch
    return f"T4 ok cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}"

@app.local_entrypoint()
def main():
    try: print(a10g.remote())
    except Exception as e: print("A10G FAILED:", type(e).__name__, str(e)[:200])
    try: print(t4.remote())
    except Exception as e: print("T4 FAILED:", type(e).__name__, str(e)[:200])
