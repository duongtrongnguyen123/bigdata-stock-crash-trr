import sys
print("=== vLLM / Blackwell probe ===", flush=True)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        mj, mn = torch.cuda.get_device_capability(0)
        print(f"device {torch.cuda.get_device_name(0)} sm_{mj}{mn}", flush=True)
except Exception as e:
    print("torch ERR", e, flush=True)
for mod in ["vllm", "flashinfer", "flash_attn"]:
    try:
        m = __import__(mod)
        print(f"{mod}: AVAILABLE {getattr(m,'__version__','?')}", flush=True)
    except Exception as e:
        print(f"{mod}: MISSING ({type(e).__name__})", flush=True)
