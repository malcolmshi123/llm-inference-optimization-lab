import modal

app = modal.App("llm-inference-optimization-lab")

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
GPU_TYPE = "L4"
VLLM_PORT = 8000
MAX_MODEL_LENGTH = 4096

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "VLLM_CACHE_ROOT": VLLM_CACHE_PATH
    })
)

hf_cache_vol = modal.Volume.from_name("inference-lab-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("inference-lab-vllm-cache", create_if_missing=True)

cache_mounts = {
    HF_CACHE_PATH: hf_cache_vol,
    VLLM_CACHE_PATH: vllm_cache_vol
}

@app.function(gpu=GPU_TYPE, image=vllm_image, volumes=cache_mounts, timeout=900, max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=600, requires_proxy_auth=True)
def serve():
    import subprocess

    command = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LENGTH),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.8",
        "--max-num-batched-tokens", "4096"
    ]

    print("Starting vLLM server with command:", " ".join(command), flush=True)
    process = subprocess.Popen(command)