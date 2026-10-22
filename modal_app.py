import modal

from server_config import (
    GPU_TYPE,
    PROFILE_ROOT,
    PROFILE_VOLUME_NAME,
    VLLM_PORT,
    VLLM_VERSION,
    ServerConfig,
)

SERVER_CONFIG = ServerConfig.from_env()

app = modal.App("llm-inference-optimization-lab")

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            **SERVER_CONFIG.environment(),
        }
    )
    .add_local_python_source("server_config")
)

hf_cache_vol = modal.Volume.from_name("inference-lab-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name(
    "inference-lab-vllm-cache", create_if_missing=True
)

cache_mounts = {
    HF_CACHE_PATH: hf_cache_vol,
    VLLM_CACHE_PATH: vllm_cache_vol,
}

if SERVER_CONFIG.profile_session is not None:
    cache_mounts[PROFILE_ROOT] = modal.Volume.from_name(
        PROFILE_VOLUME_NAME, create_if_missing=True
    )

@app.function(
    gpu=GPU_TYPE, image=vllm_image, volumes=cache_mounts, timeout=900, max_containers=1
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=600, requires_proxy_auth=True)
def serve():
    import subprocess
    from pathlib import Path

    config = ServerConfig.from_env()
    command = config.command()

    if config.profile_directory is not None:
        Path(config.profile_directory).mkdir(parents=True, exist_ok=True)
        print("Profiler traces directory:", config.profile_directory, flush=True)

    print("Starting vLLM server with command:", " ".join(command), flush=True)
    subprocess.Popen(command)
