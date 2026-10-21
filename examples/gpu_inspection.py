import modal

app = modal.App("gpu-inspection")


# TODO 1: Define a remote function called inspect_gpu
# Request an L4 GPU and set timeout=60
# Return the output of the nvidia-smi command
@app.function(gpu="L4", timeout=60)
def inspect_gpu() -> str:
    import subprocess

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    return result.stdout


# TODO 2: Define a local entrypoint
# Invoke inspect_gpu remotely and print the result
@app.local_entrypoint()
def main():
    result = inspect_gpu.remote()
    print(result)
