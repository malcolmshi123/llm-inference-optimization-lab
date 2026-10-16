import os
import socket
import modal

app = modal.App("hello-modal")

# TODO 1: Create a remote function named greet
# It should accept name: str and return str
@app.function()
def greet(name: str) -> str:
    result = (
        "Remote Environment:\n"
        f"\tHostname: {socket.gethostname()}\n"
        f"\tPID: {os.getpid()}\n"
        "\n"
        f"Hello, {name}!"
    )
    return result

# TODO 2: Create a local entrypoint named main
# Invoke greet remotely and print the returned value
@app.local_entrypoint()
def main():
    print(
        "Local Environment:\n"
        f"\tHostname: {socket.gethostname()}\n"
        f"\tPID: {os.getpid()}\n"
    )

    result = greet.remote("Modal")
    print(result)