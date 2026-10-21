import modal

app = modal.App("image-dependencies")

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("numpy==2.2.6")


# TODO 1: Define a remote function named calculate_mean
# Attach the custom image to the function
# Accept a list of numbers and return their mean
@app.function(image=image)
def calculate_mean(numbers: list[int | float]) -> float:
    import numpy as np

    return float(np.mean(numbers))


# TODO 2: Define a local entrypoint
# Call calculate_mean remotely with [10, 20, 30, 40]
# Print the result
@app.local_entrypoint()
def main():
    numbers = [1, 2, 3, 4]
    result = calculate_mean.remote(numbers)
    print(f"The mean of {numbers} is {result}.")
