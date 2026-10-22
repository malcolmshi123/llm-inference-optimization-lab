import random

def generate_prompt(target_tokens: int, seed: int) -> str:
    if target_tokens <= 0:
        return ""

    words = (
        "river",
        "tree",
        "cloud",
        "stone",
        "wind",
        "rain",
        "sun",
        "moon",
        "bird",
        "field",
        "forest",
        "road",
        "garden",
        "water",
        "light",
        "night",
    )
    content = " ".join(random.Random(seed).choices(words, k=max(0, target_tokens - 15)))
    return "Summarize the following text:\n\n" + content

def count_prompt_tokens(tokenizer, prompt: str) -> int:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )

    return len(encoded["input_ids"])

if __name__ == "__main__":
    from transformers import AutoTokenizer

    prompt_a = generate_prompt(256, 42)
    prompt_b = generate_prompt(256, 42)
    prompt_c = generate_prompt(256, 43)
    prompt_long = generate_prompt(2048, 42)

    assert prompt_a == prompt_b
    assert prompt_a != prompt_c
    print(f"prompt_a approximate word count: {len(prompt_a.split())}")
    print(f"prompt_long approximate word count: {len(prompt_long.split())}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    print("Short:", count_prompt_tokens(tokenizer, prompt_a))
    print("Long:", count_prompt_tokens(tokenizer, prompt_long))

    for target_tokens in [256, 2048]:
        print(f"\nTarget: {target_tokens} tokens")

        for seed in [42, 43, 44]:
            prompt = generate_prompt(target_tokens, seed)
            actual_tokens = count_prompt_tokens(tokenizer, prompt)

            print(
                f"Seed {seed}: "
                f"{actual_tokens} tokens "
                f"(difference: {actual_tokens - target_tokens:+d})"
            )
