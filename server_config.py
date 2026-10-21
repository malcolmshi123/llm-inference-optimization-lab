import os
from collections.abc import Mapping
from dataclasses import dataclass

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM_VERSION = "0.21.0"
GPU_TYPE = "L4"
VLLM_PORT = 8000
MAX_MODEL_LENGTH = 4096

PREFIX_CACHING_ENV = "INFERENCE_LAB_PREFIX_CACHING"
TOKEN_BUDGET_ENV = "INFERENCE_LAB_MAX_NUM_BATCHED_TOKENS"

@dataclass(frozen=True)
class ServerConfig:
    prefix_caching: bool = True
    max_num_batched_tokens: int = 2048

    def __post_init__(self):
        if type(self.prefix_caching) is not bool:
            raise ValueError(f"{PREFIX_CACHING_ENV} must be true or false")

        if type(self.max_num_batched_tokens) is not int or self.max_num_batched_tokens <= 0:
            raise ValueError(f"{TOKEN_BUDGET_ENV} must be a positive integer")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServerConfig":
        environ = os.environ if environ is None else environ
        prefix = environ.get(PREFIX_CACHING_ENV, "true").strip().lower()

        if prefix not in {"true", "false"}:
            raise ValueError(f"{PREFIX_CACHING_ENV} must be true or false")

        try:
            budget = int(environ.get(TOKEN_BUDGET_ENV, "2048"))
        except ValueError as exc:
            raise ValueError(f"{TOKEN_BUDGET_ENV} must be a positive integer") from exc

        return cls(prefix_caching=prefix == "true", max_num_batched_tokens=budget)

    def environment(self) -> dict[str, str]:
        return {
            PREFIX_CACHING_ENV: str(self.prefix_caching).lower(),
            TOKEN_BUDGET_ENV: str(self.max_num_batched_tokens)
        }

    def command(self) -> list[str]:
        return [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--tokenizer-revision", MODEL_REVISION,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LENGTH),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.8",
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--max-num-seqs", "256",
            "--enable-chunked-prefill",
            "--enable-prefix-caching" if self.prefix_caching else "--no-enable-prefix-caching"
        ]
