import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM_VERSION = "0.21.0"
GPU_TYPE = "A10"
VLLM_PORT = 8000
MAX_MODEL_LENGTH = 4096

PREFIX_CACHING_ENV = "INFERENCE_LAB_PREFIX_CACHING"
TOKEN_BUDGET_ENV = "INFERENCE_LAB_MAX_NUM_BATCHED_TOKENS"
PROFILE_SESSION_ENV = "INFERENCE_LAB_PROFILE_SESSION"
PROFILE_ROOT = "/profiles"
PROFILE_VOLUME_NAME = "inference-lab-profiles"

@dataclass(frozen=True)
class ServerConfig:
    prefix_caching: bool = True
    max_num_batched_tokens: int = 2048
    profile_session: str | None = None

    def __post_init__(self):
        if type(self.prefix_caching) is not bool:
            raise ValueError(f"{PREFIX_CACHING_ENV} must be true or false")

        if (
            type(self.max_num_batched_tokens) is not int
            or self.max_num_batched_tokens <= 0
        ):
            raise ValueError(f"{TOKEN_BUDGET_ENV} must be a positive integer")

        if self.profile_session is not None and (
            not isinstance(self.profile_session, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.profile_session)
            is None
        ):
            raise ValueError(
                f"{PROFILE_SESSION_ENV} must be 1–64 ASCII letters, digits, "
                "underscores, or hyphens, starting with a letter or digit"
            )

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

        return cls(
            prefix_caching=prefix == "true",
            max_num_batched_tokens=budget,
            profile_session=environ.get(PROFILE_SESSION_ENV, "").strip() or None,
        )

    @property
    def profile_directory(self) -> str | None:
        if self.profile_session is None:
            return None

        return f"{PROFILE_ROOT}/{self.profile_session}"

    def environment(self) -> dict[str, str]:
        environment = {
            PREFIX_CACHING_ENV: str(self.prefix_caching).lower(),
            TOKEN_BUDGET_ENV: str(self.max_num_batched_tokens),
        }

        if self.profile_session is not None:
            environment[PROFILE_SESSION_ENV] = self.profile_session

        return environment

    def command(self) -> list[str]:
        command = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--tokenizer-revision",
            MODEL_REVISION,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(MAX_MODEL_LENGTH),
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--max-num-seqs",
            "256",
            "--enable-chunked-prefill",
            "--enable-prefix-caching"
            if self.prefix_caching
            else "--no-enable-prefix-caching",
        ]

        if self.profile_directory is not None:
            command.extend(
                [
                    "--profiler-config",
                    json.dumps(
                        {
                            "profiler": "torch",
                            "torch_profiler_dir": self.profile_directory,
                            "torch_profiler_with_stack": False,
                            "torch_profiler_record_shapes": False,
                            "torch_profiler_with_memory": False,
                            "torch_profiler_with_flops": False,
                            "torch_profiler_use_gzip": True,
                        }
                    ),
                ]
            )

        return command
