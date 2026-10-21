import json
import unittest
from unittest.mock import patch

from server_config import (
    MODEL_REVISION,
    PREFIX_CACHING_ENV,
    PROFILE_SESSION_ENV,
    TOKEN_BUDGET_ENV,
    ServerConfig,
)


class ServerConfigTests(unittest.TestCase):
    def test_defaults_pin_revisions_and_experimental_controls(self):
        config = ServerConfig.from_env({})
        command = config.command()

        self.assertTrue(config.prefix_caching)
        self.assertEqual(config.max_num_batched_tokens, 2048)
        self.assertIsNone(config.profile_session)
        self.assertIsNone(config.profile_directory)
        self.assertNotIn("--profiler-config", command)
        self.assertNotIn(PROFILE_SESSION_ENV, config.environment())

        for flag, value in (
            ("--revision", MODEL_REVISION),
            ("--tokenizer-revision", MODEL_REVISION),
            ("--max-model-len", "4096"),
            ("--dtype", "bfloat16"),
            ("--gpu-memory-utilization", "0.8"),
            ("--max-num-batched-tokens", "2048"),
            ("--max-num-seqs", "256"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)

        self.assertIn("--enable-chunked-prefill", command)
        self.assertIn("--enable-prefix-caching", command)

    def test_cache_treatment_changes_only_the_cache_flag(self):
        enabled = ServerConfig.from_env(
            {
                PREFIX_CACHING_ENV: "true",
            }
        ).command()
        disabled = ServerConfig.from_env(
            {
                PREFIX_CACHING_ENV: "false",
            }
        ).command()

        self.assertEqual(enabled[:-1], disabled[:-1])
        self.assertEqual(disabled[-1], "--no-enable-prefix-caching")
        self.assertNotIn("--enable-prefix-caching", disabled)

    def test_environment_override_reaches_the_server_command(self):
        with patch.dict(
            "os.environ",
            {
                PREFIX_CACHING_ENV: " FALSE ",
                TOKEN_BUDGET_ENV: "4096",
            },
            clear=True,
        ):
            config = ServerConfig.from_env()

        command = config.command()

        self.assertFalse(config.prefix_caching)
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "4096")
        self.assertEqual(
            config.environment(),
            {
                PREFIX_CACHING_ENV: "false",
                TOKEN_BUDGET_ENV: "4096",
            },
        )
        self.assertEqual(ServerConfig.from_env(config.environment()), config)

    def test_invalid_environment_fails_before_deployment(self):
        for value in ("", "0", "1", "yes", "flase"):
            with (
                self.subTest(prefix=value),
                self.assertRaisesRegex(ValueError, PREFIX_CACHING_ENV),
            ):
                ServerConfig.from_env(
                    {
                        PREFIX_CACHING_ENV: value,
                    }
                )

        for value in ("", "0", "-1", "1.5", "many"):
            with (
                self.subTest(budget=value),
                self.assertRaisesRegex(ValueError, TOKEN_BUDGET_ENV),
            ):
                ServerConfig.from_env(
                    {
                        TOKEN_BUDGET_ENV: value,
                    }
                )

    def test_direct_configuration_rejects_invalid_types_and_values(self):
        for value in (True, 0, -1, 2.5, "2048"):
            with self.subTest(budget=value), self.assertRaises(ValueError):
                ServerConfig(max_num_batched_tokens=value)

        with self.assertRaises(ValueError):
            ServerConfig(prefix_caching="false")

    def test_profiling_is_explicit_and_preserves_serving_controls(self):
        for prefix_caching in (False, True):
            with self.subTest(prefix_caching=prefix_caching):
                baseline = ServerConfig(prefix_caching=prefix_caching)
                config = ServerConfig.from_env(
                    {
                        **baseline.environment(),
                        PROFILE_SESSION_ENV: " shared-c8-off-01 ",
                    }
                )
                command = config.command()
                index = command.index("--profiler-config")
                self.assertEqual(command[:index], baseline.command())
                self.assertEqual(len(command), index + 2)
                self.assertEqual(config.profile_session, "shared-c8-off-01")
                self.assertEqual(config.profile_directory, "/profiles/shared-c8-off-01")
                self.assertEqual(
                    json.loads(command[index + 1]),
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": "/profiles/shared-c8-off-01",
                        "torch_profiler_with_stack": False,
                        "torch_profiler_record_shapes": False,
                        "torch_profiler_with_memory": False,
                        "torch_profiler_with_flops": False,
                        "torch_profiler_use_gzip": True,
                    },
                )
                self.assertEqual(ServerConfig.from_env(config.environment()), config)

    def test_empty_profile_environment_keeps_profiling_disabled(self):
        for value in ("", "  "):
            with self.subTest(value=value):
                config = ServerConfig.from_env(
                    {
                        PROFILE_SESSION_ENV: value,
                    }
                )
                self.assertEqual(config, ServerConfig())
                self.assertNotIn("--profiler-config", config.command())

    def test_invalid_profile_sessions_fail_before_deployment(self):
        for value in (
            "../escape",
            "/tmp/traces",
            "C:\\traces",
            "a/b",
            "a\\b",
            "a b",
            "a.b",
            "-off",
            "é",
            "a" * 65,
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, PROFILE_SESSION_ENV),
            ):
                ServerConfig.from_env(
                    {
                        PROFILE_SESSION_ENV: value,
                    }
                )

        for value in (False, 42, "", " padded "):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, PROFILE_SESSION_ENV),
            ):
                ServerConfig(profile_session=value)

        self.assertEqual(
            ServerConfig(profile_session="A" * 64).profile_session, "A" * 64
        )


if __name__ == "__main__":
    unittest.main()
