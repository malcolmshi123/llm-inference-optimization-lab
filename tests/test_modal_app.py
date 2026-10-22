import json
import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bench.cache_study import startup_command
from server_config import PROFILE_ROOT, PROFILE_SESSION_ENV, PROFILE_VOLUME_NAME

class ModalAppTests(unittest.TestCase):
    def load_app(self, environment):
        modal = MagicMock()
        modal.App.return_value.function.return_value = lambda function: function
        modal.concurrent.return_value = lambda function: function
        modal.web_server.return_value = lambda function: function
        image = modal.Image.from_registry.return_value

        for method in (
            "entrypoint",
            "uv_pip_install",
            "env",
            "add_local_python_source",
        ):
            getattr(image, method).return_value = image

        with (
            patch.dict(
                "sys.modules",
                {
                    "modal": modal,
                },
            ),
            patch.dict("os.environ", environment, clear=True),
        ):
            namespace = runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "modal_app.py")
            )

        return modal, namespace

    def test_default_deployment_keeps_only_the_existing_cache_mounts(self):
        modal, namespace = self.load_app({})
        mounts = modal.App.return_value.function.call_args.kwargs["volumes"]
        self.assertEqual(set(mounts), {"/root/.cache/huggingface", "/root/.cache/vllm"})
        self.assertNotIn(
            PROFILE_VOLUME_NAME,
            [call.args[0] for call in modal.Volume.from_name.call_args_list],
        )
        self.assertNotIn(
            PROFILE_SESSION_ENV,
            modal.Image.from_registry.return_value.env.call_args.args[0],
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("subprocess.Popen") as start,
            patch("pathlib.Path.mkdir") as mkdir,
            patch("builtins.print"),
        ):
            namespace["serve"]()

        mkdir.assert_not_called()
        command = start.call_args.args[0]
        self.assertNotIn("--profiler-config", command)
        log = "Starting vLLM server with command: " + " ".join(command)
        self.assertEqual(startup_command(log, True), command)

    def test_profile_deployment_mounts_and_targets_persistent_storage(self):
        environment = {
            PROFILE_SESSION_ENV: "shared-c8-off-01",
        }
        modal, namespace = self.load_app(environment)
        mounts = modal.App.return_value.function.call_args.kwargs["volumes"]
        self.assertIn(PROFILE_ROOT, mounts)
        modal.Volume.from_name.assert_any_call(
            PROFILE_VOLUME_NAME, create_if_missing=True
        )
        image_environment = modal.Image.from_registry.return_value.env.call_args.args[0]
        self.assertEqual(image_environment[PROFILE_SESSION_ENV], "shared-c8-off-01")

        with (
            patch.dict("os.environ", environment, clear=True),
            patch("subprocess.Popen") as start,
            patch("pathlib.Path.mkdir") as mkdir,
            patch("builtins.print"),
        ):
            namespace["serve"]()

        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        command = start.call_args.args[0]
        profiler = json.loads(command[command.index("--profiler-config") + 1])
        self.assertEqual(profiler["torch_profiler_dir"], "/profiles/shared-c8-off-01")

        with self.assertRaisesRegex(ValueError, "fixed cache-study controls"):
            startup_command(
                "Starting vLLM server with command: " + " ".join(command), True
            )

if __name__ == "__main__":
    unittest.main()
