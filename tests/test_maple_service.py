# Tests for the Maple model registry and the OpenAI-compatible server aliases.
#
# Run with:
#   python -m pytest tests/test_maple_service.py -v
# (No model weights are required; model loading is stubbed.)

import argparse
import http
import threading
import unittest
from pathlib import Path

import requests

from mlx_lm.server import (
    APIHandler,
    LRUPromptCache,
    ModelProvider,
    ResponseGenerator,
    _load_model_registry,
)

MAPLE_REGISTRY = Path(__file__).resolve().parent.parent / "maple_models.json"


class RecordingModelProvider(ModelProvider):
    """Subclass of ModelProvider that records loads instead of loading weights."""

    def __init__(self, cli_args):
        self.loads = []
        ModelProvider.__init__(self, cli_args)

    def _load(
        self,
        model_path,
        adapter_path=None,
        draft_model_path=None,
        model_config=None,
    ):
        self.loads.append((model_path, adapter_path, draft_model_path, model_config))
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = "fake-model"
        self.tokenizer = "fake-tokenizer"
        self._loaded_model_config = model_config


def make_cli_args(**overrides):
    args = argparse.Namespace(
        model="deepgrove/maple-preview-2bit-mlx",
        adapter_path=None,
        draft_model=None,
        model_registry=str(MAPLE_REGISTRY),
        flash_head=None,
        trust_remote_code=True,
        chat_template="",
        use_default_chat_template=False,
        allowed_origins=["*"],
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestModelRegistry(unittest.TestCase):
    def test_loads_repo_registry(self):
        registry = _load_model_registry(str(MAPLE_REGISTRY))
        self.assertIn("maple-preview", registry)
        self.assertIn("maple-preview-flash", registry)
        self.assertEqual(registry["maple-preview"]["model"], "./maple-2bit-mlx")
        self.assertIs(registry["maple-preview"]["flash_head"], False)
        self.assertIs(registry["maple-preview-flash"]["flash_head"], True)

    def test_missing_registry_is_empty(self):
        self.assertEqual(_load_model_registry(None), {})

    def test_invalid_registry_raises(self, tmp_path=None):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("[1, 2, 3]")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                _load_model_registry(path)
        finally:
            Path(path).unlink()


class TestPerModelFlashHead(unittest.TestCase):
    def test_alias_resolves_flash_head_config(self):
        provider = RecordingModelProvider(make_cli_args())
        self.assertEqual(
            provider._model_config_for("maple-preview"), {"use_flash_head": False}
        )
        self.assertEqual(
            provider._model_config_for("maple-preview-flash"),
            {"use_flash_head": True},
        )
        # Unknown/alias-less requests fall back to the global --flash-head.
        self.assertEqual(provider._model_config_for("default_model"), {})

    def test_global_flash_head_fallback(self):
        provider = RecordingModelProvider(
            make_cli_args(model_registry=None, flash_head=True)
        )
        self.assertEqual(
            provider._model_config_for("some-model"), {"use_flash_head": True}
        )

    def test_switch_between_variants_reloads_model(self):
        provider = RecordingModelProvider(make_cli_args())
        provider.load("maple-preview")
        provider.load("maple-preview")
        # Same checkpoint, different head config -> must reload.
        provider.load("maple-preview-flash")
        provider.load("maple-preview-flash")
        # And back again.
        provider.load("maple-preview")

        self.assertEqual(len(provider.loads), 3)
        exact, flash, exact_again = provider.loads
        for load in (exact, flash, exact_again):
            self.assertEqual(load[0], "./maple-2bit-mlx")
        self.assertEqual(exact[3], {"use_flash_head": False})
        self.assertEqual(flash[3], {"use_flash_head": True})
        self.assertEqual(exact_again[3], {"use_flash_head": False})


class RegistryProvider:
    """Minimal model provider exposing the registry for /v1/models."""

    def __init__(self):
        self.cli_args = make_cli_args()
        self._model_registry = _load_model_registry(str(MAPLE_REGISTRY))

    def load_default(self):
        pass

    def load(self, model, adapter=None, draft_model=None):
        return None, None


class TestModelsEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(RegistryProvider(), LRUPromptCache())
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_models_lists_registry_aliases(self):
        response = requests.get(f"http://localhost:{self.port}/v1/models")
        self.assertEqual(response.status_code, 200)
        model_ids = [m["id"] for m in response.json()["data"]]
        self.assertIn("maple-preview", model_ids)
        self.assertIn("maple-preview-flash", model_ids)


if __name__ == "__main__":
    unittest.main()
