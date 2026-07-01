"""Smoke tests for the release repository structure."""

import importlib
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SprintOneSmokeTests(unittest.TestCase):
    """Validate the baseline/upgrade_new split and importable upgrade app."""

    def test_required_folders_exist(self) -> None:
        """Check the required top-level project folders."""
        for folder in ["baseline", "upgrade_new", "docs", "tests"]:
            self.assertTrue((ROOT / folder).exists(), folder)

    def test_baseline_app_exists(self) -> None:
        """Check that the baseline Streamlit app is present."""
        self.assertTrue((ROOT / "baseline" / "baseline_chatbot_app.py").exists())

    def test_upgrade_new_app_exists(self) -> None:
        """Check that the deployable Streamlit app is present."""
        self.assertTrue((ROOT / "upgrade_new" / "app.py").exists())

    def test_upgrade_new_modules_import(self) -> None:
        """Check that core upgrade_new modules are importable."""
        for module_name in [
            "upgrade_new.src.config",
            "upgrade_new.src.chunker",
            "upgrade_new.src.embeddings",
            "upgrade_new.src.vector_store",
            "upgrade_new.src.retriever",
            "upgrade_new.src.rag_chain",
            "upgrade_new.src.prompts",
        ]:
            importlib.import_module(module_name)

    def test_chunker_sample_text(self) -> None:
        """Check that chunk_text runs with sample text."""
        from upgrade_new.src.chunker import chunk_text

        self.assertEqual(chunk_text("hello world", size=100, overlap=0), ["hello world"])

    def test_app_title(self) -> None:
        """Check the visible app name."""
        from upgrade_new.src import config

        self.assertEqual(config.APP_TITLE, "Tank Tank Bot")


if __name__ == "__main__":
    unittest.main()
