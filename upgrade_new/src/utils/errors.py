"""Application-specific exceptions with user-safe messages."""


class UpgradeError(Exception):
    """Base class for expected upgrade errors."""


class ConfigError(UpgradeError):
    """Raised when required configuration is missing or invalid."""


class PdfLoadError(UpgradeError):
    """Raised when a PDF cannot be read or has no extractable text."""


class NotionError(UpgradeError):
    """Raised when Notion loading or parsing fails."""


class EmbeddingError(UpgradeError):
    """Raised when embedding generation fails."""


class VectorStoreError(UpgradeError):
    """Raised when ChromaDB operations fail."""


class GenerationError(UpgradeError):
    """Raised when answer generation fails."""


def clean_error_message(error: Exception) -> str:
    """Return a short message suitable for Streamlit UI display."""
    message = str(error).replace("\n", " ").strip()
    return message[:500] if message else error.__class__.__name__
