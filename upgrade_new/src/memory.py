"""Chat memory placeholder.

Follow-up question condensation and short-term chat memory are planned for a
later milestone.
"""


class ShortTermMemory:
    """Placeholder memory interface."""

    def add(self, role: str, content: str) -> None:
        raise NotImplementedError("Chat memory is planned for a later milestone.")

    def recent(self) -> list[dict[str, str]]:
        raise NotImplementedError("Chat memory is planned for a later milestone.")

