"""Future deterministic prompt-to-constraints parser, outside reward evaluation."""

from typing import Any

from marketcanvas_env.models import RequestError


def parse_prompt(prompt: str) -> list[dict[str, Any]]:
    """Translate prose into constraint dictionaries; parsing is not implemented yet.

    Callers should currently supply the constraints authored in their task YAML.
    No prompt lookup or fallback to an unrelated task is performed.
    """
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 1024 or not prompt.strip():
        raise RequestError("invalid_request", "prompt must be a nonempty string up to 1024")
    raise NotImplementedError(
        "Prompt parsing is not implemented; supply constraints from task YAML"
    )
