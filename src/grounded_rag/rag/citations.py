"""Parse and rewrite numbered citation markers like [1] or [2, 3] in model output."""

import re

_MARKER = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


def _numbers(group: str) -> list[int]:
    return [int(part) for part in group.split(",")]


def cited_indices(text: str, n_passages: int) -> tuple[int, ...]:
    """Valid 1-based passage numbers cited in `text`, unique, in order of first use."""
    seen: dict[int, None] = {}
    for match in _MARKER.finditer(text):
        for number in _numbers(match.group(1)):
            if 1 <= number <= n_passages:
                seen.setdefault(number, None)
    return tuple(seen)


def has_invalid_markers(text: str, n_passages: int) -> bool:
    """True if any marker cites a passage number that was not provided (a fabricated source)."""
    return any(
        not 1 <= number <= n_passages
        for match in _MARKER.finditer(text)
        for number in _numbers(match.group(1))
    )


def rewrite_markers(text: str, mapping: dict[int, int]) -> str:
    """Renumber markers via `mapping` (passage -> source number); drop unmapped numbers."""

    def _replace(match: re.Match[str]) -> str:
        mapped = dict.fromkeys(mapping[n] for n in _numbers(match.group(1)) if n in mapping)
        if not mapped:
            return ""
        return " [" + ", ".join(str(n) for n in mapped) + "]"

    return _MARKER.sub(_replace, text).strip()
