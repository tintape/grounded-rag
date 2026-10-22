"""Neutralise model output before it reaches a Markdown renderer.

Document text is untrusted: a planted instruction could make the model emit a Markdown image or
link that leaks data when rendered. Two layers:

- `sanitize_answer` (server side, every API client): strips link destinations, reference
  definitions, HTML and image markers, repeated until stable. Best effort.
- `to_safe_markdown` (at render time): an allowlist. Every Markdown control character is
  backslash-escaped (so [1] still displays as [1] but can never form a link or reference), and an
  invisible zero-width space is inserted into `://`, `www.` and `@` so GFM cannot autolink bare
  URLs or e-mails (character references don't help: renderers decode them first). Only
  plain text, bold and list numbering survive, whatever syntax the model produced.
"""

import re

_REFERENCE_DEFINITION = re.compile(r"^[ \t>*+\-\d.]*\[[^\]]+\]:\s*\S+[^\n]*$", re.MULTILINE)
_INLINE_DESTINATION = re.compile(r"\]\([^)]*\)+")
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>\n]*>")  # real tags only, not "<5 Nm ... >2 Nm"
_IMAGE_MARK = re.compile(r"!+\[")
_SPACE_RUNS = re.compile(r"[ \t]{2,}")
_BLANK_RUNS = re.compile(r"\n{3,}")
_MAX_PASSES = 10
_FAIL_CLOSED = re.compile(r"[\[\]()<>!]")  # no structure can survive; citations go too

_MARKDOWN_CONTROL = re.compile(r"([\\`\[\]()<>!#|~_&])")
_ZERO_WIDTH_SPACE = "\u200b"
_AUTOLINK_TRIGGER = re.compile(r"(:)(//)|(w)(ww\.)|(@)", re.IGNORECASE)


def _sanitize_once(text: str) -> str:
    text = _REFERENCE_DEFINITION.sub("", text)
    text = _INLINE_DESTINATION.sub("]", text)
    text = _HTML_TAG.sub("", text)
    return _IMAGE_MARK.sub("[", text)


def sanitize_answer(text: str) -> str:
    for _ in range(_MAX_PASSES):
        cleaned = _sanitize_once(text)
        if cleaned == text:
            break
        text = cleaned
    else:  # still changing after the cap: adversarial input. Strip every structural character
        # in one linear pass; with no citations left, the engine turns the answer into a refusal.
        text = _FAIL_CLOSED.sub("", text)
    text = "\n".join(_SPACE_RUNS.sub(" ", line).strip() for line in text.splitlines())
    return _BLANK_RUNS.sub("\n\n", text).strip()


def to_safe_markdown(text: str) -> str:
    """Escape all link/image/HTML/code syntax; keep bold, lists and citation markers."""
    escaped = _MARKDOWN_CONTROL.sub(r"\\\1", text)
    return _AUTOLINK_TRIGGER.sub(_break_autolink, escaped)


def _break_autolink(match: re.Match[str]) -> str:
    parts = [part for part in match.groups() if part]
    if len(parts) == 1:  # "@": break before it
        return _ZERO_WIDTH_SPACE + parts[0]
    return parts[0] + _ZERO_WIDTH_SPACE + parts[1]
