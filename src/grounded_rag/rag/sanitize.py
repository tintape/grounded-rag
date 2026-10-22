"""Neutralise model output before it reaches a Markdown renderer.

Document text is untrusted: a planted instruction could make the model emit a Markdown image or
link that leaks data when rendered. Answers never need links or images, so they are removed.
"""

import re

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPACE_RUNS = re.compile(r"[ \t]{2,}")


def sanitize_answer(text: str) -> str:
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    return "\n".join(_SPACE_RUNS.sub(" ", line).strip() for line in text.splitlines())
