"""Neutralise model output before it reaches a Markdown renderer.

Document text is untrusted: a planted instruction could make the model emit a Markdown image or
link that leaks data when rendered. Answers never need links, images or HTML, so instead of trying
to recognise every link syntax (nested brackets, angle-bracket URLs, reference style), we remove
the parts that make any of them work: inline destinations, reference definitions and tags.
"""

import re

_REFERENCE_DEFINITION = re.compile(r"^[ \t]*\[[^\]]+\]:\s*\S+[^\n]*$", re.MULTILINE)
_INLINE_DESTINATION = re.compile(r"\]\([^)]*\)+")
_HTML_TAG = re.compile(r"<[^>]*>")
_IMAGE_MARK = re.compile(r"!\[")
_NON_CITATION_BRACKETS = re.compile(r"\[(?!\d+(?:\s*,\s*\d+)*\])([^\[\]]*)\]")
_SPACE_RUNS = re.compile(r"[ \t]{2,}")
_BLANK_RUNS = re.compile(r"\n{3,}")


def sanitize_answer(text: str) -> str:
    text = _REFERENCE_DEFINITION.sub("", text)
    text = _INLINE_DESTINATION.sub("]", text)
    text = _HTML_TAG.sub("", text)
    text = _IMAGE_MARK.sub("[", text)
    previous = None
    while previous != text:  # unwrap nested [..[..]..] link texts from the inside out
        previous, text = text, _NON_CITATION_BRACKETS.sub(r"\1", text)
    text = "\n".join(_SPACE_RUNS.sub(" ", line).strip() for line in text.splitlines())
    return _BLANK_RUNS.sub("\n\n", text).strip()
