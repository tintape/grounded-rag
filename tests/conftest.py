from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Build a small PDF where each item of `pages` is the text of one page ("" = blank)."""

    def _make(name: str, pages: list[str], header: str | None = None) -> Path:
        doc = pymupdf.open()
        for number, text in enumerate(pages, start=1):
            page = doc.new_page()
            if header:
                page.insert_text((50, 30), header, fontsize=8)
            if text:
                page.insert_textbox(pymupdf.Rect(50, 60, 550, 760), text, fontsize=11)
            if header:
                page.insert_text((50, 800), f"Page {number} of {len(pages)}", fontsize=8)
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return _make
