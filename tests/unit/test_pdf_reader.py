from pathlib import Path

import pymupdf

from grounded_rag.ingest.pdf_reader import clean_text, read_pdf
from grounded_rag.models import FileStatus

LONG = "The robot must be cleaned weekly to keep suction strong and quiet."


def test_reads_each_page_with_one_based_numbers(make_pdf):
    result = read_pdf(make_pdf("guide.pdf", [f"First page. {LONG}", f"Second page. {LONG}"]))
    assert result.status is FileStatus.LOADED
    assert [p.page for p in result.pages] == [1, 2]
    assert result.pages[1].text.startswith("Second page.")
    assert all(p.doc_name == "guide.pdf" for p in result.pages)


def test_blank_pages_are_skipped_but_counted(make_pdf):
    result = read_pdf(make_pdf("gap.pdf", [f"Intro. {LONG}", "", f"Outro. {LONG}"]))
    assert result.page_count == 3
    assert [p.page for p in result.pages] == [1, 3]


def test_image_only_pdf_reported_as_no_text(make_pdf, tmp_path: Path):
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    page.insert_image(pymupdf.Rect(50, 50, 200, 200), pixmap=pix)
    path = tmp_path / "scan.pdf"
    doc.save(path)

    result = read_pdf(path)
    assert result.status is FileStatus.SKIPPED_NO_TEXT
    assert result.pages == ()


def test_corrupt_file_reported_as_failed(tmp_path: Path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf at all")
    result = read_pdf(path)
    assert result.status is FileStatus.FAILED
    assert result.error


def test_encrypted_pdf_reported_as_failed(tmp_path: Path):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), LONG)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    result = read_pdf(path)
    assert result.status is FileStatus.FAILED
    assert "password" in result.error.lower()


def test_repeated_headers_and_footers_are_removed(make_pdf):
    pages = [f"Topic {n}. {LONG}" for n in range(1, 4)]
    result = read_pdf(make_pdf("doc.pdf", pages, header="ACME Internal Handbook"))
    for page in result.pages:
        assert "ACME Internal Handbook" not in page.text
        assert "Page" not in page.text.split()[0]
        assert "of 3" not in page.text


def test_clean_text_normalises_ligatures_hyphenation_and_spaces():
    raw = "Speciﬁcation  of the   ﬂoor\nclean-\ning cycle\n\n\n  next line  "
    assert clean_text(raw) == "Specification of the floor\ncleaning cycle\nnext line"
