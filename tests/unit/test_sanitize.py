from grounded_rag.rag.sanitize import sanitize_answer


def test_removes_markdown_images_entirely():
    text = "Reset it [1]. ![x](https://evil.example/leak?q=secret)"
    assert sanitize_answer(text) == "Reset it [1]."


def test_turns_links_into_plain_text():
    assert sanitize_answer("See [the guide](https://evil.example) [2].") == "See the guide [2]."


def test_strips_raw_html_tags():
    assert sanitize_answer('Hi <img src="https://evil.example/x"> there [1].') == "Hi there [1]."


def test_keeps_citations_lists_and_bold():
    text = "Steps:\n1. Hold **Home** [1]\n2. Wait [1, 2]"
    assert sanitize_answer(text) == text
