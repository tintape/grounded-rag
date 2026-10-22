from grounded_rag.rag.citations import cited_indices, rewrite_markers


def test_extracts_unique_indices_in_order_of_first_use():
    assert cited_indices("A [2]. B [1][2]. C [3, 1].", n_passages=3) == (2, 1, 3)


def test_ignores_out_of_range_and_zero():
    assert cited_indices("A [0]. B [4]. C [2].", n_passages=3) == (2,)


def test_no_markers_means_no_citations():
    assert cited_indices("Plain answer with no references.", n_passages=3) == ()


def test_does_not_treat_other_brackets_as_citations():
    assert cited_indices("Press [Home] then see [a].", n_passages=3) == ()


def test_rewrite_renumbers_and_drops_invalid():
    text = "Reset [3]. Battery [1, 9]. Bogus [7]."
    assert rewrite_markers(text, {3: 1, 1: 2}) == "Reset [1]. Battery [2]. Bogus."


def test_rewrite_merges_duplicates_inside_one_marker():
    assert rewrite_markers("Both [1, 2].", {1: 1, 2: 1}) == "Both [1]."
