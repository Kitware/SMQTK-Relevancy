import numpy
from typing import Dict, List, Sequence, Iterable

from smqtk_relevancy.interfaces.rank_relevancy import RankRelevancy
from smqtk_relevancy.impls.rank_relevancy.sorted_sampling import (
    RankRelevancyWithSortedFeedback,
)
from smqtk_descriptors import DescriptorElement


def test_is_usable() -> None:
    assert RankRelevancyWithSortedFeedback.is_usable()


class DummyRankRelevancy(RankRelevancy):
    def get_config(self) -> Dict: ...

    def rank(self, pos: Iterable[DescriptorElement], neg: Iterable[DescriptorElement], pool: Sequence) -> List:
        return [v[0] for v in pool]


def make_sorted_ranker(n: int) -> RankRelevancyWithSortedFeedback:
    return RankRelevancyWithSortedFeedback(
        DummyRankRelevancy(), n
    )


def test_parameter_n() -> None:
    """
    Check that the "n" parameter has the expected effect on feedback
    request count
    """
    n = 10
    mr = make_sorted_ranker(n)
    for i in range(3, 31, 3):
        pool = numpy.linspace(0, 1, i)[:, numpy.newaxis]
        pool_uids = [object() for _ in range(i)]
        scores, requests = mr.rank_with_feedback([], [], pool, pool_uids)
        assert len(requests) == min(n, i)


def test_pass_through() -> None:
    """
    Check that scores from the wrapped RankRelevancy are passed
    through regardless of the `n` value.
    """
    mr = make_sorted_ranker(3)
    pool = [[.3], [.1], [.45], [.29], [.03]]
    expected = [.3, .1, .45, .29, .03]
    uids = [object() for _ in pool]
    scores, requests = mr.rank_with_feedback([], [], pool, uids)
    assert list(scores) == expected


def test_descending_feedback() -> None:
    """
    Check that feedback results are returned in correct descending
    order based on scores from the wrapped RankRelevancy.
    """
    mr = make_sorted_ranker(3)
    pool = [[.3], [.1], [.45], [.29], [.03]]
    uids = [object() for _ in pool]
    ranks = numpy.argsort(pool, axis=0)[::-1].ravel()
    expected = [uids[i] for i in ranks[:mr._n]]
    scores, requests = mr.rank_with_feedback([], [], pool, uids)
    assert list(requests) == expected
