import numpy
from typing import Dict, List, Sequence, Iterable

from smqtk_relevancy.interfaces.rank_relevancy import RankRelevancy
from smqtk_relevancy.impls.rank_relevancy.random_sampling import (
    RankRelevancyWithRandomFeedback,
)
from smqtk_descriptors import DescriptorElement


def test_is_usable() -> None:
    assert RankRelevancyWithRandomFeedback.is_usable()


class DummyRankRelevancy(RankRelevancy):
    def get_config(self) -> Dict: ...

    def rank(self, pos: Iterable[DescriptorElement], neg: Iterable[DescriptorElement], pool: Sequence) -> List:
        return [v[0] for v in pool]


def make_random_ranker(n: int, seed: int = 0) -> RankRelevancyWithRandomFeedback:
    return RankRelevancyWithRandomFeedback(
        DummyRankRelevancy(), n, seed
    )


def test_parameter_n() -> None:
    """
    Check that the "n" parameter has the expected effect on feedback
    request count
    """
    n = 10
    mr = make_random_ranker(n)
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
    mr = make_random_ranker(3)
    pool = [[.3], [.1], [.45], [.29], [.03]]
    expected = [.3, .1, .45, .29, .03]
    uids = [object() for _ in pool]
    scores, requests = mr.rank_with_feedback([], [], pool, uids)
    assert list(scores) == expected


def test_parameter_seed() -> None:
    """
    Check that the "seed" parameter has the expected effect on the
    choice of feedback requests
    """
    pool = [[i ** 2 / 100] for i in range(11)]
    uids = 'abcdefghijk'
    seed = [1, 2]
    expected = ['cj', 'ab']  # [(.04, .01), (.16, .25)]
    for s, e in zip(seed, expected):
        mr = make_random_ranker(2, s)
        scores, requests = mr.rank_with_feedback([], [], pool, uids)
        assert list(requests) == list(e)
