from unittest import mock
from typing import Dict, Any, Tuple, Sequence, Hashable

import numpy
import pytest

from smqtk_relevancy.interfaces.rank_relevancy import RankRelevancyWithFeedback


class DummyRRWF(RankRelevancyWithFeedback):
    def __init__(self) -> None:
        # mock object to be called
        self._rwf_mock = mock.Mock()

    # Implement RankRelevancyWithFeedback._rank_with_feedback using a
    # per-instance Mock.  self._rank_with_feedback is thus a Mock
    # instead of a bound method.
    def _rank_with_feedback(self,
                            pos: Sequence[numpy.ndarray],
                            neg: Sequence[numpy.ndarray],
                            pool: Sequence[numpy.ndarray],
                            pool_uids: Sequence[Hashable],) -> Tuple[Sequence[float], Sequence[Hashable]]:
        return self._rwf_mock(pos, neg, pool, pool_uids)

    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError


def test_rrwf_length_check() -> None:
    """
    Check that :meth:`RankRelevancyWithFeedback.rank_with_feedback`
    raises the documented :class:`ValueError` when pool and UID list
    length don't match
    """
    rrwf = DummyRRWF()
    with pytest.raises(ValueError):
        v = numpy.ones(16)
        rrwf.rank_with_feedback(
            [v] * 4, [v] * 4, [v] * 10,
            # Not 10 UIDs
            ['a', 'b', 'c', 'd'],
        )
    rrwf._rwf_mock.assert_not_called()


def test_rrwf_calls_impl_method() -> None:
    """
    Check that :meth:`RankRelevancyWithFeedback.rank_with_feedback`
    delegates to :meth:`RankRelevancyWithFeedback._rank_with_feedback`
    """
    rrwf = DummyRRWF()
    v = numpy.ones(16)
    args = (
        [v] * 4, [v] * 4, [v] * 10,
        # 10 UIDs
        list('abcdefghij'),
    )
    result = rrwf.rank_with_feedback(*args)
    rrwf._rwf_mock.assert_called_once_with(*args)
    assert result is rrwf._rwf_mock.return_value
