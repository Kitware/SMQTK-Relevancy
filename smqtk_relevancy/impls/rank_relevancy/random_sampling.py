from typing import Hashable, Sequence, Tuple, Dict, Any, TypeVar, Type

import random
from numpy import ndarray

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from smqtk_core.dict import merge_dict
from smqtk_relevancy.interfaces.rank_relevancy import RankRelevancy, RankRelevancyWithFeedback

T = TypeVar("T", bound="RankRelevancyWithRandomFeedback")


class RankRelevancyWithRandomFeedback(RankRelevancyWithFeedback):
    """
    Wrap an instance of :class:`RankRelevancy` to provide random feedback

    :param rank_relevancy: :class:`RankRelevancy` to use for computing
        relevancy scores
    :param n: Maximum number of items to return for feedback
    :param seed: Seed value for random number generation to ensure reproducibility
        of feedback results (default: 0)

    :raises ValueError: n is negative

    """

    def __init__(self, rank_relevancy: RankRelevancy,
                 n: int, seed: int = 0):
        self._rank_relevancy = rank_relevancy
        if n < 0:
            raise ValueError(f"n must be nonnegative but got {n}")
        self._n = n
        self._seed = seed
        random.seed(seed)

    def _rank_with_feedback(
            self,
            pos: Sequence[ndarray],
            neg: Sequence[ndarray],
            pool: Sequence[ndarray],
            pool_uids: Sequence[Hashable],
    ) -> Tuple[Sequence[float], Sequence[Hashable]]:
        scores = self._rank_relevancy.rank(pos, neg, pool)
        ranked = random.sample(list(zip(scores, pool_uids)), len(scores))
        return scores, [r[1] for r in ranked[:self._n]]

    @classmethod
    def from_config(cls: Type[T], config_dict: Dict[str, Any], merge_default: bool = True) -> T:
        config_dict = dict(config_dict, rank_relevancy=from_config_dict(
            config_dict['rank_relevancy'], RankRelevancy.get_impls(),
        ))
        return super().from_config(config_dict, merge_default=merge_default)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        c = super().get_default_config()
        rr_default = make_default_config(RankRelevancy.get_impls())
        return dict(c, rank_relevancy=rr_default)

    def get_config(self) -> Dict[str, Any]:
        return merge_dict(self.get_default_config(), dict(
            rank_relevancy=to_config_dict(self._rank_relevancy),
            n=self._n,
            seed=self._seed,
        ))
