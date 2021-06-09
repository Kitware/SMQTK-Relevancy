from __future__ import division, print_function
import unittest

import unittest.mock as mock

from smqtk_relevancy.interfaces.relevancy_index import RelevancyIndex
from smqtk_descriptors import DescriptorElement

from typing import Dict, Any, Iterable


class DummyRI (RelevancyIndex):

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def rank(self, pos: Iterable[DescriptorElement],
             neg: Iterable[DescriptorElement]) -> Dict[DescriptorElement, float]:
        pass

    def get_config(self) -> Dict[str, Any]:
        pass

    def count(self) -> int:
        return 0

    def build_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        pass


class TestSimilarityIndexAbstract (unittest.TestCase):

    def test_count(self) -> None:
        index = DummyRI()
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.count(), len(index))

        # Pretend that there were things in there. Len should pass it though
        index.count = mock.Mock()  # type: ignore
        index.count.return_value = 5
        self.assertEqual(len(index), 5)
