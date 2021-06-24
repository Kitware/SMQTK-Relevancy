import abc
from typing import Iterable, Dict
from warnings import warn

from smqtk_core import Plugfigurable
from smqtk_descriptors import DescriptorElement


warn("Relevancy Index is deprecated and may be removed in a future release."
     "Please use Rank Relevancy instead.")


class NoIndexError (Exception):
    """
    When a RelevancyIndex does not yet have a index built yet when one is
    needed.
    """


class RelevancyIndex (Plugfigurable):
    """
    Abstract class for IQR index implementations.

    Similar to a traditional nearest-neighbors algorithm, An IQR index provides
    a specialized nearest-neighbors interface that can take multiple examples of
    positively and negatively relevant exemplars in order to produce a [0, 1]
    ranking of the indexed elements by determined relevancy.

    """

    def __len__(self) -> int:
        return self.count()

    @abc.abstractmethod
    def count(self) -> int:
        """
        :return: Number of elements in this index.
        :rtype: int
        """

    @abc.abstractmethod
    def build_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Build the index based on the given iterable of descriptor elements.

        Subsequent calls to this method should rebuild the index, not add to it.

        :raises ValueError: No data available in the given iterable.

        :param descriptors: Iterable of descriptor elements to build index over.
        :type descriptors:
            collections.abc.Iterable[smqtk.representation.DescriptorElement]

        """

    @abc.abstractmethod
    def rank(self, pos: Iterable[DescriptorElement],
             neg: Iterable[DescriptorElement]) -> Dict[DescriptorElement, float]:
        """
        Rank the currently indexed elements given ``pos`` positive and ``neg``
        negative exemplar descriptor elements.

        :param pos: Iterable of positive exemplar DescriptorElement instances.
            This may be optional for some implementations.
        :type pos: collections.abc.Iterable[smqtk.representation.DescriptorElement]

        :param neg: Iterable of negative exemplar DescriptorElement instances.
            This may be optional for some implementations.
        :type neg: collections.abc.Iterable[smqtk.representation.DescriptorElement]

        :raises NoIndexError:
            If index ranking is requested without an index to rank.

        :return: Map of indexed descriptor elements to a rank value between
            [0, 1] (inclusive) range, where a 1.0 means most relevant and 0.0
            meaning least relevant.
        :rtype: dict[smqtk.representation.DescriptorElement, float]

        """
