Relevancy Interfaces
--------------------


RankRelevancy
+++++++++++++

This interface defines one method: ``rank``.  The ``rank`` method
takes examples of relevant and not-relevant example descriptor vectors
as :class:`numpy.ndarray` sequences and uses them to compute relevancy
scores (on a ``[0, 1]`` scale) on a provided pool of other descriptor
vectors.

.. autoclass:: smqtk_relevancy.interfaces.rank_relevancy.RankRelevancy
   :members:


RankRelevancyWithFeedback
+++++++++++++++++++++++++

This interface defines one method: ``rank_with_feedback``.  Like
:meth:`RankRelevancy.rank`, ``rank_with_feedback`` takes examples of
relevant and not-relevant example descriptor vectors as
:class:`numpy.ndarray` sequences and uses them to compute relevancy
scores (on a ``[0, 1]`` scale) on a provided pool of other descriptor
vectors.  However, it also expects a sequence of corresponding UIDs
for the pool vectors and additionally returns a sequence of UIDs,
possibly not all from the pool, on which feedback would be most
useful.

.. autoclass:: smqtk_relevancy.interfaces.rank_relevancy.RankRelevancyWithFeedback
   :members:
   :private-members:


RelevancyIndex
++++++++++++++

**[Deprecated]
Please use RankRelevancy instead of RelevancyIndex**

This interface defines two methods: ``build_index`` and ``rank``.
The ``build_index`` method is, like a ``NearestNeighborsIndex``, used to build an index of ``DescriptorElement`` instances.
The ``rank`` method takes examples of relevant and not-relevant ``DescriptorElement`` examples with which the algorithm uses to rank (think sort) the indexed ``DescriptorElement`` instances by relevancy (on a ``[0, 1]`` scale).

.. autoclass::smqtk_relevancy.interfaces.relevancy_index.RelevancyIndex
   :members:
