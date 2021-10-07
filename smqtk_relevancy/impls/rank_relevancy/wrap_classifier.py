from typing import Sequence, Dict, Any, TypeVar, Type

import numpy as np

from smqtk_classifier import ClassifyDescriptorSupervised
from smqtk_descriptors import DescriptorElement
from smqtk_descriptors.impls.descriptor_element.memory import DescriptorMemoryElement
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    cls_conf_to_config_dict,
)

from smqtk_relevancy.interfaces.rank_relevancy import RankRelevancy


T = TypeVar("T", bound="RankRelevancyWithSupervisedClassifier")


class RankRelevancyWithSupervisedClassifier(RankRelevancy):
    """
    Relevancy ranking that utilizes a usable supervised classifier for
    on-the-fly training and inference.

    While the name of this class merely states "supervised classifier," we
    specifically utilize the interface for descriptor classification as opposed
    to the interfaces for other modalities (like images).

    # Classifier "cloning"
    The input supervised classifier instance to the constructor is not directly
    used, but its type and configuration are recorded in order to create a new
    instance in ``rank`` to train and classify the index.

    The caveat here is that any non-configuration reflected, runtime
    modifications to the input classifier will not be reflected by the
    classifier used in ``rank``.

    Using a copy of the input classifier allows the ``rank`` method to be used
    in parallel without blocking other calls to ``rank``.

    :param classifier_inst:
        Supervised classifier instance to base the ephemeral ranking classifier
        on. The type and configuration of this classifier is used to create a
        clone at rank time. The input classifier instance is not modified.
    """

    def __init__(self, classifier_inst: ClassifyDescriptorSupervised):
        super().__init__()
        self._classifier_type = type(classifier_inst)
        self._classifier_config = classifier_inst.get_config()

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        c = super().get_default_config()
        c['classifier_inst'] = \
            make_default_config(ClassifyDescriptorSupervised.get_impls())
        return c

    @classmethod
    def from_config(cls: Type[T], config_dict: Dict[str, Any], merge_default:
                    bool = True) -> T:
        config_dict = dict(config_dict)  # shallow copy to write to input dict
        config_dict['classifier_inst'] = \
            from_config_dict(config_dict.get('classifier_inst', {}),
                             ClassifyDescriptorSupervised.get_impls())
        return super(RankRelevancyWithSupervisedClassifier, cls).from_config(
            config_dict, merge_default=merge_default,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            'classifier_inst':
                cls_conf_to_config_dict(self._classifier_type,
                                        self._classifier_config),
        }

    def rank(
            self,
            pos: Sequence[np.ndarray],
            neg: Sequence[np.ndarray],
            pool: Sequence[np.ndarray],
    ) -> Sequence[float]:
        if len(pool) == 0:
            return []

        # Train supervised classifier with positive/negative examples.
        label_pos = 'pos'
        label_neg = 'neg'

        i = 0

        def create_de(v: np.ndarray) -> DescriptorElement:
            nonlocal i
            # Hopefully type_str doesn't matter
            de = DescriptorMemoryElement(i)
            de.set_vector(v)
            i += 1
            return de

        classifier = self._classifier_type.from_config(self._classifier_config)
        classifier.train({
            label_pos: map(create_de, pos),
            label_neg: map(create_de, neg),
        })

        # Report ``label_pos`` class probabilities as rank score.
        scores = classifier.classify_arrays(pool)
        return [c_map.get(label_pos, 0.0) for c_map in scores]
