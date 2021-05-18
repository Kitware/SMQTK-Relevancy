import pkg_resources

from .interfaces.rank_relevancy import RankRelevancy, RankRelevancyWithFeedback  # noqa: F401
__version__ = pkg_resources.get_distribution(__name__).version
