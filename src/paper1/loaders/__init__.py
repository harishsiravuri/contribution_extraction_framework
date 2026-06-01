"""Gold-benchmark loaders. Each returns a list of GoldPaper."""

from paper1.loaders.base import GoldContribution, GoldPaper
from paper1.loaders.nlp_tdms import load_nlp_tdms
from paper1.loaders.scirex import load_scirex, load_scirex_train
from paper1.loaders.tdmsci import load_tdmsci

__all__ = [
    "GoldContribution",
    "GoldPaper",
    "load_nlp_tdms",
    "load_scirex",
    "load_scirex_train",
    "load_tdmsci",
]
