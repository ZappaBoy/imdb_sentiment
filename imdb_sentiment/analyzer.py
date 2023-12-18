import importlib.metadata as metadata

from imdb_sentiment.shared.utils.logger import Logger

__version__ = metadata.version(__package__ or __name__)


class Analyzer:
    def __init__(self):
        self.logger = Logger()

    def run(self):
        self.logger.info(f"Analyzing")
