# knowledge_base.py
import logging
from pyswip import Prolog

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class KnowledgeBase:
    def __init__(self):
        self.prolog = Prolog()
        self.predicates = []  # List of asserted predicates

    def assert_predicate(self, pred: str):
        for line in pred.split("\n"):
            line = line.rstrip(".")
            self.predicates.append(line)
            self.prolog.assertz(line)
            logger.info("Asserted predicate: %s", line)

    def query(self, query_str: str):
        try:
            results = list(self.prolog.query(query_str))
            logger.info("Query '%s' returned: %s", query_str, results)
            return results
        except Exception as e:
            logger.error("Error querying '%s': %s", query_str, e)
            return []

    def get_all_predicates(self):
        return self.predicates
