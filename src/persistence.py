# persistence.py
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PersistenceManager:
    def __init__(self, kb):
        self.kb = kb

    def save_to_file(self, filename: str):
        try:
            with open(filename, "w") as f:
                for pred in self.kb.get_all_predicates():
                    f.write(pred + ".\n")
            logger.info("Predicates saved to %s", filename)
        except Exception as e:
            logger.error("Error saving predicates: %s", e)
            raise

    def export_to_neo4j(self, uri, user, password):
        try:
            from py2neo import Graph, Node
        except ImportError:
            logger.error("py2neo is not installed. Install with 'pip install py2neo'.")
            raise
        try:
            graph = Graph(uri, auth=(user, password))
            graph.delete_all()  # Clear the database (for demo)
            for pred in self.kb.get_all_predicates():
                functor, args = self._parse_predicate(pred)
                node = Node("Predicate", name=functor, args=str(args))
                graph.create(node)
            logger.info("Exported predicates to Neo4j.")
        except Exception as e:
            logger.error("Error exporting to Neo4j: %s", e)
            raise

    def _parse_predicate(self, pred: str):
        if ":-" in pred:
            return ("rule", pred)
        try:
            functor, rest = pred.split("(", 1)
            args = rest.rstrip(")")
            return (functor, args.split(","))
        except Exception:
            return (pred, [])
