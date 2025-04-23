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
        """
        Export binary predicates (relationship(X, Y)) as graph edges:
        Node(X)-[:RELATIONSHIP]->Node(Y).
        """
        try:
            from py2neo import Graph, Node, Relationship
        except ImportError:
            logger.error("py2neo is not installed. Install with 'pip install py2neo'.")
            raise

        try:
            graph = Graph(uri, auth=(user, password))
            graph.delete_all()  # Clear the database (for demo)

            for pred in self.kb.get_all_predicates():
                functor, args = self._parse_predicate(pred)

                # Only handle binary predicates relationship(X, Y)
                if len(args) == 2:
                    subj, obj = args[0].strip(), args[1].strip()
                    # Merge subject and object nodes by name
                    node_subj = Node("Entity", name=subj)
                    node_obj = Node("Entity", name=obj)
                    graph.merge(node_subj, "Entity", "name")
                    graph.merge(node_obj, "Entity", "name")

                    # Create relationship edge
                    rel = Relationship(node_subj, functor.upper(), node_obj)
                    graph.create(rel)
                else:
                    logger.debug("Skipping non-binary predicate: %s", pred)

            logger.info("Exported binary predicates to Neo4j as relationships.")
        except Exception as e:
            logger.error("Error exporting to Neo4j: %s", e)
            raise

    def _parse_predicate(self, pred: str):
        """
        Returns tuple (functor, args_list) for a predicate string like 'likes(john,chocolate)'.
        Ignores rules (contains ':-').
        """
        if ':-' in pred:
            # Rules are not exported as relationships
            return (None, [])
        try:
            functor, rest = pred.split('(', 1)
            args = rest.rstrip(')').split(',')
            return (functor.strip(), args)
        except ValueError:
            return (pred.strip(), [])
