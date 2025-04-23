from flask import Flask, request, jsonify
import logging
from flask_cors import CORS  # <-- Import flask-cors
from semantic_parser import SemanticParser
from knowledge_base import KnowledgeBase
from persistence import PersistenceManager

app = Flask(__name__)
# Enable CORS for all routes (adjust origins as needed)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize backend components
parser = SemanticParser()
kb = KnowledgeBase()
persistence_manager = PersistenceManager(kb)

@app.route("/assert", methods=["POST", "OPTIONS"])
def assert_statement():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json(force=True)
    sentence = data.get("sentence")
    if not sentence:
        return jsonify({"error": "Missing 'sentence' in request"}), 400

    try:
        predicate = parser.parse_sentence(sentence)
        kb.assert_predicate(predicate)
        return jsonify({"predicate": predicate}), 200
    except Exception as e:
        logger.error("Error asserting statement: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST", "OPTIONS"])
def query_statement():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json(force=True)
    raw_query = data.get("query")
    if not raw_query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    try:
        # Convert English or Prolog input into a Prolog predicate
        parsed_query = parser.parse_query(raw_query)
        # Execute against the knowledge base
        results = kb.query(parsed_query)
        return jsonify({
            "parsed_query": parsed_query,
            "results": results
        }), 200
    except Exception as e:
        logger.error("Error querying: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/save", methods=["POST", "OPTIONS"])
def save_predicates():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json(force=True)
    filename = data.get("filename", "knowledge_base.pl")
    try:
        persistence_manager.save_to_file(filename)
        return jsonify({"message": f"Saved to {filename}"}), 200
    except Exception as e:
        logger.error("Error saving predicates: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/export_neo4j", methods=["POST", "OPTIONS"])
def export_neo4j():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json(force=True)
    uri = data.get("uri")
    user = data.get("user")
    password = data.get("password")
    if not (uri and user and password):
        return jsonify({"error": "Missing Neo4j connection parameters"}), 400

    try:
        persistence_manager.export_to_neo4j(uri, user, password)
        return jsonify({"message": "Exported to Neo4j"}), 200
    except Exception as e:
        logger.error("Error exporting to Neo4j: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
