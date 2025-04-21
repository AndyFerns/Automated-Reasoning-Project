from flask import Flask, request, jsonify
import logging
from flask_cors import CORS  # <-- Import flask-cors
from semantic_parser import SemanticParser
from knowledge_base import KnowledgeBase
from persistence import PersistenceManager

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True) # Enable CORS for all routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize backend components
parser = SemanticParser()   
kb = KnowledgeBase()
persistence_manager = PersistenceManager(kb)

@app.route("/assert", methods=["POST", "OPTIONS"])
def assert_statement():
    if request.method == "OPTIONS":
        # Just respond to preflight with OK and proper headers
        return '', 200
    
    data = request.get_json(force=True)
    if "sentence" not in data:
        return jsonify({"error": "Missing 'sentence' in request"}), 400

    sentence = data["sentence"]
    try:
        predicate = parser.parse_sentence(sentence)
        kb.assert_predicate(predicate)
        return jsonify({"predicate": predicate}), 200
    except Exception as e:
        logger.error("Error asserting statement: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["GET"])
def query_statement():
    query_str = request.args.get("q")
    if not query_str:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    try:
        results = kb.query(query_str)
        return jsonify({"results": results}), 200
    except Exception as e:
        logger.error("Error querying: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/save", methods=["POST", "OPTIONS"])
def save_predicates():
    if request.method == "OPTIONS":
        # Just respond to preflight with OK and proper headers
        return '', 200
    
    data = request.json
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
        # Just respond to preflight with OK and proper headers
        return '', 200
    
    data = request.json
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
