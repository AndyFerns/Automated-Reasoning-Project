# main.py
import argparse
from ui import interactive_mode
from api import app
from solve_questions import solve_questions_file
from persistence import PersistenceManager
from knowledge_base import KnowledgeBase
from semantic_parser import SemanticParser

def main():
    parser = argparse.ArgumentParser(description="Advanced NLP-to-FOL Inference Engine")
    parser.add_argument("--mode", choices=["api", "cli", "file", "solve"], default="cli",
                        help="Run mode: 'api' to launch the Flask API; 'cli' for interactive mode; 'file' to process inferences from a file; 'solve' to process questions from a file.")
    parser.add_argument("--infile", help="Input file path for file or solve mode.")
    args = parser.parse_args()

    if args.mode == "api":
        app.run(debug=True)
    elif args.mode == "file":
        if not args.infile:
            print("Error: --infile must be provided for file mode.")
            return
        kb = KnowledgeBase()
        pm = PersistenceManager(kb)
        sp = SemanticParser()
        with open(args.infile, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pred = sp.parse_sentence(line)
                        kb.assert_predicate(pred)
                        print("Asserted:", pred)
                    except Exception as e:
                        print(f"Error processing line '{line}':", e)
        pm.save_to_file("knowledge_base.pl")
    elif args.mode == "solve":
        if not args.infile:
            print("Error: --infile must be provided for solve mode.")
            return
        solve_questions_file(args.infile)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
