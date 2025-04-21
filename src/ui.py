# ui.py
import sys
from semantic_parser import SemanticParser
from knowledge_base import KnowledgeBase

def interactive_mode():
    parser = SemanticParser()
    kb = KnowledgeBase()
    print("Entering CLI interactive mode. Type 'exit' to quit.")
    while True:
        sentence = input("\nEnter an English sentence: ").strip()
        if sentence.lower() == "exit":
            sys.exit(0)
        try:
            pred = parser.parse_sentence(sentence)
            kb.assert_predicate(pred)
            print("Asserted predicate:")
            print(pred)
        except Exception as e:
            print("Error:", e)
        query_input = input("Enter a Prolog query (or press Enter to skip): ").strip()
        if query_input:
            try:
                # Always try to parse the query into a Prolog predicate.
                query = parser.parse_sentence(query_input)
            except Exception as e:
                print("Error parsing query:", e)
                query = query_input  # Fallback
            results = kb.query(query)
            print("Query:", query)
            if results:
                print("\nQuery result: True")
            else:
                print("\nQuery result: False")

if __name__ == "__main__":
    interactive_mode()
