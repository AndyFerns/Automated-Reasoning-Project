import sys
import re
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
            print("Error asserting sentence:", e)

        query_input = input("Enter a Prolog or English query (or press Enter to skip): ").strip()
        if not query_input:
            continue

        try:
            # Normalize input (remove trailing dot)
            q_raw = query_input.rstrip('.').strip()
            # Detect Prolog-style query by presence of parentheses
            if '(' in q_raw and ')' in q_raw:
                query = q_raw
            else:
                # Treat as English and convert via semantic parser
                query = parser.parse_query(query_input)
        except Exception as e:
            print("Error parsing query:", e)
            continue

        try:
            results = kb.query(query)
            print("Query:", query)
            if results:
                print("\nQuery result: True")
            else:
                print("\nQuery result: False")
        except Exception as e:
            print("Error executing query:", e)

if __name__ == "__main__":
    interactive_mode()
