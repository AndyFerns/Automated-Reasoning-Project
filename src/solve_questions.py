# solve_questions.py
import re
import logging
from semantic_parser import SemanticParser
from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

def solve_questions_file(filename):
    """
    Process a file containing multiple questions.
    
    Each question block is structured as follows:
    
      Q 1
      
      <Inference statement lines>
      
      Prove that:
      <Query statement line>
      
    This function:
      - Extracts the question identifier (e.g., "Q 1", "Q 2", …)
      - Registers all inference statements into a new Knowledge Base for that question
      - Executes the query (the first line after "Prove that:")
      - Prints the question ID, extracted predicates, query, and results.
    """
    with open(filename, "r") as f:
        content = f.read()

    # Use regex to capture question blocks. Each block starts with a line like "Q 1"
    # The split pattern uses a multiline flag to split on lines beginning with "Q" followed by a number.
    question_blocks = re.split(r"(?m)^Q\s*\d+", content)
    # Also, capture all question IDs (the numbers) to label output.
    question_ids = re.findall(r"(?m)^Q\s*(\d+)", content)
    
    # Loop through each block. Note that the first split may be empty if the file starts with a Q-line.
    for idx, block in enumerate(question_blocks):
        block = block.strip()
        if not block:
            continue

        # Split the block into two parts using the "Prove that:" line.
        parts = re.split(r"(?mi)^prove that:\s*", block)
        if len(parts) < 2:
            print(f"Error: Question block {question_ids[idx] if idx < len(question_ids) else idx+1} does not contain a 'Prove that:' line.")
            continue

        # Inference text: all lines before the "Prove that:" line.
        inference_text = parts[0].strip()
        # Query text: we take the first non-empty line after "Prove that:".
        query_text = ""
        for line in parts[1].splitlines():
            line = line.strip()
            if line:
                query_text = line
                break
        if not query_text:
            print(f"Error: No query found in question block {question_ids[idx] if idx < len(question_ids) else idx+1}.")
            continue

        # Extract individual inference sentences (assume one per line).
        inference_sentences = [line.strip() for line in inference_text.splitlines() if line.strip()]

        # Print a header for the question.
        qid = question_ids[idx] if idx < len(question_ids) else idx+1
        print(f"\n=== Processing Question {qid} ===")
        
        # Create a new KB and Semantic Parser instance for this question.
        sp = SemanticParser()
        kb = KnowledgeBase()

        # Process each inference sentence.
        for sentence in inference_sentences:
            try:
                pred = sp.parse_sentence(sentence)
                kb.assert_predicate(pred)
                print("Extracted:", pred)
            except Exception as e:
                print(f"Error processing sentence '{sentence}':", e)
        
        # Process the query. Use parse_query if it begins with "is", otherwise use parse_sentence.
        try:
            if query_text.lower().startswith("is "):
                query = sp.parse_query(query_text)
            else:
                query = sp.parse_sentence(query_text)
            results = kb.query(query)
            print("Query:", query)
            print("Results:", results)
        except Exception as e:
            print("Error processing query:", e)
