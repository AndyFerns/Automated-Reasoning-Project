# Automated Reasoning Engine

An advanced Python-based engine designed to perform automated reasoning on first-order logic statements. This project leverages Natural Language Processing (NLP) techniques to analyze, parse, and evaluate logical statements, providing a robust foundation for applications in AI-driven reasoning and decision-making.

## Features

- **Tokenization & Part-of-Speech Tagging**: Efficiently identifies and classifies parts of speech (e.g., nouns, verbs, prepositions) using Python's NLTK library.
- **Proper Noun Classification**: Distinguishes proper nouns from common nouns using a comprehensive dataset.
- **Atomic Sentence Detection**: Parses and identifies atomic sentences to construct logical predicates.
- **Predicate Formation**: Translates natural language into predicate forms, similar to Prolog statements.
- **Text Parsing**: Analyzes large text files to extract relevant logical constructs.
- **Customizable Logic Rules**: Allows users to define and extend reasoning rules to fit specific needs.
- **Modular Design**: Highly modular codebase for ease of understanding, debugging, and extension.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/automated-reasoning-engine.git
   cd automated-reasoning-engine
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Workflow

1. Prepare your input text or logical statements.
2. Run the main script:
   ```bash
   python main.py --input "path/to/your/input.txt"
   ```
3. View the parsed predicates, reasoning results, and any inferred conclusions.

### Example Input

```
All humans are mortal.
Socrates is a human.
Therefore, Socrates is mortal.
```

### Example Output

```
Predicates:
- human(Socrates)
- mortal(Socrates)

Reasoning Result:
- True: Socrates is mortal.
```

## Project Structure

```
.
├── src/                      # Source code
│   ├── api.py                # Flask api used for deployment
│   ├── knowledge_base.py     # Predicate storage and formatting 
│   ├── main.py               # Main entry point for the application
│   ├── nlp_processor.py      # Initialize the nlp module
│   ├── persistence.py        # Storage into neo4j
│   ├── semantic_parser.py    # Sentence parsing and predicate formation   
│   ├── solve_questions.py    # Sample module for testing
│   ├── ui.py                 # CLI for user interfacing
│   ├── Front End/            # Web-based user interface
│   │   ├── graph.html        # Displays the stored nodes
│   │   ├── index.html        # Main page of the inference engine
│   │   ├── script.js         # Handle requests 
│   │   ├── styles.css        # Style for the website
```

## Contributions

Contributions are welcome! If you'd like to contribute, please:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to your forked repository and create a pull request.

## Contact

If you have any questions or suggestions, feel free to reach out:

- Email: write2andrew.important@gmail.com

