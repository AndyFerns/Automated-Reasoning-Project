import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple, Set, Union
import spacy
from neo4j import GraphDatabase
import re
from dataclasses import dataclass
from enum import Enum
import json
import os

class LogicalOperator(Enum):
    AND = '∧'
    OR = '∨'
    IMPLIES = '→'
    NOT = '¬'
    FORALL = '∀'
    EXISTS = '∃'
    EQUALS = '='

@dataclass
class Term:
    """Represents a term in FOL"""
    name: str
    type: str  # 'variable', 'constant', or 'function'
    args: List['Term'] = None

@dataclass
class Predicate:
    """Represents a predicate in FOL"""
    name: str
    args: List[Term]
    negated: bool = False

@dataclass
class Formula:
    """Represents a FOL formula"""
    type: str  # 'atomic', 'compound', 'quantified'
    operator: LogicalOperator = None
    predicates: List[Union[Predicate, 'Formula']] = None
    variable: Term = None

class FOLInferenceEngine:
    def __init__(self, db_config: Dict = None):
        """
        Initialize the enhanced inference engine
        
        Args:
            db_config: Neo4j database configuration
        """
        # Initialize NLP components
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize database connection
        self.db_config = db_config or {
            'uri': "bolt://localhost:7687",
            'user': "neo4j",
            'password': "password"
        }
        
        # Define FOL rules
        self.fol_rules = {
            'universal': {
                'pattern': r'(all|every|each|any)',
                'operator': LogicalOperator.FORALL
            },
            'existential': {
                'pattern': r'(some|there exists|there is)',
                'operator': LogicalOperator.EXISTS
            },
            'implication': {
                'pattern': r'(if|when|whenever)',
                'operator': LogicalOperator.IMPLIES
            },
            'conjunction': {
                'pattern': r'(and|both)',
                'operator': LogicalOperator.AND
            },
            'disjunction': {
                'pattern': r'(or|either)',
                'operator': LogicalOperator.OR
            },
            'negation': {
                'pattern': r'(not|no|never)',
                'operator': LogicalOperator.NOT
            }
        }
    
    def create_term(self, text: str, term_type: str) -> Term:
        """Create a FOL term from text"""
        return Term(
            name=text.lower(),
            type=term_type
        )
    
    def create_predicate(self, verb: str, args: List[Term], negated: bool = False) -> Predicate:
        """Create a FOL predicate"""
        return Predicate(
            name=verb.lower(),
            args=args,
            negated=negated
        )
    
    def create_formula(self, 
                      predicates: List[Union[Predicate, 'Formula']], 
                      formula_type: str,
                      operator: LogicalOperator = None,
                      variable: Term = None) -> Formula:
        """Create a FOL formula"""
        return Formula(
            type=formula_type,
            operator=operator,
            predicates=predicates,
            variable=variable
        )

    def extract_fol_components(self, doc) -> List[Formula]:
        """Extract FOL components from processed text"""
        formulas = []
        
        for sent in doc.sents:
            # Check for quantifiers and logical operators
            quantifier = None
            operator = None
            
            # Extract main components
            for token in sent:
                # Check for quantifiers
                for rule_type, rule in self.fol_rules.items():
                    if re.search(rule['pattern'], token.text.lower()):
                        operator = rule['operator']
                        break
            
            # Extract terms and predicates
            terms = []
            predicates = []
            
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    term = self.create_term(
                        token.text,
                        'constant' if token.ent_type_ else 'variable'
                    )
                    terms.append(term)
                
                elif token.pos_ == 'VERB':
                    if terms:  # Only create predicate if we have terms
                        pred = self.create_predicate(
                            token.lemma_,
                            terms.copy(),
                            any(child.dep_ == 'neg' for child in token.children)
                        )
                        predicates.append(pred)
                        terms = []  # Reset terms for next predicate
            
            # Create appropriate formula based on structure
            if predicates:
                if operator in [LogicalOperator.FORALL, LogicalOperator.EXISTS]:
                    # Create quantified formula
                    formula = self.create_formula(
                        predicates,
                        'quantified',
                        operator,
                        terms[0] if terms else None
                    )
                elif len(predicates) > 1:
                    # Create compound formula
                    formula = self.create_formula(
                        predicates,
                        'compound',
                        LogicalOperator.AND  # Default to conjunction
                    )
                else:
                    # Create atomic formula
                    formula = self.create_formula(
                        predicates,
                        'atomic'
                    )
                
                formulas.append(formula)
        
        return formulas

    def formula_to_string(self, formula: Formula) -> str:
        """Convert FOL formula to string representation with improved error handling"""
        if not formula or not formula.predicates:
            return ""
            
        if formula.type == 'atomic':
            pred = formula.predicates[0]
            args = ', '.join(arg.name for arg in pred.args)
            return f"{'¬' if pred.negated else ''}{pred.name}({args})"
        
        elif formula.type == 'compound':
            subformulas = [self.formula_to_string(p) if isinstance(p, Formula) 
                          else f"{'¬' if p.negated else ''}{p.name}({', '.join(arg.name for arg in p.args)})"
                          for p in formula.predicates]
            # Use AND as default operator if none is specified
            op_value = formula.operator.value if formula.operator else LogicalOperator.AND.value
            return f" {op_value} ".join(f"({sf})" for sf in subformulas if sf)
        
        elif formula.type == 'quantified':
            var = formula.variable.name if formula.variable else 'x'
            # Create compound formula only if there are multiple predicates
            if len(formula.predicates) > 1:
                subformula = self.formula_to_string(self.create_formula(
                    formula.predicates, 
                    'compound',
                    LogicalOperator.AND
                ))
            else:
                subformula = self.formula_to_string(self.create_formula(
                    formula.predicates, 
                    'atomic'
                ))
            op_value = formula.operator.value if formula.operator else LogicalOperator.FORALL.value
            return f"{op_value}{var}({subformula})"
        
        return ""

    def extract_fol_components(self, doc) -> List[Formula]:
        """Extract FOL components from processed text with improved operator handling"""
        formulas = []
        
        for sent in doc.sents:
            # Check for quantifiers and logical operators
            quantifier = None
            operator = None
            
            # Extract main components
            for token in sent:
                # Check for quantifiers
                for rule_type, rule in self.fol_rules.items():
                    if re.search(rule['pattern'], token.text.lower()):
                        operator = rule['operator']
                        break
            
            # Extract terms and predicates
            terms = []
            predicates = []
            
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    # Handle entity type
                    term_type = 'constant' if token.ent_type_ else 'variable'
                    term = self.create_term(token.text, term_type)
                    terms.append(term)
                
                elif token.pos_ == 'VERB':
                    if terms:  # Only create predicate if we have terms
                        pred = self.create_predicate(
                            token.lemma_,
                            terms.copy(),
                            any(child.dep_ == 'neg' for child in token.children)
                        )
                        predicates.append(pred)
                        terms = []  # Reset terms for next predicate
            
            # Create appropriate formula based on structure
            if predicates:
                if operator in [LogicalOperator.FORALL, LogicalOperator.EXISTS]:
                    # Create quantified formula with default FORALL if no operator specified
                    formula = self.create_formula(
                        predicates,
                        'quantified',
                        operator or LogicalOperator.FORALL,
                        terms[0] if terms else self.create_term('x', 'variable')
                    )
                elif len(predicates) > 1:
                    # Create compound formula with default AND operator
                    formula = self.create_formula(
                        predicates,
                        'compound',
                        LogicalOperator.AND
                    )
                else:
                    # Create atomic formula
                    formula = self.create_formula(
                        predicates,
                        'atomic'
                    )
                
                formulas.append(formula)
        
        return formulas

    def process_and_export(self, text: str, output_file: str = 'fol_export.cypher') -> Dict:
        """Process text and export to Neo4j Workbench format with error handling"""
        try:
            # Process text
            doc = self.nlp(text)
            
            # Extract FOL components
            formulas = self.extract_fol_components(doc)
            
            # Convert to strings for display with error handling
            fol_strings = []
            for f in formulas:
                try:
                    fol_str = self.formula_to_string(f)
                    if fol_str:  # Only add non-empty strings
                        fol_strings.append(fol_str)
                except Exception as e:
                    print(f"Warning: Could not convert formula to string: {str(e)}")
                    continue
            
            # Export to workbench format
            if formulas:
                self.export_to_workbench(formulas, output_file)
            
            return {
                'formulas': formulas,
                'fol_strings': fol_strings,
                'export_file': output_file
            }
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                'formulas': [],
                'fol_strings': [],
                'export_file': None
            }

    def export_to_workbench(self, formulas: List[Formula], output_file: str):
        """Export FOL formulas to Neo4j Desktop Workbench format"""
        cypher_statements = []
        
        for formula in formulas:
            if formula.type == 'atomic':
                pred = formula.predicates[0]
                args = pred.args
                if len(args) >= 2:
                    # Create nodes and relationship
                    stmt = f"""
                    MERGE (s:{args[0].type} {{name: '{args[0].name}'}})
                    MERGE (o:{args[1].type} {{name: '{args[1].name}'}})
                    MERGE (s)-[r:{pred.name}]->(o)
                    """
                    if pred.negated:
                        stmt += f"SET r.negated = true"
                    cypher_statements.append(stmt)
            
            elif formula.type in ['compound', 'quantified']:
                # Handle compound and quantified formulas
                for pred in formula.predicates:
                    if isinstance(pred, Predicate):
                        args = pred.args
                        if len(args) >= 2:
                            stmt = f"""
                            MERGE (s:{args[0].type} {{name: '{args[0].name}'}})
                            MERGE (o:{args[1].type} {{name: '{args[1].name}'}})
                            MERGE (s)-[r:{pred.name}]->(o)
                            SET r.operator = '{formula.operator.value if formula.operator else ''}'
                            """
                            cypher_statements.append(stmt)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('// Neo4j Desktop Workbench Export\n')
            f.write('// FOL Predicates and Relationships\n\n')
            for stmt in cypher_statements:
                f.write(f"{stmt}\n\n")

    def process_and_export(self, text: str, output_file: str = 'fol_export.cypher') -> Dict:
        """Process text and export to Neo4j Workbench format"""
        # Process text
        doc = self.nlp(text)
        
        # Extract FOL components
        formulas = self.extract_fol_components(doc)
        
        # Convert to strings for display
        fol_strings = [self.formula_to_string(f) for f in formulas]
        
        # Export to workbench format
        self.export_to_workbench(formulas, output_file)
        
        return {
            'formulas': formulas,
            'fol_strings': fol_strings,
            'export_file': output_file
        }

def test_enhanced_engine():
    """Test the enhanced FOL inference engine"""
    # Initialize engine
    engine = FOLInferenceEngine()
    
    # Test texts with various logical structures
    texts = [
        "All students study mathematics.",
        "If John works hard, then he will succeed.",
        "Some birds can fly and sing.",
        "No cats are dogs.",
        "Every person who works hard achieves success.",
        "There exists a book that John and Mary both read."
    ]
    
    # Process each text
    for text in texts:
        print(f"\nProcessing: {text}")
        results = engine.process_and_export(
            text,
            f"fol_export_{texts.index(text)}.cypher"
        )
        
        print("FOL Representations:")
        for fol in results['fol_strings']:
            print(f"  {fol}")
        
        print(f"Export file created: {results['export_file']}")

if __name__ == "__main__":
    test_enhanced_engine()