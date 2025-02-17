"""
An advanced system that integrates:
  • Robust natural language parsing (using spaCy)
  • Conversion of parsed sentences to first-order logic (FOL) in clause form
  • A resolution engine for FOL inspired by Prolog's techniques

This version allows you to input your statements (facts and rules)
and your query via the command line. The system will automatically
parse the input, build the knowledge base, and attempt to prove the query.
"""

import spacy
import nltk
import re
import copy

nltk.download('punkt', quiet=True)
nlp = spacy.load("en_core_web_sm")

def is_variable(term):
    return isinstance(term, str) and term.startswith('?')

def unify(x, y, subs=None):
    if subs is None:
        subs = {}
    if subs is False:
        return None
    if x == y:
        return subs
    if is_variable(x):
        return unify_var(x, y, subs)
    if is_variable(y):
        return unify_var(y, x, subs)
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        for xi, yi in zip(x, y):
            subs = unify(xi, yi, subs)
            if subs is None:
                return None
        return subs
    return None

def unify_var(var, x, subs):
    if var in subs:
        return unify(subs[var], x, subs)
    elif is_variable(x) and x in subs:
        return unify(var, subs[x], subs)
    else:
        new_subs = subs.copy()
        new_subs[var] = x
        return new_subs

def substitute(term, subs):
    if isinstance(term, str):
        if is_variable(term) and term in subs:
            return subs[term]
        else:
            return term
    elif isinstance(term, tuple):
        return tuple(substitute(t, subs) for t in term)
    else:
        return term


class Fact:
    def __init__(self, predicate, args, positive=True):
        self.predicate = predicate
        self.args = tuple(args)
        self.positive = positive

    def substitute(self, subs):
        new_args = tuple(substitute(arg, subs) for arg in self.args)
        return Fact(self.predicate, new_args, self.positive)

    def __eq__(self, other):
        return (self.predicate == other.predicate and
                self.args == other.args and
                self.positive == other.positive)

    def __hash__(self):
        return hash((self.predicate, self.args, self.positive))

    def __repr__(self):
        sign = "" if self.positive else "not "
        return f"{sign}{self.predicate}({', '.join(self.args)})"

class Rule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents  # list of Fact objects
        self.consequent = consequent    # a Fact object

    def __repr__(self):
        ants = ", ".join(map(str, self.antecedents))
        return f"{self.consequent} :- {ants}"


class Literal:
    def __init__(self, predicate, args, positive=True):
        self.predicate = predicate
        self.args = tuple(args)
        self.positive = positive

    def __eq__(self, other):
        return (self.predicate == other.predicate and 
                self.args == other.args and 
                self.positive == other.positive)

    def __hash__(self):
        return hash((self.predicate, self.args, self.positive))

    def __repr__(self):
        sign = "" if self.positive else "~"
        return f"{sign}{self.predicate}({', '.join(self.args)})"

def fact_to_literal(fact):
    return Literal(fact.predicate, fact.args, fact.positive)

def rule_to_clause(rule):
    clause = set()
    clause.add(fact_to_literal(rule.consequent))
    for ant in rule.antecedents:
        lit = fact_to_literal(ant)
        lit.positive = not lit.positive  # negate antecedents
        clause.add(lit)
    return frozenset(clause)

def fact_to_clause(fact):
    return frozenset({fact_to_literal(fact)})

def apply_substitution_clause(clause, subs):
    new_clause = set()
    for lit in clause:
        new_args = tuple(substitute(arg, subs) for arg in lit.args)
        new_clause.add(Literal(lit.predicate, new_args, lit.positive))
    return frozenset(new_clause)

def resolve_clauses(ci, cj):
    resolvents = set()
    for li in ci:
        for lj in cj:
            if li.predicate == lj.predicate and li.positive != lj.positive:
                subs = unify(li.args, lj.args, {})
                if subs is not None:
                    new_ci = set(ci)
                    new_cj = set(cj)
                    new_ci.remove(li)
                    new_cj.remove(lj)
                    new_clause = new_ci.union(new_cj)
                    new_clause = apply_substitution_clause(new_clause, subs)
                    resolvents.add(new_clause)
    return resolvents


class FOLResolutionEngine:
    def __init__(self):
        self.clauses = set()

    def add_fact(self, fact):
        self.clauses.add(fact_to_clause(fact))

    def add_rule(self, rule):
        self.clauses.add(rule_to_clause(rule))

    def add_knowledge_base(self, statements):
        for stmt in statements:
            if isinstance(stmt, Fact):
                self.add_fact(stmt)
            elif isinstance(stmt, Rule):
                self.add_rule(stmt)

    def resolution(self, query):
        negated_query = Fact(query.predicate, query.args, positive=not query.positive)
        clauses = self.clauses.copy()
        clauses.add(fact_to_clause(negated_query))
        new = set()
        while True:
            pairs = [(ci, cj) for ci in clauses for cj in clauses if ci != cj]
            for (ci, cj) in pairs:
                resolvents = resolve_clauses(ci, cj)
                if frozenset() in resolvents:
                    return True
                new = new.union(resolvents)
            if new.issubset(clauses):
                return False
            clauses = clauses.union(new)

class RobustNLUParser:
    def __init__(self):
        self.entity_count = 0

    def new_entity(self):
        self.entity_count += 1
        return f"entity{self.entity_count}"

    def parse_sentence(self, sentence):
        doc = nlp(sentence)
        if any(tok.lower_ == "if" for tok in doc) and any(tok.lower_ == "then" for tok in doc):
            return self.parse_implication(sentence)
        else:
            return self.parse_fact(sentence)

    def parse_implication(self, sentence):
        parts = sentence.split("then")
        if len(parts) < 2:
            return None
        antecedent_text = parts[0]
        antecedent_text = re.sub(r'^\s*if\s+', '', antecedent_text, flags=re.IGNORECASE)
        consequent_text = parts[1]
        antecedents = self.extract_facts(antecedent_text)
        consequent = self.extract_fact(consequent_text)
        if antecedents and consequent:
            # If consequent's subject is a pronoun (or not a proper name), replace it.
            pronouns = {"he", "she", "it", "him", "her"}
            subj_ant = antecedents[0].args[0]
            # If consequent subject is missing or is a pronoun, use antecedent subject.
            if not consequent.args or (consequent.args[0] in pronouns):
                # Rebuild consequent with antecedent's subject.
                new_args = (subj_ant,) + consequent.args[1:] if consequent.args else (subj_ant,)
                consequent = Fact(consequent.predicate, new_args, positive=consequent.positive)
        if antecedents and consequent:
            return Rule(antecedents, consequent)
        return None

    def extract_facts(self, text):
        parts = re.split(r'\band\b', text, flags=re.IGNORECASE)
        facts = []
        for part in parts:
            fact = self.extract_fact(part)
            if fact:
                facts.append(fact)
        return facts

    def extract_fact(self, text):
        doc = nlp(text)
        subject = None
        predicate = None
        obj = None
        negated = False
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and subject is None:
                subject = token.text.lower()
            if token.dep_ == "ROOT":
                predicate = token.lemma_.lower()
                for child in token.children:
                    if child.dep_ == "neg":
                        negated = True
            if token.dep_ in ("dobj", "attr") and obj is None:
                obj = token.text.lower()
        if subject is None or predicate is None:
            return None
        if obj is None:
            return Fact(predicate, (subject,), positive=not negated)
        return Fact(predicate, (subject, obj), positive=not negated)

    def parse_fact(self, sentence):
        return self.extract_fact(sentence)

    def parse_text(self, text):
        sentences = nltk.sent_tokenize(text)
        statements = []
        for sent in sentences:
            stmt = self.parse_sentence(sent)
            if stmt:
                statements.append(stmt)
        return statements


if __name__ == "__main__":
    print("Enter your knowledge base statements (facts and rules).")
    print("Enter one statement per line. When done, enter an empty line.")
    kb_lines = []
    while True:
        line = input("> ")
        if line.strip() == "":
            break
        kb_lines.append(line)
    input_text = "\n".join(kb_lines)
    
    parser = RobustNLUParser()
    statements = parser.parse_text(input_text)
    
    print("\nParsed Statements:")
    for stmt in statements:
        print(stmt)
    
    resolution_engine = FOLResolutionEngine()
    resolution_engine.add_knowledge_base(statements)
    
    print("\nEnter your query (as a sentence):")
    query_text = input("> ")
    query_stmt = parser.parse_sentence(query_text)
    
    if query_stmt is None or not isinstance(query_stmt, Fact):
        print("Unable to parse the query as a fact. Please check your input.")
    else:
        result = resolution_engine.resolution(query_stmt)
        print("\nQuery:", query_stmt)
        print("Query entailed by the knowledge base:", result)
