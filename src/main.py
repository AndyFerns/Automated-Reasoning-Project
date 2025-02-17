"""
A demonstration of a forward-chaining reasoning engine that attempts to 
parse natural language input into facts and rules (using heuristics) and then
infers additional facts. This code is very limited and meant only as a proof-of-concept.
"""

import nltk
import re
import copy

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


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
        """
        predicate: string name (e.g., "works_hard")
        args: tuple of arguments (e.g., ("jack",) or ("jack", "job"))
        positive: True means a positive fact; False for negation.
        """
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
        """
        antecedents: a list of Fact objects (conditions, which may contain variables)
        consequent: a Fact (possibly with variables) that should follow.
        """
        self.antecedents = antecedents
        self.consequent = consequent

    def __repr__(self):
        ants = " and ".join(map(str, self.antecedents))
        return f"If {ants} then {self.consequent}"

class ReasoningEngine:
    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_fact(self, fact):
        self.facts.add(fact)

    def add_rule(self, rule):
        self.rules.append(rule)

    def infer(self):
        """Run forward chaining until no new facts can be derived."""
        added = True
        while added:
            added = False
            for rule in self.rules:
                substitutions = self.match_antecedents(rule.antecedents, self.facts)
                for subs in substitutions:
                    new_fact = rule.consequent.substitute(subs)
                    if new_fact not in self.facts:
                        # Uncomment the next line to see each inference step:
                        print(f"Inferred: {new_fact} via {rule} with substitution {subs}")
                        self.facts.add(new_fact)
                        added = True

    def match_antecedents(self, antecedents, facts, subs=None):
        """Recursively match a list of antecedents against known facts."""
        if subs is None:
            subs = {}
        if not antecedents:
            return [subs]
        first, rest = antecedents[0], antecedents[1:]
        results = []
        for fact in facts:
            if fact.predicate == first.predicate and fact.positive == first.positive:
                subs_new = unify(first.args, fact.args, subs)
                if subs_new is not None:
                    for subs_final in self.match_antecedents(rest, facts, subs_new):
                        results.append(subs_final)
        return results

    def query(self, query_fact):
        """Return True if any known fact unifies with query_fact."""
        for fact in self.facts:
            if fact.predicate == query_fact.predicate and fact.positive == query_fact.positive:
                subs = unify(query_fact.args, fact.args, {})
                if subs is not None:
                    return True
        return False

    def get_facts(self, predicate=None):
        if predicate:
            return [f for f in self.facts if f.predicate == predicate]
        return list(self.facts)


class NaturalLanguageParser:
    def __init__(self):
        # Used to generate new constant names for existentially quantified entities.
        self.entity_count = 0

    def new_entity(self):
        self.entity_count += 1
        return f"entity{self.entity_count}"

    def parse_text(self, text):
        """
        Splits the input text into sentences and attempts to parse each as either
        a fact or a rule.
        """
        sentences = nltk.sent_tokenize(text)
        parsed_statements = []
        for sentence in sentences:
            sentence = sentence.strip().rstrip(".!?")
            lower = sentence.lower()
            if "if" in lower and "then" in lower:
                rule = self.parse_rule(sentence)
                if rule:
                    parsed_statements.append(rule)
            elif lower.startswith("everyone") or lower.startswith("all"):
                rule = self.parse_universal(sentence)
                if rule:
                    parsed_statements.append(rule)
            elif lower.startswith("somebody") or lower.startswith("someone"):
                fact = self.parse_existential(sentence)
                if fact:
                    parsed_statements.append(fact)
            else:
                fact = self.parse_fact(sentence)
                if fact:
                    parsed_statements.append(fact)
        return parsed_statements

    def parse_rule(self, sentence):
        """
        Very naively splits a sentence of the form:
          "If <antecedents> then <consequent>"
        into antecedents and conclusion, and parses each part.
        """
        parts = re.split(r'\bthen\b', sentence, flags=re.IGNORECASE)
        if len(parts) < 2:
            return None
        antecedent_part = parts[0]
        # Remove leading "if" (if present)
        antecedent_part = re.sub(r'^\s*if\s+', '', antecedent_part, flags=re.IGNORECASE).strip()
        conclusion_part = parts[1].strip()
        # If multiple antecedents are joined by "and", split them.
        antecedent_sentences = re.split(r'\band\b', antecedent_part, flags=re.IGNORECASE)
        antecedents = []
        for ant in antecedent_sentences:
            fact = self.parse_fact(ant.strip())
            if fact:
                antecedents.append(fact)
        conclusion = self.parse_fact(conclusion_part)
        if not antecedents or conclusion is None:
            return None
        return Rule(antecedents, conclusion)

    def parse_fact(self, sentence):
        """
        A very simple heuristic fact parser that tokenizes the sentence and
        tries to extract a subject, a verb, and (optionally) an object.
        For example, "Jack works hard" becomes Fact("works", ("jack", "hard")).
        """
        tokens = nltk.word_tokenize(sentence)
        if not tokens:
            return None
        tags = nltk.pos_tag(tokens)
        subject = None
        verb = None
        obj = None
        positive = True
        if "not" in tokens:
            positive = False
            tokens.remove("not")
        # Heuristically pick the first noun as subject, first verb as predicate,
        # and (if available) the next noun as object.
        for word, tag in tags:
            if subject is None and tag in ["NNP", "NN", "PRP"]:
                subject = word.lower()
            elif verb is None and tag.startswith("VB"):
                verb = word.lower()
            elif obj is None and tag in ["NN", "NNS", "NNP", "PRP"]:
                if subject and word.lower() != subject:
                    obj = word.lower()
        if subject is None or verb is None:
            return None
        if obj is None:
            return Fact(verb, (subject,), positive)
        return Fact(verb, (subject, obj), positive)

    def parse_universal(self, sentence):
        """
        Attempts to convert a sentence that begins with "everyone" or "all" into a rule.
        For example:
          "Everyone in this class passed the first exam."
        is heuristically transformed into a rule:
          IF in_group(?x, this) THEN passed(..., with ?x as subject)
        (This is a very limited interpretation.)
        """
        lower = sentence.lower()
        if lower.startswith("everyone"):
            remainder = sentence[len("everyone"):].strip()
        elif lower.startswith("all"):
            remainder = sentence[len("all"):].strip()
        else:
            remainder = sentence
        antecedent = None
        if " in " in remainder:
            m = re.search(r'in ([\w\s]+)', remainder, flags=re.IGNORECASE)
            if m:
                group = m.group(1).strip().split()[0]
                antecedent = Fact("in_group", ("?x", group))
        if antecedent is None:
            antecedent = Fact("person", ("?x",))
        conclusion = self.parse_fact(sentence)
        if conclusion is None:
            return None
        args = list(conclusion.args)
        args[0] = "?x"
        conclusion = Fact(conclusion.predicate, tuple(args), conclusion.positive)
        return Rule([antecedent], conclusion)

    def parse_existential(self, sentence):
        """
        For sentences starting with "somebody" or "someone", we generate a new entity name.
        For example, "Somebody in the circus is a mole." might become:
          Fact("is", (entity1, "mole"))
        """
        entity = self.new_entity()
        fact = self.parse_fact(sentence)
        if fact is None:
            return None
        args = list(fact.args)
        args[0] = entity
        return Fact(fact.predicate, tuple(args), fact.positive)


if __name__ == "__main__":
    input_text = """
    Jack works hard.
    If Jack works hard, then he is a dull boy.
    If Jack is a dull boy, then he will not get the job.
    Everyone in this class passed the first exam.
    Somebody in the circus is a mole.
    """

    print("Input text:")
    print(input_text)
    
    parser = NaturalLanguageParser()
    statements = parser.parse_text(input_text)
    print("\nParsed Statements:")
    for s in statements:
        print(s)
    
    engine = ReasoningEngine()
    for s in statements:
        if isinstance(s, Fact):
            engine.add_fact(s)
        elif isinstance(s, Rule):
            engine.add_rule(s)
    
    engine.infer()
    print("\nInferred Facts:")
    for fact in engine.get_facts():
        print(fact)
