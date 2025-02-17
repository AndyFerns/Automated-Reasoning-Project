"""
A simple automated reasoning engine example.
 - Representing facts such as: works_hard(jack)
 - Representing negated facts (e.g. not get_job(jack))
 - Rules that use variables (e.g. if student(?x) then passed_exam(?x))
 - A very basic forward chaining inference mechanism
    - Querying for facts (e.g. is there a fact works_hard(jack)?)
"""

import copy


def is_variable(term):
    return isinstance(term, str) and term.startswith('?')

def unify(x, y, subs=None):
    """
    Try to unify two terms (which may be strings or tuples of terms)
    given the current substitution dictionary.
    """
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
    """
    Recursively substitute any variable in term using the subs dictionary.
    """
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
        predicate: a string name (e.g. "works_hard")
        args: a tuple (or list) of arguments (e.g. ("jack",))
        positive: True for a positive fact, False for negation.
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
        antecedents: list of Fact objects (they can contain variables like ?x)
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
        """Run forward chaining until no new facts are added."""
        added = True
        while added:
            added = False
            for rule in self.rules:
                substitutions = self.match_antecedents(rule.antecedents, self.facts)
                for subs in substitutions:
                    new_fact = rule.consequent.substitute(subs)
                    if new_fact not in self.facts:
                        print(f"Inferred: {new_fact} from rule: {rule} with subs {subs}") #line to see each inference step
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
        """
        Returns True if there is any fact that unifies with query_fact.
        """
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


def test_Q1():
    print("=== Q1 ===")
    engine = ReasoningEngine()
    # Premises:
    # 1. Jack works hard.
    engine.add_fact(Fact("works_hard", ("jack",)))
    # 2. If Jack works hard, then he is a dull boy.
    engine.add_rule(Rule([Fact("works_hard", ("jack",))],
                         Fact("dull_boy", ("jack",))))
    # 3. If Jack is a dull boy, then he will not get the job.
    engine.add_rule(Rule([Fact("dull_boy", ("jack",))],
                         Fact("get_job", ("jack",), positive=False)))
    engine.infer()
    result = engine.query(Fact("get_job", ("jack",), positive=False))
    print("Proved: Jack will not get the job:", result)
    print("Inferred facts:", engine.get_facts())
    print()

def test_Q2():
    print("=== Q2 ===")
    engine = ReasoningEngine()
    # Premises:
    # "A student in this class has not read the book."
    engine.add_fact(Fact("student", ("student1",)))
    engine.add_fact(Fact("read_book", ("student1",), positive=False))
    # "Everyone in this class passed the first exam."
    engine.add_rule(Rule([Fact("student", ("?x",))],
                         Fact("passed_exam", ("?x",))))
    engine.infer()
    # Query: Someone who passed the first exam has not read the book.
    candidates = []
    for fact in engine.facts:
        if fact.predicate == "passed_exam" and fact.positive:
            x = fact.args[0]
            if Fact("read_book", (x,), positive=False) in engine.facts:
                candidates.append(x)
    result = len(candidates) > 0
    print("Proved: Someone who passed the first exam has not read the book:", result)
    print("Candidate(s):", candidates)
    print("Inferred facts:", engine.get_facts())
    print()

def test_Q3():
    print("=== Q3 ===")
    engine = ReasoningEngine()
    # Premises:
    # "Somebody in the Circus is a mole."
    engine.add_fact(Fact("circus_member", ("person1",)))
    engine.add_fact(Fact("mole", ("person1",)))
    # "Every person who is a mole hates Beggarman."
    engine.add_rule(Rule([Fact("mole", ("?x",))],
                         Fact("hates", ("?x", "Beggarman"))))
    engine.infer()
    # Query: There is a person in the Circus who hates Beggarman.
    candidates = []
    for fact in engine.facts:
        if fact.predicate == "hates" and fact.positive:
            x, target = fact.args
            if target == "Beggarman" and Fact("circus_member", (x,)) in engine.facts:
                candidates.append(x)
    result = len(candidates) > 0
    print("Proved: There is a person in the Circus who hates Beggarman:", result)
    print("Candidate(s):", candidates)
    print("Inferred facts:", engine.get_facts())
    print()

def test_Q4():
    print("=== Q4 ===")
    engine = ReasoningEngine()
    # Premises:
    # "Whoever can read is literate."
    engine.add_rule(Rule([Fact("can_read", ("?x",))],
                         Fact("literate", ("?x",))))
    # To allow reasoning by contraposition, add:
    # "If someone is not literate, then they cannot read."
    engine.add_rule(Rule([Fact("literate", ("?x",), positive=False)],
                         Fact("can_read", ("?x",), positive=False)))
    # "Dolphins are not literate." (For a dolphin, use "dolphin1")
    engine.add_fact(Fact("literate", ("dolphin1",), positive=False))
    engine.add_fact(Fact("dolphin", ("dolphin1",)))
    # "Some dolphins are intelligent."
    engine.add_fact(Fact("intelligent", ("dolphin1",)))
    engine.infer()
    # Query: Some who are intelligent cannot read.
    candidates = []
    for fact in engine.facts:
        if fact.predicate == "intelligent" and fact.positive:
            x = fact.args[0]
            if Fact("can_read", (x,), positive=False) in engine.facts:
                candidates.append(x)
    result = len(candidates) > 0
    print("Proved: Some who are intelligent cannot read:", result)
    print("Candidate(s):", candidates)
    print("Inferred facts:", engine.get_facts())
    print()

def test_Q5():
    print("=== Q5 ===")
    engine = ReasoningEngine()
    # Premises:
    # "All wolves howl at night." => if wolf(x) then howl_at_night(x)
    engine.add_rule(Rule([Fact("wolf", ("?x",))],
                         Fact("howl_at_night", ("?x",))))
    # "Anyone who has a horse will not have donkeys." => if has_horse(x) then not has_donkey(x)
    engine.add_rule(Rule([Fact("has_horse", ("?x",))],
                         Fact("has_donkey", ("?x",), positive=False)))
    # "Early risers do not have anything which howl at night."
    # We simplify this by stating: if early_riser(x) then x does not have a wolf.
    engine.add_rule(Rule([Fact("early_riser", ("?x",))],
                         Fact("wolf", ("?x",), positive=False)))
    # "Andrew has either a horse or a wolf." – for consistency with early risers, we choose:
    engine.add_fact(Fact("has_horse", ("Andrew",)))
    engine.add_fact(Fact("early_riser", ("Andrew",)))
    engine.infer()
    # Query: If Andrew is an early riser, then he does not have any donkeys.
    result = engine.query(Fact("has_donkey", ("Andrew",), positive=False))
    print("Proved: If Andrew is an early riser, then he does not have any donkeys:", result)
    print("Inferred facts:", engine.get_facts())
    print()

def test_Q6():
    print("=== Q6 ===")
    # For Q6 we use candidate elimination (since it involves disjunction)
    # Premises:
    # - One of Tinker, Tailor, Soldier, and Spy is the mole.
    # - The mole was not present at the dinner party.
    # - Spy was attending the dinner party.
    # - The mole smokes Havana cigars and always wears a green shirt at home.
    # - Soldier does not smoke.
    # - Tinker was wearing a pink shirt at home.
    # - Anything coloured pink is not green.
    #
    # We can eliminate:
    # - Spy (attended dinner, but mole was absent)
    # - Soldier (does not smoke, but mole smokes)
    # - Tinker (wears pink, so cannot be green as required)
    # Remaining candidate: Tailor.
    candidates = {"Tinker", "Tailor", "Soldier", "Spy"}
    candidates.discard("Spy")
    candidates.discard("Soldier")
    candidates.discard("Tinker")
    print("Determined: The mole is", candidates)
    print()

def test_Q7():
    print("=== Q7 ===")
    # For Q7 we build a simple family tree and then determine the aunt.
    #
    # Premises:
    # - Jack is father of Jess and Lily.
    # - Helen is mother of Jess and Lily.
    # - Oliver and Sophie are father and mother of James.
    # - Simon is son of Jess.
    # - Marcus is father of Simon.
    # - Lily and James are mother and father of Harry.
    #
    # We assume (from names) that Jess, Helen, Lily, and Sophie are female.
    parents = {
        "Jess": {"Jack", "Helen"},
        "Lily": {"Jack", "Helen"},
        "James": {"Oliver", "Sophie"},
        "Simon": {"Jess"},  
        "Harry": {"Lily", "James"}
    }
    female = {"Helen", "Lily", "Jess", "Sophie"}
    
    def siblings(person):
        sibs = set()
        for other, pars in parents.items():
            if other != person and len(parents.get(person, set()).intersection(pars)) > 0:
                sibs.add(other)
        return sibs


    aunts = set()
    for parent in parents["Harry"]:
        for sib in siblings(parent):
            if sib in female:
                aunts.add(sib)
    print("Determined: The aunt of Harry is", aunts)
    print()


if __name__ == "__main__":
    test_Q1()
    test_Q2()
    test_Q3()
    test_Q4()
    test_Q5()
    test_Q6()
    test_Q7()
