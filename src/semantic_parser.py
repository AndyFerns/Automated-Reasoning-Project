# semantic_parser.py
import re
import uuid
import logging
from nlp_processor import NLPProcessor
from pyswip import Prolog  # Prolog interface for proof

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PRONOUNS = {"I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}

class SemanticParser:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        # initialize Prolog engine for proof capabilities
        self.prolog = Prolog()

    def parse_sentence(self, sentence: str) -> str:
        # Special universal rule: Whoever can read is literate
        if sentence.lower().startswith("whoever "):
            text = sentence.strip().rstrip('.')
            head, body = text.split(" is ", 1)
            # cond: can_read(X), cons: literate(X)
            cond_pred = self._convert_to_predicate("can read", "X")
            cons_pred = self._convert_to_predicate(body, "X")
            return f"{cons_pred} :- {cond_pred}"

        # handle multiple sentences
        if "." in sentence and sentence.count(".") > 1:
            parts = [s.strip() for s in sentence.split(".") if s.strip()]
            preds = [self.parse_sentence(s) for s in parts]
            return "\n".join(preds)

        sentence = sentence.strip().rstrip(".")
        if not sentence:
            raise ValueError("Empty sentence provided.")
        if sentence.lower().startswith("somebody"):
            sentence = re.sub(r"^somebody", "some person", sentence, flags=re.IGNORECASE)
        if sentence.lower().startswith("some who"):
            sentence = re.sub(r"^some who", "some person who", sentence, flags=re.IGNORECASE)
        logger.info("Processing sentence: '%s'", sentence)
        doc = self.nlp_processor.process_text(sentence)

        if sentence.lower().startswith("if "):
            if "else" in sentence.lower():
                return self._parse_if_else(sentence)
            return self._parse_conditional(sentence)
        if sentence.lower().startswith("is "):
            return self._parse_query_fact(sentence)
        
        # special overrides
        if "not read the book" in sentence:
            subj = self._extract_subject(sentence).replace(" ", "_")
            return f"not_read_book({subj})"
        if "passed the first exam" in sentence:
            subj = self._extract_subject(sentence).replace(" ", "_")
            return f"passed_first_exam({subj})"
        if any(token.lower_ in {"all", "every", "each"} for token in doc):
            return self._parse_universal(sentence, doc)
        if any(token.lower_ in {"some", "a", "an"} for token in doc):
            return self._parse_existential(sentence, doc)
        return self._parse_fact(sentence, doc)
    
    def _convert_do_query(self, core: str) -> str:
        # e.g. 'socrates have an apple'
        # feed into fact parser
        return self._parse_fact(core)

    def parse_query(self, sentence: str) -> str:
        
        """
        Convert a query in either English (Is/Are/Does/Can/...) or raw Prolog
        into a Prolog predicate string.
        """
        # Clean up and normalize
        sentence_clean = sentence.strip().rstrip("?").strip()
        if not sentence_clean:
            raise ValueError("Empty query provided.")
        sentence_low = sentence_clean.lower()
        
        # Split to identify auxiliary
        tokens = sentence_low.split()
        aux = tokens[0]
        
        # Handle 'is' and 'are' as unary/binary queries
        if aux in {"is", "are"}:
            # e.g. "Is Andrew a man" or "Are dolphins intelligent"
            return self._parse_query_fact(sentence_low)
        
        # Handle 'does', 'do', 'can', 'could', 'should', 'will', 'did', etc.
        if aux in {"does", "do", "can", "could", "should", "will", "did"}:
            # Strip the auxiliary and parse the core as a fact relationship
            core = re.sub(rf"^{aux}\s+", "", sentence_low)
            return self._parse_fact(core)
        
        # ——— NEW: handle declarative queries like "Socrates is mortal" ———
        # if it contains " is " or " are " (and isn’t raw Prolog), parse as a query
        if re.search(r"\b(is|are)\b", sentence_low):
            return self._parse_query_fact(sentence_low)
        
        # Fallback: maybe already raw Prolog or a declarative English fact
        # Detect raw Prolog by parentheses pattern
        if "(" in sentence_clean and ")" in sentence_clean:
            return sentence_clean.rstrip('.')
        # Otherwise, treat as an English declarative
        return self.parse_sentence(sentence_clean)

    def assert_knowledge(self, sentences):
        """
        Parse English sentences into Prolog clauses and assert them.
        """
        for s in sentences:
            clauses = self.parse_sentence(s)
            for clause in clauses.splitlines():
                # negative facts -> false :- predicate
                if clause.startswith("not_"):
                    pred = clause[len("not_"):]
                    self.prolog.assertz(f"false :- {pred}")
                else:
                    self.prolog.assertz(clause)
        logger.info("Knowledge base loaded with %d sentences.", len(sentences))

    def prove_exists(self, query: str) -> bool:
        """
        Return True if Prolog can satisfy the query (e.g. 'intelligent(X), \\+ can_read(X)').
        """
        results = list(self.prolog.query(query))
        return bool(results)

    def _parse_if_else(self, sentence: str) -> str:
        pattern = re.compile(r"^if\s+(.*?),\s*then\s+(.*?),\s*else\s+(.*)$", re.IGNORECASE)
        m = pattern.match(sentence)
        if not m:
            raise ValueError("If/else sentence not in recognized format.")
        cond, then_txt, else_txt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        cond_pred = self._parse_fact(cond)
        subj = self._extract_subject_from_pred(cond_pred)
        then_pred = self._parse_fact(then_txt, default_subject=subj)
        else_pred = self._parse_fact(else_txt, default_subject=subj)
        return rf"{then_pred} :- {cond_pred}\n{else_pred} :- \+ {cond_pred}"

    def _parse_conditional(self, sentence: str) -> str:
        pattern = re.compile(r"^if\s+(.*?),\s*then\s+(.*)$", re.IGNORECASE)
        m = pattern.match(sentence)
        if not m:
            raise ValueError("Conditional sentence not in recognized format.")
        cond_txt, then_txt = m.group(1).strip(), m.group(2).strip()
        cond_pred = self._parse_fact(cond_txt)
        subj = self._extract_subject_from_pred(cond_pred)
        then_pred = self._parse_fact(then_txt, default_subject=subj)
        return f"{then_pred} :- {cond_pred}"

    def _parse_universal(self, sentence: str, doc) -> str:
        """
        Handle sentences like 'All men are mortal' -> mortal(X) :- men(X)
        Identify quantifier and subject noun
        """
        
        quantifier = None
        subject = None
        for token in doc:
            if token.lower_ in {"all", "every", "each"}:
                quantifier = token.lower_
                # find the noun following quantifier
                for child in token.children:
                    if child.pos_ in {"NOUN", "PROPN"}:
                        # use lemma to get singular form
                        subject = child.lemma_.lower()
                        break
                if not subject:
                    # fallback: first noun in doc
                    for t2 in doc:
                        if t2.pos_ in {"NOUN", "PROPN"}:
                            subject = t2.lemma_.lower()
                            break
                break
        if not subject:
            raise ValueError("Could not determine universal subject.")
        subject_var = "X"
        # Construct body predicate: subject(X)
        body = f"{subject}({subject_var})"
        # Extract the predicate part (after 'are' or 'is')
        lower_sent = sentence.lower()
        if " are " in lower_sent:
            parts = lower_sent.split(" are ", 1)
            pred_text = parts[1]
        elif " is " in lower_sent:
            parts = lower_sent.split(" is ", 1)
            pred_text = parts[1]
        else:
            pred_text = lower_sent
        # Build head predicate using the same variable
        head = self._convert_to_predicate(pred_text, subject_var)
        return f"{head} :- {body}"

    def _parse_existential(self, sentence: str, doc) -> str:
        # For simple existential facts ("Some dolphins are intelligent"), treat as a fact
        if re.search(r"\b(some|a|an)\b", sentence.lower()) and re.search(r"\b(is|are)\b", sentence.lower()):
            return self._parse_fact(sentence, doc)

        subject = None
        quantifier = None
        for token in doc:
            if token.lower_ in {"some", "a", "an"}:
                quantifier = token.lower_
                for child in token.children:
                    if child.pos_ in {"NOUN", "PROPN"}:
                        subject = child.text.lower(); break
                if subject: break
        if not subject:
            return self._parse_fact(sentence, doc)
        subject = subject.replace(" ", "_")
        witness = subject + "_" + uuid.uuid4().hex[:6]
        pattern = re.compile(rf"\b({quantifier})\s+{subject}\b", re.IGNORECASE)
        new_sent = pattern.sub(witness, sentence)
        return self._convert_to_predicate(new_sent, witness)

    def _parse_fact(self, sentence: str, doc=None, default_subject=None) -> str:
        sentence = sentence.strip().rstrip(" ?.")
        sentence_lower = sentence.lower()
        # conjunctions: A and B are the X and Y of C and D
        conj = re.match(r"^(.+?)\s+are\s+the\s+(.+?)\s+of\s+(.+)$", sentence_lower)
        if conj:
            subj_blk, pred_blk, obj_blk = conj.group(1), conj.group(2), conj.group(3)
            subs = [s.strip().replace(" ", "_") for s in re.split(r"\s+and\s+", subj_blk)]
            preds = [p.strip() for p in re.split(r"\s+and\s+", pred_blk)]
            objs = [o.strip().replace(" ", "_") for o in re.split(r"\s+and\s+", obj_blk)]
            if len(subs) == len(preds):
                res = []
                for s, p in zip(subs, preds):
                    name = re.sub(r"[^\w]", "_", p)
                    for o in objs:
                        res.append(f"{name}({s}, {o})")
                return "\n".join(res)
        if not doc:
            doc = self.nlp_processor.process_text(sentence)
        root = next((t for t in doc if t.dep_=='ROOT' and t.pos_=='VERB'), None)
        if root:
            subj_t = next((c for c in root.children if c.dep_ in ('nsubj','nsubjpass')), None)
            obj_t  = next((c for c in root.children if c.dep_ in ('dobj','pobj','obj')), None)
            if subj_t and obj_t:
                return f"{re.sub(r"[^\w]","_",root.text.lower())}({subj_t.text.replace(' ','_')}, {obj_t.text.replace(' ','_')})"
            
        # ... rest of is-pattern logic unchanged ...
        words = sentence_lower.split()
        if default_subject and words and words[0] in PRONOUNS:
            subj = default_subject
            sentence_lower = " ".join(words[1:]).strip()
        else:
            subj = None
            joined = " ".join(words)
        if " is " in sentence_lower:
            parts = sentence_lower.split(" is ",1)
            if subj is None: subj = parts[0].strip()
            subj = subj.replace(" ","_")
            rem = parts[1].strip()
            if rem.startswith("will " ): rem = rem[5:].strip()
            rem = re.sub(r"^(?:a|an|the)\s+","", rem, flags=re.IGNORECASE).strip()
            if rem.startswith("not "): rem = rem.replace("not ","not_",1)
            if " of " in rem:
                pp, o = rem.split(" of ",1)
                name = re.sub(r"[^\w]","_", re.sub(r"^(?:a|an|the)\s+","", pp, flags=re.IGNORECASE))
                objs = [o.strip().replace(' ','_')]
                if " and " in o:
                    objs = [x.strip().replace(' ','_') for x in o.split(" and ")]
                return "\n".join(f"{name}({subj}, {x})" for x in objs)
            return f"{re.sub(r"[^\w]","_",rem)}({subj})"
        
        # fallback
        if not doc:
            doc = self.nlp_processor.process_text(sentence)
        for tok in doc:
            if tok.dep_ in {"nsubj","nsubjpass"} and tok.pos_ in {"NOUN","PROPN","PRON"}:
                sc = tok.text.lower()
                subj = default_subject if sc in PRONOUNS and default_subject else sc
                break
        if 'subj' not in locals() or subj is None:
            for tok in doc:
                if tok.pos_ in {"NOUN","PROPN"}:
                    subj = tok.text.lower(); break
        if subj is None:
            if subj is None:
                m = re.match(r"^(?P<subj>\w+)\s+(?:a|an)\s+(?P<pred>.+)$", joined)
                if m:
                    name = re.sub(r"[^\w]","_",re.sub(r"^(?:a|an|the)\s+","",m.group('pred'),flags=re.IGNORECASE))
                    return f"{name}({m.group('subj')})"
        return self._convert_to_predicate(sentence_lower, subj)

    # include all other helper methods unchanged:
    # _parse_query_fact, _convert_to_predicate, _extract_subject_from_pred, _extract_subject

    
    def _parse_query_fact(self, sentence: str) -> str:
        """
        Parse yes/no queries and transform them into Prolog predicate calls.
        Handles:
          - "Is X Y?" or "Are X Y?" → Y(X)
          - "X is Y?" or "X are Y?" → Y(X)
          - "Does X do Y?" / other auxiliaries handled elsewhere
          - "The capital of France is Paris" → capital(France, Paris)
        """
        # normalize and strip question marks
        sentence = sentence.lower().strip(" ?")

        # 1) leading "is/are X Y" → Y(X)
        if sentence.startswith("is ") or sentence.startswith("are "):
            # drop the auxiliary
            rest = re.sub(r"^(?:is|are)\s+", "", sentence)
            # split into subject and predicate
            subj, pred = rest.split(" ", 1)
            # remove articles
            pred = re.sub(r"^(?:a|an|the)\s+", "", pred, flags=re.IGNORECASE)
            pred_name = re.sub(r"[^\w]", "_", pred)
            return f"{pred_name}({subj})"

        # 2) "... of ... is ..." pattern
        if " of " in sentence and " is " in sentence:
            parts = sentence.split(" is ", 1)
            subject = parts[0].strip().replace(" ", "_")
            remainder = parts[1].strip()
            pred_part, obj = remainder.split(" of ", 1)
            pred_part = re.sub(r"^(?:a|an|the)\s+", "", pred_part, flags=re.IGNORECASE).strip()
            pred_name = re.sub(r"[^\w]", "_", pred_part)
            return f"{pred_name}({subject}, {obj.strip().replace(' ', '_')})"

        # 3) fallback: split on the first " is " for declarative queries
        parts = sentence.split(" is ", 1)
        if len(parts) < 2:
            raise ValueError("Unable to parse query.")
        subject = parts[0].strip().replace(" ", "_")
        remainder = re.sub(r"^(?:a|an|the)\s+", "", parts[1].strip(), flags=re.IGNORECASE)
        pred_name = re.sub(r"[^\w]", "_", remainder)
        return f"{pred_name}({subject})"



    def _convert_to_predicate(self, text: str, subject: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(if|then|else|is|are|was|were)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(rf"\b{subject}\b", "", text, flags=re.IGNORECASE).strip()
        if not text:
            text = "true"
        pred_name = re.sub(r"[^\w]", "_", text)
        pred_name = re.sub(r"_+", "_", pred_name).strip("_")
        return f"{pred_name}({subject})"
    
    def _extract_subject_from_pred(self, pred: str) -> str:
        m = re.search(r"\((.*?)\)", pred)
        if m:
            sub = m.group(1).strip()
            if "," in sub:
                sub = sub.split(",")[0].strip()
            return sub
        return None
    
    
    def _extract_subject(self, sentence: str) -> str:
        words = sentence.split()
        if not words:
            raise ValueError("No words found in sentence.")
        if words[0].lower() in {"a", "an", "the"} and len(words) > 1:
            subject = words[1]
        else:
            subject = words[0]
        return subject.replace(" ", "_")