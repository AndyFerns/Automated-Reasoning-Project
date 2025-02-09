import nltk
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.chunk import ne_chunk

class PredicateExtractor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        self.predicates = []

    def extract_predicates(self, text):
        sentences = nltk.sent_tokenize(text)
        extracted_predicates = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            chunked = ne_chunk(tagged)
            subject, predicate, obj = None, None, None
            
            for i, (word, tag) in enumerate(tagged):
                if tag.startswith('NN'):
                    if not subject:
                        subject = word
                    else:
                        obj = word
                elif tag.startswith('VB'):
                    predicate = word
            
            if subject and predicate:
                extracted_predicates.append((subject, predicate, obj))
        
        self.predicates.extend(extracted_predicates)
        return extracted_predicates

    def store_predicates(self, filename="predicates.txt"):
        with open(filename, "w") as file:
            for subject, predicate, obj in self.predicates:
                file.write(f"{subject}({predicate}, {obj})\n")

if __name__ == "__main__":
    text = "John reads books. Mary loves programming. The dog chases the cat."
    extractor = PredicateExtractor()
    predicates = extractor.extract_predicates(text)
    
    print("Extracted Predicates:")
    for pred in predicates:
        print(pred)
    
    extractor.store_predicates()
