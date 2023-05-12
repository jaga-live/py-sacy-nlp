import re
import spacy
from spacy.training import Example
from spacy.matcher import Matcher

# Define the email regex pattern
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Define the training data
TRAIN_DATA = [
    ("find email adam@gmail.com", {"entities": [(12, 25, "EMAIL")]}),
    ("find email john@gmail.com", {"entities": [(12, 25, "EMAIL")]}),
    ("find email sami@gmail.com", {"entities": [(12, 25, "EMAIL")]}),
    ("find email robes@gmail.com", {"entities": [(12, 25, "EMAIL")]}),
    ("find email wills@gmail.com", {"entities": [(12, 25, "EMAIL")]}),
    ("find email woods@gmail.com", {"entities": [(12, 25, "EMAIL")]}),
    ("find email bill at gmail.com", {"entities": [(11, 24, "EMAIL")]}),
    ("find email adam at gmail.com", {"entities": [(10, 26, "EMAIL")]})
]

def train_spacy_ner(training_data, iterations=20):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    matcher = Matcher(nlp.vocab)
    matcher.add("EMAIL", None, [{"TEXT": {"REGEX": EMAIL_PATTERN}}])
    
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            losses = {}
            for text, annotations in training_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
                matches = matcher(nlp(text))
                for match_id, start, end in matches:
                    entities = annotations.get("entities", [])
                    entities.append((start, end, "EMAIL"))
                    annotations["entities"] = entities
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print("Iteration:", itn + 1, "Loss:", losses["ner"])
    
    return nlp

# Train the NER model
ner_model = train_spacy_ner(TRAIN_DATA)

# Test the trained model
test_texts = [
    "find email adam@gmail.com",
    "find email adam at gmail.com",
    "find email adam@gmail.com",
]

for text in test_texts:
    doc = ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("Text:", text)
    print("Entities:", entities)
    print()
