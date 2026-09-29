from transformers import pipeline


class NERModel:
    def __init__(self):
        self.ner = pipeline(
            task="ner",
            model="NYTK/named-entity-recognition-nerkor-hubert-hungarian",
            aggregation_strategy="simple"
        )

    def extract_entities(self, text: str):
        return self.ner(text)