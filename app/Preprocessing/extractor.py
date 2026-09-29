from app.models.ner_model import NERModel


class FactExtractor:
	_entity_labels = {
		"PER": "PERSON",
		"PERSON": "PERSON",
		"LOC": "LOCATION",
		"LOCATION": "LOCATION",
		"ORG": "ORGANIZATION",
		"ORGANIZATION": "ORGANIZATION",
	}

	def __init__(self, ner_model: NERModel | None = None):
		self.ner_model = ner_model if ner_model is not None else NERModel()

	def extract(self, text: str | None) -> dict[str, list[str]]:
		if not text or not text.strip():
			return {}

		facts: dict[str, list[str]] = {}
		for entity in self.ner_model.extract_entities(text):
			label = entity.get("entity_group") or entity.get("entity") or entity.get("label")
			value = entity.get("word") or entity.get("text")

			if not label or not isinstance(value, str):
				continue

			label = str(label).upper().removeprefix("B-").removeprefix("I-")
			label = self._entity_labels.get(label, label)
			value = value.strip()
			if not value:
				continue

			values = facts.setdefault(label, [])
			if value not in values:
				values.append(value)

		return facts
