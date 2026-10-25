import re

from app.models.ner_model import NERModel


class FactExtractor:
	_product_pattern = re.compile(
		r"\b(?:a\s+)?termék(?:et)?"
		r"(?:\s+(?:amit\s+)?(?:vásároltam|vettem))?"
		r"\s*[:=-]\s*(?P<product>[^,.;\n]+)",
		re.IGNORECASE,
	)
	_price_pattern = re.compile(
		r"(?<!\w)\d+(?:[ .]\d{3})*(?:,\d{1,2})?\s*(?:Ft|HUF|forint)\b",
		re.IGNORECASE,
	)
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
		for match in self._product_pattern.finditer(text):
			self._add_fact(facts, "PRODUCT", match.group("product").strip())

		for match in self._price_pattern.finditer(text):
			self._add_fact(facts, "PRICE", match.group().strip())

		for entity in self.ner_model.extract_entities(text):
			label = entity.get("entity_group") or entity.get("entity") or entity.get("label")
			value = entity.get("word") or entity.get("text")

			if not label or not isinstance(value, str):
				continue

			label = str(label).upper().removeprefix("B-").removeprefix("I-")
			label = self._entity_labels.get(label, label)
			value = value.strip()
			self._add_fact(facts, label, value)

		return facts

	@staticmethod
	def _add_fact(facts: dict[str, list[str]], label: str, value: str) -> None:
		if not value:
			return

		values = facts.setdefault(label, [])
		if value not in values:
			values.append(value)
