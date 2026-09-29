import re


class MessageAnonymizer:
	_patterns = (
		("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
		(
			"PHONE",
			re.compile(r"(?<!\w)(?:\+36|0036|06)[\s()./-]?(?:\d[\s()./-]?){7,9}(?!\w)"),
		),
		(
			"IBAN",
			re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b", re.IGNORECASE),
		),
		(
			"CARD",
			re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
		),
		(
			"ID",
			re.compile(
				r"\b(?:TAJ(?:-szám)?|adóazonosító(?:\s+jel)?|adószám)\s*[:#]?\s*"
				r"[0-9][0-9 -]{7,11}\b",
				re.IGNORECASE,
			),
		),
		(
			"NAME",
			re.compile(
				r"\b(?:nevem|név|name)\s*[:=-]\s*"
				r"[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+"
				r"(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+){1,2}",
				re.IGNORECASE,
			),
		),
	)

	@classmethod
	def anonymize(cls, message: str | None) -> str:
		if not message:
			return ""

		for label, pattern in cls._patterns:
			message = pattern.sub(f"[{label}]", message)

		return message
