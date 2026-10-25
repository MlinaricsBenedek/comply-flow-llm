from collections.abc import Callable
import math
import re
from threading import Lock
from typing import Any


class MessageClassifier:
	MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
	HYPOTHESIS_TEMPLATE = "Az ügyfél panasza {}."
	MIN_MODEL_CONFIDENCE = 0.7
	CANDIDATE_LABELS = {
		"REFUND": "visszatérítéssel vagy pénzvisszafizetéssel kapcsolatos",
		"DEFECTIVE_PRODUCT": "hibás vagy sérült termékkel kapcsolatos",
		"DELIVERY": "szállítással vagy kézbesítéssel kapcsolatos",
		"WARRANTY": "garanciával kapcsolatos",
		"CANCELLATION": "rendelés vagy szolgáltatás lemondásával kapcsolatos",
		"GENERAL_COMPLAINT": "általános ügyfélpanasz",
		"OTHER": "egyik felsorolt kategóriába sem illeszkedő ügy",
	}
	_RULES = (
		(
			"REFUND",
			re.compile(
				r"\b(?:visszatérít\w*|visszafizet\w*|visszaigényel\w*|"
				r"visszakap\w*|pénzem(?:et)?\s+vissza|vételár\w*\s+vissza)\b"
			),
		),
		(
			"CANCELLATION",
			re.compile(
				r"\b(?:(?:lemond\w*|töröl\w*)\b.{0,40}\b(?:rendel\w*|szolgáltat\w*)"
				r"|(?:rendel\w*|szolgáltat\w*)\b.{0,40}\b(?:lemond\w*|töröl\w*))\b"
			),
		),
		(
			"WARRANTY",
			re.compile(r"\b(?:garanci\w*|jótáll\w*|szerviz\w*)\b"),
		),
		(
			"DELIVERY",
			re.compile(
				r"\b(?:szállít\w*|kiszállít\w*|kézbesít\w*|futár\w*|"
				r"csomag\w*|rendelésem\s+nem\s+érkezett)\b"
			),
		),
		(
			"DEFECTIVE_PRODUCT",
			re.compile(
				r"\b(?:elromlott|hibás|meghibásod\w*|sérült|"
				r"nem\s+működik|nem\s+kapcsol\s+be)\b"
			),
		),
		(
			"GENERAL_COMPLAINT",
			re.compile(r"\b(?:panasz\w*|ügyintézés\w*)\b"),
		),
		(
			"OTHER",
			re.compile(
				r"\b(?:nyitvatart\w*|nyitva\s+vannak|mikor\s+nyit\w*|"
				r"készleten\s+van|termékinformáció\w*)\b"
			),
		),
	)

	def __init__(
		self,
		model_name: str = MODEL_NAME,
		pipeline_factory: Callable[..., Any] | None = None,
	) -> None:
		self.model_name = model_name
		self._pipeline_factory = pipeline_factory
		self._classifier: Any | None = None
		self._load_lock = Lock()
		self._labels_by_description = {
			description: category
			for category, description in self.CANDIDATE_LABELS.items()
		}

	def classify(self, text: str) -> str:
		if not isinstance(text, str) or not text.strip():
			raise ValueError("text must be a non-empty string.")

		normalized_text = text.casefold()
		for category, pattern in self._RULES:
			if pattern.search(normalized_text):
				return category

		classifier = self._load()
		result = classifier(
			normalized_text,
			candidate_labels=list(self.CANDIDATE_LABELS.values()),
			hypothesis_template=self.HYPOTHESIS_TEMPLATE,
			multi_label=False,
		)
		labels = result.get("labels")
		scores = result.get("scores")
		if (
			not isinstance(labels, list)
			or not labels
			or not isinstance(scores, list)
			or len(labels) != len(scores)
		):
			raise RuntimeError("The zero-shot classifier returned no labels.")

		try:
			category = self._labels_by_description[labels[0]]
		except KeyError as error:
			raise RuntimeError("The zero-shot classifier returned an unknown label.") from error
		try:
			confidence = float(scores[0])
		except (TypeError, ValueError) as error:
			raise RuntimeError("The zero-shot classifier returned an invalid score.") from error
		if not math.isfinite(confidence) or not 0 <= confidence <= 1:
			raise RuntimeError("The zero-shot classifier returned an invalid score.")
		if confidence < self.MIN_MODEL_CONFIDENCE:
			return "OTHER"
		return category

	def _load(self) -> Any:
		if self._classifier is None:
			with self._load_lock:
				if self._classifier is None:
					pipeline_factory = self._pipeline_factory
					if pipeline_factory is None:
						from transformers import pipeline

						pipeline_factory = pipeline
					self._classifier = pipeline_factory(
						task="zero-shot-classification",
						model=self.model_name,
					)

		return self._classifier
