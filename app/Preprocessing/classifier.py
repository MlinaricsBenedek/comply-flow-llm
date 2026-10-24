from collections.abc import Callable
from threading import Lock
from typing import Any


class MessageClassifier:
	MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
	HYPOTHESIS_TEMPLATE = "Az ügyfél panasza {}."
	CANDIDATE_LABELS = {
		"REFUND": "visszatérítéssel vagy pénzvisszafizetéssel kapcsolatos",
		"DEFECTIVE_PRODUCT": "hibás vagy sérült termékkel kapcsolatos",
		"DELIVERY": "szállítással vagy kézbesítéssel kapcsolatos",
		"WARRANTY": "garanciával kapcsolatos",
		"CANCELLATION": "rendelés vagy szolgáltatás lemondásával kapcsolatos",
		"GENERAL_COMPLAINT": "általános ügyfélpanasz",
		"OTHER": "egyik felsorolt kategóriába sem illeszkedő ügy",
	}

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

		classifier = self._load()
		result = classifier(
			text,
			candidate_labels=list(self.CANDIDATE_LABELS.values()),
			hypothesis_template=self.HYPOTHESIS_TEMPLATE,
			multi_label=False,
		)
		labels = result.get("labels")
		if not isinstance(labels, list) or not labels:
			raise RuntimeError("The zero-shot classifier returned no labels.")

		try:
			return self._labels_by_description[labels[0]]
		except KeyError as error:
			raise RuntimeError("The zero-shot classifier returned an unknown label.") from error

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
