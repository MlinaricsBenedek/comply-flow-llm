from collections.abc import Sequence


class MessageClassifier:
	CATEGORIES = (
		"REFUND",
		"DEFECTIVE_PRODUCT",
		"DELIVERY",
		"WARRANTY",
		"CANCELLATION",
		"GENERAL_COMPLAINT",
		"OTHER",
	)

	def __init__(
		self,
		max_tokens: int = 10_000,
		sequence_length: int = 100,
		embedding_dimension: int = 64,
	):
		self.max_tokens = max_tokens
		self.sequence_length = sequence_length
		self.embedding_dimension = embedding_dimension
		self._model = None
		self._vectorizer = None

	def train(
		self,
		texts: Sequence[str],
		labels: Sequence[str],
		epochs: int = 10,
		batch_size: int = 32,
		verbose: int = 1,
	):
		import tensorflow as tf

		texts = list(texts)
		labels = list(labels)

		if not texts:
			raise ValueError("Training data must not be empty.")
		if len(texts) != len(labels):
			raise ValueError("texts and labels must have the same number of items.")
		if any(not isinstance(text, str) or not text.strip() for text in texts):
			raise ValueError("Training texts must be non-empty strings.")

		invalid_labels = set(labels) - set(self.CATEGORIES)
		if invalid_labels:
			raise ValueError(f"Unsupported categories: {sorted(invalid_labels)}")

		vectorizer = tf.keras.layers.TextVectorization(
			max_tokens=self.max_tokens,
			output_mode="int",
			output_sequence_length=self.sequence_length,
		)
		vectorizer.adapt(texts)

		inputs = tf.keras.Input(shape=(), dtype=tf.string)
		tokens = vectorizer(inputs)
		embeddings = tf.keras.layers.Embedding(
			input_dim=len(vectorizer.get_vocabulary()),
			output_dim=self.embedding_dimension,
			mask_zero=True,
		)(tokens)
		features = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
		features = tf.keras.layers.Dense(64, activation="relu")(features)
		features = tf.keras.layers.Dropout(0.2)(features)
		outputs = tf.keras.layers.Dense(len(self.CATEGORIES), activation="softmax")(features)

		model = tf.keras.Model(inputs=inputs, outputs=outputs)
		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"],
		)

		label_indices = [self.CATEGORIES.index(label) for label in labels]
		training_data = tf.data.Dataset.from_tensor_slices(
			(tf.constant(texts), tf.constant(label_indices, dtype=tf.int32))
		).shuffle(buffer_size=len(texts)).batch(batch_size)
		history = model.fit(
			training_data,
			epochs=epochs,
			verbose=verbose,
			shuffle=False,
		)
		self._model = model
		self._vectorizer = vectorizer
		return history

	def classify(self, text: str) -> str:
		import tensorflow as tf

		if self._model is None:
			raise RuntimeError("The classifier has not been trained yet.")
		if not isinstance(text, str) or not text.strip():
			raise ValueError("text must be a non-empty string.")

		scores = self._model(tf.constant([text]), training=False).numpy()[0]
		return self.CATEGORIES[int(scores.argmax())]
