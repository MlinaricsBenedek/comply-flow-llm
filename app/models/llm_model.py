from threading import Lock
from typing import Any


class HuggingFaceChatModel:
	def __init__(
		self,
		model_name: str,
		max_new_tokens: int = 512,
		do_sample: bool = False,
		temperature: float = 1.0,
		top_p: float = 1.0,
	) -> None:
		if max_new_tokens <= 0:
			raise ValueError("max_new_tokens must be greater than zero.")
		if temperature <= 0:
			raise ValueError("temperature must be greater than zero.")
		if not 0 < top_p <= 1:
			raise ValueError("top_p must be greater than zero and at most one.")

		self.model_name = model_name
		self.max_new_tokens = max_new_tokens
		self.do_sample = do_sample
		self.temperature = temperature
		self.top_p = top_p
		self._tokenizer: Any | None = None
		self._model: Any | None = None
		self._load_lock = Lock()

	def generate(
		self,
		prompt: str,
		**generation_parameters: int | float | bool,
	) -> str:
		if not prompt.strip():
			raise ValueError("Prompt must not be empty.")

		max_new_tokens = generation_parameters.get("max_new_tokens", self.max_new_tokens)
		do_sample = generation_parameters.get("do_sample", self.do_sample)
		temperature = generation_parameters.get("temperature", self.temperature)
		top_p = generation_parameters.get("top_p", self.top_p)

		import torch

		tokenizer, model = self._load()
		formatted_prompt = tokenizer.apply_chat_template(
			[{"role": "user", "content": prompt}],
			tokenize=False,
			add_generation_prompt=True,
		)
		model_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(
			model.device
		)

		pad_token_id = tokenizer.pad_token_id
		if pad_token_id is None:
			pad_token_id = tokenizer.eos_token_id

		with torch.inference_mode():
			generation_options = {
				"max_new_tokens": max_new_tokens,
				"do_sample": do_sample,
				"pad_token_id": pad_token_id,
			}
			if do_sample:
				generation_options.update(
					temperature=temperature,
					top_p=top_p,
				)

			generated_tokens = model.generate(
				**model_inputs,
				**generation_options,
			)

		input_length = model_inputs["input_ids"].shape[-1]
		response = tokenizer.decode(
			generated_tokens[0][input_length:],
			skip_special_tokens=True,
		)
		return response.strip()

	def _load(self) -> tuple[Any, Any]:
		if self._model is None:
			with self._load_lock:
				if self._model is None:
					from transformers import AutoModelForCausalLM, AutoTokenizer

					tokenizer = AutoTokenizer.from_pretrained(self.model_name)
					model = AutoModelForCausalLM.from_pretrained(
						self.model_name,
						torch_dtype="auto",
						device_map="auto",
					)
					model.eval()
					self._tokenizer = tokenizer
					self._model = model

		return self._tokenizer, self._model


class QwenModel(HuggingFaceChatModel):
	def __init__(
		self,
		model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
		max_new_tokens: int = 512,
		do_sample: bool = False,
		temperature: float = 1.0,
		top_p: float = 1.0,
	) -> None:
		super().__init__(model_name, max_new_tokens, do_sample, temperature, top_p)


class LlamaModel(HuggingFaceChatModel):
	def __init__(
		self,
		model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
		max_new_tokens: int = 512,
		do_sample: bool = False,
		temperature: float = 1.0,
		top_p: float = 1.0,
	) -> None:
		super().__init__(model_name, max_new_tokens, do_sample, temperature, top_p)
