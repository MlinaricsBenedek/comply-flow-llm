import os
from pathlib import Path


DEFAULT_PROMPT_DIR = Path(r"C:\Users\mlina\Documents\Projektmappa\Prompt")


class PromptService:
    def __init__(self, prompt_dir: str | Path | None = None) -> None:
        configured_dir = prompt_dir or os.getenv("PROMPT_DIR") or DEFAULT_PROMPT_DIR
        self.prompt_dir = Path(configured_dir)
        self._prompts: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        if not self.prompt_dir.is_dir():
            raise FileNotFoundError(f"Prompt directory does not exist: {self.prompt_dir}")

        self._prompts = {
            prompt_file.stem: prompt_file.read_text(encoding="utf-8-sig")
            for prompt_file in sorted(self.prompt_dir.iterdir())
            if prompt_file.is_file() and prompt_file.suffix.lower() == ".txt"
        }

    def get_prompt(self, name: str) -> str:
        try:
            return self._prompts[name]
        except KeyError as error:
            raise KeyError(f"Prompt not found: {name}") from error

    def get_prompts(self) -> dict[str, str]:
        return self._prompts.copy()