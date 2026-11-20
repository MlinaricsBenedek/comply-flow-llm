import os
from pathlib import Path

from dotenv import dotenv_values


def hugging_face_token_options() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token is None:
        token = dotenv_values(Path(__file__).resolve().parents[1] / ".env").get("HF_TOKEN")
    return {"token": token} if token else {}
