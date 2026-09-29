import re
import unicodedata


class MessageCleaner:
    @staticmethod
    def clean(message: str | None) -> str:
        if not message:
            return ""

        message = unicodedata.normalize("NFC", message)
        message = message.replace("\r\n", "\n").replace("\r", "\n")

        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in message.split("\n")
        ]

        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()