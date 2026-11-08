import os
from pathlib import Path


SUPPORTED_BANKS = ("OTP", "Raiffeisen")
BANK_DOCUMENT_FILENAMES = {
    "OTP": (
        "OTP-Uzletszabalyzat.txt",
        "2012. évi CXVI. törvény a pénzügyi.txt",
        "OTP-HIRDETMÉNY.txt",
    ),
    "Raiffeisen": (
        "2012. évi CXVI. törvény a pénzügyi.txt",
        "HIRDETENY-Raiffeisen.txt",
        "Lakossagi-uzletszabalyzat-Raiffeisen.txt",
    ),
}
REFERENCE_DOCUMENT_FILENAMES = BANK_DOCUMENT_FILENAMES["OTP"]


class ReferenceDocumentLoader:
    def __init__(
        self,
        directory: str | Path | None = None,
        bank: str = "OTP",
    ) -> None:
        if bank not in BANK_DOCUMENT_FILENAMES:
            raise ValueError(f"Unsupported bank: {bank}")
        if directory is not None:
            self.directory = Path(directory)
        else:
            setting_name = f"{bank.upper()}_RULES_DIR"
            configured_directory = os.getenv(setting_name)
            if configured_directory is None and bank == "OTP":
                configured_directory = os.getenv("REFERENCE_DOCUMENTS_DIR")
            if configured_directory is not None:
                configured_directory = configured_directory.strip()
                if not configured_directory:
                    raise ValueError(f"{setting_name} must not be empty.")
                self.directory = Path(configured_directory)
            else:
                self.directory = Path.home() / bank
        self.bank = bank

    def load(self) -> dict[str, str]:
        documents: dict[str, str] = {}
        for filename in BANK_DOCUMENT_FILENAMES[self.bank]:
            content = (self.directory / filename).read_bytes()
            documents[filename] = self._decode(content)
        return documents

    @staticmethod
    def _decode(content: bytes) -> str:
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            return content.decode("utf-16")
        if content.startswith(b"\xef\xbb\xbf"):
            return content.decode("utf-8-sig")
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("cp1250")
