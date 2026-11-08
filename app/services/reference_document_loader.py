from pathlib import Path


REFERENCE_DOCUMENT_FILENAMES = (
    "OTP-Uzletszabalyzat.txt",
    "2012. évi CXVI. törvény a pénzügyi.txt",
    "OTP-HIRDETMÉNY.txt",
)


class ReferenceDocumentLoader:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else Path.home()

    def load(self) -> dict[str, str]:
        documents: dict[str, str] = {}
        for filename in REFERENCE_DOCUMENT_FILENAMES:
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
