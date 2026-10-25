import unittest

from app.Preprocessing.cleaner import MessageCleaner


class MessageCleanerTests(unittest.TestCase):
    def test_normalizes_unicode_to_nfc(self) -> None:
        self.assertEqual(MessageCleaner.clean("e\u0301"), "\u00e9")

    def test_normalizes_line_endings(self) -> None:
        self.assertEqual(
            MessageCleaner.clean("first\r\nsecond\rthird"),
            "first\nsecond\nthird",
        )

    def test_collapses_whitespace_within_lines(self) -> None:
        self.assertEqual(
            MessageCleaner.clean("  hello\t  world  \n  next   line "),
            "hello world\nnext line",
        )

    def test_preserves_paragraphs_and_limits_blank_lines(self) -> None:
        self.assertEqual(
            MessageCleaner.clean("first\n\n\n\nsecond"),
            "first\n\nsecond",
        )

    def test_returns_empty_string_for_none_or_whitespace(self) -> None:
        self.assertEqual(MessageCleaner.clean(None), "")
        self.assertEqual(MessageCleaner.clean(" \t\r\n "), "")


if __name__ == "__main__":
    unittest.main()
