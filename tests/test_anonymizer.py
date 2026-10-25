import unittest

from app.Preprocessing.anonymizer import MessageAnonymizer


class MessageAnonymizerTests(unittest.TestCase):
    def test_anonymizes_name_after_hungarian_name_introduction(self) -> None:
        self.assertEqual(
            MessageAnonymizer.anonymize("A nevem Kovács Anna."),
            "A nevem [NAME].",
        )

    def test_anonymizes_name_before_vagyok(self) -> None:
        self.assertEqual(
            MessageAnonymizer.anonymize("Kovács Anna vagyok."),
            "[NAME] vagyok.",
        )

    def test_anonymizes_name_in_existing_label_format(self) -> None:
        self.assertEqual(
            MessageAnonymizer.anonymize("Név: Kovács Anna"),
            "Név: [NAME]",
        )

    def test_anonymizes_email_and_hungarian_phone_number(self) -> None:
        self.assertEqual(
            MessageAnonymizer.anonymize(
                "Email: benedek@example.com, telefon: +36 30 123 4567"
            ),
            "Email: [EMAIL], telefon: [PHONE]",
        )

    def test_does_not_replace_unrelated_capitalized_phrase(self) -> None:
        message = "Nagyon boldog vagyok."

        self.assertEqual(MessageAnonymizer.anonymize(message), message)

    def test_returns_empty_string_for_none_or_empty_input(self) -> None:
        self.assertEqual(MessageAnonymizer.anonymize(None), "")
        self.assertEqual(MessageAnonymizer.anonymize(""), "")


if __name__ == "__main__":
    unittest.main()
