import unittest

from app.Preprocessing.extractor import FactExtractor
from app.Preprocessing.anonymizer import MessageAnonymizer
from app.Preprocessing.cleaner import MessageCleaner
from app.Preprocessing.state_builder import StateBuilder
from app.models.ner_model import NERModel
from app.schemas.preprocessing import PreprocessingResponse


class NERModelTests(unittest.TestCase):
    def test_extracts_product_and_price_from_anonymized_complaint(self) -> None:
        message = (
            "A nevem Kovács Anna; email: anna@example.com. "
            "A termék amit vásároltam: ASUS TUF A16 laptop, az ára 419000 Ft."
        )
        anonymized_message = MessageAnonymizer.anonymize(message)
        extractor = FactExtractor(
            NERModel(pipeline_factory=lambda **kwargs: lambda text: [])
        )

        facts = extractor.extract(anonymized_message)

        self.assertEqual(
            facts,
            {
                "PRODUCT": ["ASUS TUF A16 laptop"],
                "PRICE": ["419000 Ft"],
            },
        )
        self.assertNotIn("Kovács Anna", anonymized_message)
        self.assertNotIn("anna@example.com", anonymized_message)

    def test_extracts_formatted_hungarian_prices_and_deduplicates_facts(self) -> None:
        extractor = FactExtractor(
            NERModel(pipeline_factory=lambda **kwargs: lambda text: [])
        )

        facts = extractor.extract(
            "Termék: ASUS TUF A16; ára 419 000 Ft, kedvezményes ára 419 000 Ft."
        )

        self.assertEqual(facts["PRODUCT"], ["ASUS TUF A16"])
        self.assertEqual(facts["PRICE"], ["419 000 Ft"])

    def test_preprocessing_response_serializes_product_and_price_facts(self) -> None:
        extractor = FactExtractor(
            NERModel(pipeline_factory=lambda **kwargs: lambda text: [])
        )
        builder = StateBuilder(
            cleaner=MessageCleaner.clean,
            anonymizer=MessageAnonymizer.anonymize,
            extractor=extractor.extract,
            classifier=lambda text: "DEFECTIVE_PRODUCT",
        )

        result = PreprocessingResponse.model_validate(
            builder.build(
                "Név: Kovács Anna email: anna@example.com. "
                "A termék amit vásároltam: ASUS TUF A16 laptop, az ára 419000 Ft."
            )
        ).model_dump()

        self.assertEqual(
            result["facts"],
            {
                "PRODUCT": ["ASUS TUF A16 laptop"],
                "PRICE": ["419000 Ft"],
            },
        )
        self.assertNotIn("Kovács Anna", result["message"])
        self.assertNotIn("anna@example.com", result["message"])

    def test_loads_hungarian_ner_pipeline_lazily_once(self) -> None:
        factory_calls: list[dict[str, object]] = []

        def pipeline_factory(**kwargs: object):
            factory_calls.append(kwargs)
            return lambda text: [{"entity_group": "PER", "word": "Kovács Anna"}]

        model = NERModel(pipeline_factory=pipeline_factory)

        self.assertEqual(factory_calls, [])
        first_result = model.extract_entities("Kovács Anna Budapesten él.")
        second_result = model.extract_entities("Kovács Anna dolgozik.")

        self.assertEqual(first_result, [{"entity_group": "PER", "word": "Kovács Anna"}])
        self.assertEqual(second_result, first_result)
        self.assertEqual(
            factory_calls,
            [
                {
                    "task": "ner",
                    "model": NERModel.MODEL_NAME,
                    "aggregation_strategy": "simple",
                }
            ],
        )

    def test_fact_extractor_normalizes_hugging_face_labels_and_deduplicates(self) -> None:
        ner_model = NERModel(
            pipeline_factory=lambda **kwargs: lambda text: [
                {"entity_group": "PER", "word": " Kovács Anna "},
                {"entity_group": "PER", "word": "Kovács Anna"},
                {"entity": "B-LOC", "text": "Budapest"},
                {"label": "ORG", "word": "Példa Kft."},
                {"entity_group": "PER", "word": ""},
                {"word": "Unknown label"},
            ]
        )
        extractor = FactExtractor(ner_model)

        self.assertEqual(
            extractor.extract("Kovács Anna Budapesten dolgozik."),
            {
                "PERSON": ["Kovács Anna"],
                "LOCATION": ["Budapest"],
                "ORGANIZATION": ["Példa Kft."],
            },
        )

    def test_fact_extractor_skips_empty_input_without_calling_ner(self) -> None:
        def fail_if_called(**kwargs: object):
            raise AssertionError("NER pipeline should not load for empty input.")

        extractor = FactExtractor(NERModel(pipeline_factory=fail_if_called))

        self.assertEqual(extractor.extract(" \t\n "), {})


if __name__ == "__main__":
    unittest.main()
