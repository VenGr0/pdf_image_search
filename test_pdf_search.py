import unittest
from pdf_search import extract_text_from_pdf, search_in_text, preprocess_text

class TestPDFSearch(unittest.TestCase):
    def test_extract_text_from_pdf(self):
        # Проверка извлечения текста из PDF
        text = extract_text_from_pdf('1984.pdf')
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_preprocess_text(self):
        # Проверка предварительной обработки текста
        text = "This is a test sentence with stopwords and punctuation!"
        processed_text = preprocess_text(text)
        self.assertNotIn("this", processed_text)  # Проверка удаления стоп-слов
        self.assertNotIn("!", processed_text)     # Проверка удаления пунктуации

    def test_search_in_text(self):
        # Проверка поиска релевантных фрагментов
        text = "This is a test. Another sentence for testing. Search works well."
        query = "test"
        results = search_in_text(text, query)
        self.assertEqual(len(results), 3)  # Проверка количества результатов
        self.assertIn("This is a test.", results)  # Проверка релевантности

if __name__ == "__main__":
    unittest.main()